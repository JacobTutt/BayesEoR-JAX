"""Helper utilities for matrix path, save, and load workflows."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils


def _dense_matrix_path(bm, matrix_name: str) -> Path:
    """Return the `.npy` path for a stored dense matrix."""
    return Path(bm.array_save_directory) / f"{matrix_name}.npy"


def get_T_component_layout(bm) -> dict[str, dict[str, int]]:
    """Return the direct-T column layout for EoR and foreground components."""
    n_non_mp = int(bm.n_uv_fg - int(bm.fit_for_monopole))
    eor_width = int(bm.n_uv_eor * (bm.n_eta - 1))
    fg_eta0_width = int(n_non_mp)
    fg_lssm_width = int(n_non_mp * bm.n_modes_fg)
    fg_mp_width = int((bm.n_eta + bm.n_modes_fg) if bm.fit_for_monopole else 0)

    col = 0
    layout: dict[str, dict[str, int]] = {}
    for name, width in (
        ("eor", eor_width),
        ("fg_eta0", fg_eta0_width),
        ("fg_lssm", fg_lssm_width),
        ("fg_monopole", fg_mp_width),
    ):
        layout[name] = {
            "col_start": col,
            "col_stop": col + width,
        }
        col += width
    return layout


def check_if_matrix_exists(bm, matrix_name: str) -> bool:
    """Return True if a saved dense matrix exists on disk."""
    return _dense_matrix_path(bm, matrix_name).exists()


def _wait_for_file(path: Path, timeout_s: float = 120.0) -> None:
    """Poll until a file becomes visible on the shared filesystem."""
    deadline = time.time() + timeout_s
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.2)


def _prepare_memmap_output(
    bm,
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    sync_name: str,
) -> np.memmap:
    """
    Create or attach to a shared `.npy` memmap output.

    Process 0 creates the file and publishes a ready token. Other processes
    attach after a distributed barrier or, for non-distributed runs, by
    polling the ready token.
    """
    ready_path = path.with_suffix(path.suffix + ".ready")
    token = uuid4().hex

    if jax.process_index() == 0:
        if ready_path.exists():
            ready_path.unlink()
        mm = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        del mm
        ready_path.write_text(token)

    if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
        multihost_utils.sync_global_devices(sync_name)
    else:
        _wait_for_file(ready_path)
        if ready_path.read_text().strip() != token:
            raise RuntimeError(f"Unexpected ready token in {ready_path}")

    return np.load(path, mmap_mode="r+")


def save_dense_matrix(bm, matrix_name: str, matrix: jnp.ndarray) -> None:
    """
    Save a dense matrix to disk as `.npy`.

    Distributed arrays are written shard-by-shard into a shared memmap so
    that no process needs to gather the full global array.
    """
    path = _dense_matrix_path(bm, matrix_name)
    if not hasattr(matrix, "addressable_shards"):
        np.save(path, np.asarray(matrix))
        return

    mm = _prepare_memmap_output(
        bm,
        path,
        shape=tuple(int(dim) for dim in matrix.shape),
        dtype=np.dtype(matrix.dtype),
        sync_name=f"{matrix_name}-save-ready",
    )
    for shard in matrix.addressable_shards:
        mm[shard.index] = np.asarray(shard.data)
    mm.flush()
    if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
        multihost_utils.sync_global_devices(f"{matrix_name}-save-done")


def save_matrix(bm, matrix_name: str, matrix) -> None:
    """Save a matrix to disk as `.npy`."""
    save_dense_matrix(bm, matrix_name, matrix)


def load_dense_matrix(bm, matrix_name: str) -> jnp.ndarray:
    """Load a dense matrix from disk."""
    path = _dense_matrix_path(bm, matrix_name)
    return jnp.array(np.load(path))


def load_matrix(bm, matrix_name: str):
    """Load a saved matrix from disk."""
    if _dense_matrix_path(bm, matrix_name).exists():
        return load_dense_matrix(bm, matrix_name)
    raise FileNotFoundError(f"No saved matrix found for '{matrix_name}'.")
