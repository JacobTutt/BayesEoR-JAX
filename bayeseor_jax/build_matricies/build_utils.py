"""Helper utilities for dense matrix path, save, and load workflows."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils


def _dense_matrix_path(bm, matrix_name: str) -> Path:
    """
    Return the output path for a dense matrix saved as ``.npy``.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory.
    matrix_name : str
        Logical matrix name, e.g. ``"T"`` or ``"Ninv"``.

    Returns
    -------
    pathlib.Path
        Full path to the `.npy` file on disk.
    """
    return Path(bm.array_save_directory) / f"{matrix_name}.npy"


def _log_build(bm, message: str) -> None:
    """
    Emit a build/save progress message from process 0.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the build.
    message : str
        Message to print.
    """
    if jax.process_index() == 0:
        print(f"[BuildMatrices] {message}", flush=True)


def get_T_component_layout(bm) -> dict[str, dict[str, int]]:
    """
    Return the direct-``T`` column layout for EoR and foreground components.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance supplying the UV counts and basis dimensions.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapping from component name to ``{"col_start", "col_stop"}`` slices in
        the legacy direct-``T`` column ordering.
    """
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
    """
    Return ``True`` when a saved dense matrix exists on disk.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory.
    matrix_name : str
        Logical matrix name, e.g. ``"T"`` or ``"Ninv"``.
    """
    path = _dense_matrix_path(bm, matrix_name)
    exists = path.exists()
    if getattr(jax, "distributed", None) is None or not jax.distributed.is_initialized():
        return exists

    decision_path = path.with_suffix(path.suffix + ".exists")
    if jax.process_index() == 0:
        decision_path.write_text("1" if exists else "0")

    multihost_utils.sync_global_devices(f"{matrix_name}-exists-ready")
    _wait_for_file(decision_path)
    exists = decision_path.read_text().strip() == "1"
    multihost_utils.sync_global_devices(f"{matrix_name}-exists-read")
    return exists


def _wait_for_file(path: Path, timeout_s: float = 120.0) -> None:
    """
    Poll until a file becomes visible on the shared filesystem.

    Parameters
    ----------
    path : pathlib.Path
        File expected to appear.
    timeout_s : float, optional
        Maximum wait time in seconds. Defaults to ``120``.
    """
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

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance coordinating the distributed write.
    path : pathlib.Path
        Output file path.
    shape : tuple of int
        Global dense matrix shape.
    dtype : numpy.dtype
        Output element dtype.
    sync_name : str
        Barrier name used when distributed JAX is initialized.

    Returns
    -------
    numpy.memmap
        Writable memmap opened in ``r+`` mode.

    Notes
    -----
    Process 0 creates the file and publishes a ready token. Other processes
    attach after a distributed barrier or, for non-distributed runs, by
    polling the ready token.
    """
    ready_path = path.with_suffix(path.suffix + ".ready")
    token = uuid4().hex

    if jax.process_index() == 0:
        if ready_path.exists():
            ready_path.unlink()
        # Create the final `.npy` container once so every process can attach to
        # the same file without racing on the header write.
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


def _slice_bounds(sl: slice, extent: int) -> tuple[int, int]:
    """
    Convert a possibly open-ended slice into concrete local bounds.

    Parameters
    ----------
    sl : slice
        Slice object, potentially containing ``None`` bounds.
    extent : int
        Full length of the axis that the slice indexes.

    Returns
    -------
    tuple of int
        Concrete ``(start, stop)`` bounds within ``[0, extent]``.
    """
    start = 0 if sl.start is None else int(sl.start)
    stop = extent if sl.stop is None else int(sl.stop)
    return start, stop


def save_dense_matrix(bm, matrix_name: str, matrix: jnp.ndarray) -> None:
    """
    Save a dense matrix to disk as `.npy`.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory and distributed state.
    matrix_name : str
        Logical matrix name, e.g. ``"T"``.
    matrix : jax.Array or numpy.ndarray
        Dense matrix to persist.

    Notes
    -----
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
        # Each process writes only the global slices for the shards it owns.
        mm[shard.index] = np.asarray(shard.data)
    mm.flush()
    if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
        multihost_utils.sync_global_devices(f"{matrix_name}-save-done")


def save_matrix(bm, matrix_name: str, matrix) -> None:
    """
    Save a matrix to disk as `.npy`.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory.
    matrix_name : str
        Logical matrix name.
    matrix : jax.Array or numpy.ndarray
        Matrix to persist.
    """
    save_dense_matrix(bm, matrix_name, matrix)


def load_dense_matrix(bm, matrix_name: str) -> jnp.ndarray:
    """
    Load a dense matrix from disk.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory.
    matrix_name : str
        Logical matrix name.

    Returns
    -------
    jax.numpy.ndarray
        Dense matrix loaded from the saved `.npy` file.
    """
    path = _dense_matrix_path(bm, matrix_name)
    return jnp.array(np.load(path))


def load_matrix(bm, matrix_name: str):
    """
    Load a saved dense matrix from disk.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory.
    matrix_name : str
        Logical matrix name.

    Returns
    -------
    jax.numpy.ndarray
        Dense matrix loaded from disk.
    """
    if _dense_matrix_path(bm, matrix_name).exists():
        return load_dense_matrix(bm, matrix_name)
    raise FileNotFoundError(f"No saved matrix found for '{matrix_name}'.")
