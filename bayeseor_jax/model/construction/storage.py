"""Store forward-model arrays and coordinate access across JAX processes.

These helpers manage paths and shared memory-mapped files. They preserve the operator ordering supplied by the builders and
perform no transformation of the sky model or observational coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils


def _dense_matrix_path(bm, matrix_name: str) -> Path:
    r"""Resolve a cached model array's name to its NumPy file in the build directory.

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

    Notes
    -----
    Names such as ``T``, ``Ninv``, and ``dbar`` identify matrices, vectors,
    or scalars. This helper returns a path without creating or loading a file.
    """
    return Path(bm.array_save_directory) / f"{matrix_name}.npy"


def _log_build(bm, message: str) -> None:
    r"""Report construction or saving progress once across the distributed build.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the build.
    message : str
        Message to print.

    Returns
    -------
    None
        The message is printed on process zero.

    Notes
    -----
    Only process zero prints the message, avoiding duplicate output when
    multiple processes construct different portions of the same operator.
    """
    if jax.process_index() == 0:
        print(f"[BuildMatrices] {message}", flush=True)


def check_if_matrix_exists(bm, matrix_name: str) -> bool:
    r"""Decide consistently across processes whether a cached array file exists.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory.
    matrix_name : str
        Logical matrix name, e.g. ``"T"`` or ``"Ninv"``.

    Returns
    -------
    exists : bool
        Whether the saved array exists, using the same decision on every
        process.

    Notes
    -----
    In distributed runs, process zero publishes its decision on the shared
    filesystem and all processes read it before building or reusing a product.
    Only existence is checked, not array contents or physical configuration.
    """
    path = _dense_matrix_path(bm, matrix_name)
    exists = path.exists()
    if getattr(jax, "distributed", None) is None or not jax.distributed.is_initialized():
        return exists

    # Publish one cache decision so processes cannot take different build paths.
    decision_path = path.with_suffix(path.suffix + ".exists")
    if jax.process_index() == 0:
        decision_path.write_text("1" if exists else "0")

    # All readers must observe that decision before any process begins a rebuild.
    multihost_utils.sync_global_devices(f"{matrix_name}-exists-ready")
    _wait_for_file(decision_path)
    exists = decision_path.read_text().strip() == "1"
    multihost_utils.sync_global_devices(f"{matrix_name}-exists-read")
    return exists


def _wait_for_file(path: Path, timeout_s: float = 120.0) -> None:
    r"""Wait for another process's file to become visible on shared storage.

    Parameters
    ----------
    path : pathlib.Path
        File expected to appear.
    timeout_s : float, optional
        Maximum wait time in seconds. Defaults to ``120``.

    Returns
    -------
    None
        Returns when the path is visible; raises TimeoutError on timeout.

    Notes
    -----
    Used for coordination files during distributed construction. Raise
    ``TimeoutError`` if the path remains absent after the allowed interval.
    """
    # Bound filesystem polling so a missing producer cannot hang the build forever.
    deadline = time.time() + timeout_s
    while not path.exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.2)


def _prepare_memmap_output(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    sync_name: str,
) -> np.memmap:
    r"""Allocate a shared NumPy array on disk for processes to fill in place.

    Parameters
    ----------
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
    Process zero creates the file header and shape; every process then opens
    the same array for writing its assigned slices. The caller determines
    which slices belong to each process and when the completed output can
    be published. An existing file at this path is overwritten.

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
    r"""Resolve the start and stop of an array shard's axis slice.

    Parameters
    ----------
    sl : slice
        Slice object, potentially containing ``None`` bounds.
    extent : int
        Full length of the axis that the slice indexes.

    Returns
    -------
    tuple of int
        Explicit ``(start, stop)`` bounds, with stop exclusive.

    Notes
    -----
    Replace missing bounds by zero and the full axis length so a shard can
    be matched to its disk-backed array region. This helper assumes the
    unit-stride, in-range slices supplied by the sharding layout; it does
    not clamp arbitrary indices or interpret a slice step.
    """
    start = 0 if sl.start is None else int(sl.start)
    stop = extent if sl.stop is None else int(sl.stop)
    return start, stop


def save_matrix(bm, matrix_name: str, matrix: jnp.ndarray) -> None:
    r"""Save a model matrix, vector, or scalar without changing its axis ordering.

    Parameters
    ----------
    bm : BuildMatrices
        Matrix builder instance owning the save directory and distributed state.
    matrix_name : str
        Logical matrix name, e.g. ``"T"``.
    matrix : jax.Array or numpy.ndarray
        Dense matrix to persist.

    Returns
    -------
    None
        The array is written to the configured save directory.

    Notes
    -----
    Ordinary arrays are written directly. For JAX arrays, each process
    writes its addressable shards into the shared NumPy file and flushes
    them before the final distributed barrier.

    Distributed arrays are written shard-by-shard into a shared memmap so
    that no process needs to gather the full global array.
    """
    path = _dense_matrix_path(bm, matrix_name)
    if not hasattr(matrix, "addressable_shards"):
        np.save(path, np.asarray(matrix))
        return

    mm = _prepare_memmap_output(
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


def load_matrix(bm, matrix_name: str):
    r"""Load a complete cached model array into a JAX array.

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

    Notes
    -----
    This reads the full NumPy file into memory; it does not stream blocks or
    distribute the array across devices. Raise ``FileNotFoundError`` when
    the requested product has not been saved.
    """
    path = _dense_matrix_path(bm, matrix_name)
    if path.exists():
        return jnp.array(np.load(path))
    raise FileNotFoundError(f"No saved matrix found for '{matrix_name}'.")
