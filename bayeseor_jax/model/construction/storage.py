"""Shared-file publication and dependency-aware cache reuse for construction."""

from pathlib import Path
from uuid import uuid4
from collections.abc import Callable, Sequence
import json
import time

import jax
import numpy as np
from jax.experimental import multihost_utils


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


def publish_output(partial_path: Path, path: Path) -> None:
    """Publish a completed matrix only after every participating process has finished.

    Parameters
    ----------
    partial_path : pathlib.Path
        Temporary NumPy file, already flushed and closed by each writer.
    path : pathlib.Path
        Final output path. An existing file is replaced on publication.

    Returns
    -------
    None
        Renames the completed file and removes its allocation marker.
    """
    distributed = jax.distributed.is_initialized()
    # Prevent readers from observing a partially written operator as a cache hit.
    if distributed:
        multihost_utils.sync_global_devices(f"{path.name}-write-complete")
    if jax.process_index() == 0:
        partial_path.replace(path)
        partial_path.with_suffix(partial_path.suffix + ".ready").unlink(missing_ok=True)
    if distributed:
        multihost_utils.sync_global_devices(f"{path.name}-published")


def save_array(path: Path | str, array) -> Path:
    """Publish a small replicated array or scalar from process zero.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination NumPy file, shared by all participating processes.
    array : array-like
        Host array or replicated JAX array. Distributed non-replicated arrays
        must instead be written shard-by-shard using a shared memmap.

    Returns
    -------
    pathlib.Path
        Completed file, published only after its temporary write finishes.
    """
    path = Path(path)
    if isinstance(array, jax.Array) and not array.is_fully_replicated:
        raise ValueError("save_array requires a replicated array")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")
    if jax.process_index() == 0:
        # A failed write must not replace an existing completed product.
        with partial_path.open("wb") as handle:
            np.save(handle, np.asarray(array))
    publish_output(partial_path, path)
    return path


def _file_signature(path: Path | str) -> dict:
    """Identify a saved input by its resolved path, byte size and modification time."""
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def ensure_cached(
    path: Path | str,
    build: Callable[[], object],
    *,
    dependencies: Sequence[Path | str] = (),
    clobber: bool = False,
) -> Path:
    """Reuse a completed product or collectively build it from explicit inputs.

    Parameters
    ----------
    path : pathlib.Path or str
        Shared destination file.
    build : callable
        No-argument callback that builds and publishes path on all processes.
    dependencies : sequence of pathlib.Path or str, optional
        Saved inputs whose paths, sizes and modification times define reuse.
        Selecting a different foreground representation therefore invalidates
        a cached joint matrix even if both representations have the same shape.
    clobber : bool, optional
        Force a rebuild without deleting the previous completed output first.

    Returns
    -------
    pathlib.Path
        Existing or newly completed product.

    Notes
    -----
    All participating processes must call together. Physical component files
    without dependencies are reused by filename; use separate directories or
    clobber when changing their observation/model configuration. File stamps
    are not content hashes. Data projections are rebuilt by setup rather than
    cached by dataset name. No downstream files are silently deleted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = [_file_signature(item) for item in dependencies]
    metadata_path = path.with_suffix(path.suffix + ".cache.json")
    distributed = jax.distributed.is_initialized()
    decision_path = path.with_suffix(path.suffix + ".exists")
    reuse = False
    if jax.process_index() == 0:
        reuse = path.exists() and not clobber
        if reuse and inputs:
            try:
                stored = json.loads(metadata_path.read_text())
                reuse = stored == {"inputs": inputs, "output": _file_signature(path)}
            except (OSError, ValueError):
                reuse = False
        if distributed:
            decision_path.write_text("1" if reuse else "0")
    if distributed:
        multihost_utils.sync_global_devices(f"{path.name}-cache-ready")
        _wait_for_file(decision_path)
        reuse = decision_path.read_text().strip() == "1"
        multihost_utils.sync_global_devices(f"{path.name}-cache-read")
    if reuse:
        return path

    # Call the same writer on every rank; it owns temporary-file publication.
    build()
    if not path.exists():
        raise RuntimeError(f"Builder did not publish {path}")
    if jax.process_index() == 0:
        metadata = {"inputs": inputs, "output": _file_signature(path)}
        temporary = metadata_path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
    if distributed:
        multihost_utils.sync_global_devices(f"{path.name}-cache-saved")
    return path
