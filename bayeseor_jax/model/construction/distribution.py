"""Partition visibility rows and arrange JAX devices for distributed construction."""

import os
from dataclasses import dataclass

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax.sharding import Mesh


@dataclass(frozen=True)
class CPUResources:
    """Resources visible to one participating construction process.

    Parameters
    ----------
    process_index : int
        This process's zero-based rank in the initialised JAX job.
    process_count : int
        Number of participating JAX processes, not the number of CPU cores.
    device : jax.Device
        Local CPU device on which construction is explicitly placed.
    cpu_count : int
        Available logical CPU count, bounded by affinity and a Slurm
        per-task allocation when available; this is not an XLA thread count.
    """

    process_index: int
    process_count: int
    device: jax.Device
    cpu_count: int


def discover_cpu_resources() -> CPUResources:
    """Inspect the existing JAX job and this process's available CPU allocation.

    Returns
    -------
    resources : CPUResources
        Process rank, job size, local CPU device and allocated logical CPU count.

    Notes
    -----
    This does not launch workers or initialise distributed JAX. The launcher
    must establish participating processes and their CPU affinity beforehand.
    JAX/XLA controls CPU threading; changing thread pools after initialisation
    is not attempted. A CPU device can use multiple host threads and should
    not be interpreted as one physical core.
    """
    # Affinity describes this process's usable CPUs more accurately than node size.
    if hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        count = getattr(os, "process_cpu_count", os.cpu_count)()
        cpu_count = count or 1
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        allocated = int(os.environ["SLURM_CPUS_PER_TASK"])
        if allocated < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be positive")
        cpu_count = min(cpu_count, allocated)

    # Select a host device even if the job's default backend is GPU.
    devices = jax.local_devices(backend="cpu")
    if not devices:
        raise RuntimeError("No local JAX CPU device is available for construction")
    return CPUResources(jax.process_index(), jax.process_count(), devices[0], cpu_count)


def partition_blocks(n_blocks: int, process_count: int, process_index: int) -> tuple[int, int]:
    """Assign a contiguous, non-overlapping block range to one process.

    Parameters
    ----------
    n_blocks : int
        Total number of blocks; zero is permitted.
    process_count : int
        Positive number of participating processes.
    process_index : int
        Zero-based process rank, smaller than process_count.

    Returns
    -------
    start, stop : int
        Half-open block range. Empty assignments have start equal to stop.

    Notes
    -----
    Flattening time and frequency into a single block axis allows multiple
    processes to share work even when there are fewer times than processes.
    Counts differ by at most one block; no padding or overlap is required.
    """
    if n_blocks < 0 or process_count < 1 or not 0 <= process_index < process_count:
        raise ValueError("Invalid block count, process count or process index")
    # Spread remainder blocks over the first ranks, preserving global row ordering.
    base, remainder = divmod(n_blocks, process_count)
    start = process_index * base + min(process_index, remainder)
    return start, start + base + int(process_index < remainder)


def build_solver_mesh() -> Mesh:
    r"""Place all visible JAX devices on a single solver mesh axis.

    Returns
    -------
    jax.sharding.Mesh
        Device mesh with axis name ``("x",)``.

    Notes
    -----
    Uses the currently available JAX devices; no input arguments.

    The returned axis ``x`` is used to distribute rows of the dense
    coefficient-space system across devices for solvers such as ``jaxmg``.
    This defines the device layout only; the caller applies array sharding
    and performs factorisation or solves. Distributed JAX must already be
    initialised to include devices on other hosts.
    """
    # Include all registered devices; distributed initialisation happens upstream.
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(devices, ("x",))


def padded_visibility_partition(
    n_vis: int,
    process_count: int,
    local_device_count: int,
    process_index: int,
) -> tuple[int, int, int]:
    r"""Assign a contiguous range of measured visibility rows to one process.

    Parameters
    ----------
    n_vis : int
        Number of measured complex visibilities before any padding.
    process_count : int
        Number of JAX processes sharing the visibility rows.
    local_device_count : int
        Number of devices on each process across which its rows are sharded.
    process_index : int
        Zero-based index of the process requesting its visibility-row range.

    Returns
    -------
    row_start : int
        First assigned row in the original visibility array.
    row_stop : int
        Exclusive stop, bounded above by n_vis; may precede start for an empty
        assignment.
    padded_row_count : int
        Number of local rows including trailing zeros for even device sharding.

    Notes
    -----
    The cached distributed builders shard the visibility axis across hosts and
    then across local devices. When ``n_vis`` does not divide that layout
    exactly, this returns a process-local stripe length rounded up so each host
    can pad with trailing zero rows. That leaves ``T^H N^{-1} T`` and
    ``T^H N^{-1} d`` unchanged provided the padded rows of ``T`` and the
    corresponding ``N^{-1}`` / ``N^{-1} d`` entries are set to zero.

    Return ``(row_start, row_stop, padded_row_count)``. Start and stop index
    the original visibility vector; the count includes zero-padding required
    for the local devices. For processes beyond the data extent, the slice
    is empty even if the returned start exceeds stop.
    """
    n_vis = int(n_vis)
    process_count = max(1, int(process_count))
    local_device_count = max(1, int(local_device_count))
    process_index = int(process_index)

    # Round up per host, then per local device so every shard has equal length.
    rows_per_process = (n_vis + process_count - 1) // process_count
    rows_per_process = (
        (rows_per_process + local_device_count - 1) // local_device_count
    ) * local_device_count

    # Clip real-data coverage; the remaining local rows are zero-padding.
    row0 = process_index * rows_per_process
    row1 = min(row0 + rows_per_process, n_vis)
    return row0, row1, rows_per_process

