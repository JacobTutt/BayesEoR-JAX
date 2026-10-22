"""Partition visibility rows and arrange JAX devices for distributed construction."""

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax.sharding import Mesh


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

