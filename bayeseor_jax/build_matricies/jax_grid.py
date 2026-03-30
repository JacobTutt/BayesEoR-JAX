"""Simple JAX mesh/sharding helpers for matrix-build workloads.

This module intentionally keeps a single mesh axis over all devices and shards
only the leading array dimension. That matches the current BuildMatrices usage,
where frequency/time-like batch axes are the natural split dimension.
"""

import math

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def factor_squareish(n_devices: int) -> tuple[int, int]:
    """
    Return a near-square 2D factorization of ``n_devices``.

    Parameters
    ----------
    n_devices : int
        Total number of devices to arrange in a 2D mesh.

    Returns
    -------
    tuple of int
        ``(n_rows, n_cols)`` mesh shape with factors chosen to be as square
        as possible.
    """
    if n_devices < 1:
        raise ValueError("n_devices must be >= 1")

    best_rows = 1
    best_cols = n_devices
    best_gap = n_devices - 1
    for rows in range(1, int(math.isqrt(n_devices)) + 1):
        if n_devices % rows != 0:
            continue
        cols = n_devices // rows
        gap = abs(cols - rows)
        if gap < best_gap:
            best_rows = rows
            best_cols = cols
            best_gap = gap
    return best_rows, best_cols


def build_likelihood_mesh() -> Mesh:
    """
    Build a square-ish 2D mesh for dense likelihood matrices.

    Returns
    -------
    jax.sharding.Mesh
        Device mesh with axis names ``("row", "col")``.
    """
    devices = np.asarray(jax.devices(), dtype=object)
    n_rows, n_cols = factor_squareish(int(devices.size))
    return Mesh(devices.reshape(n_rows, n_cols), ("row", "col"))


def build_solver_mesh() -> Mesh:
    """
    Build a 1D mesh for row-sharded solver backends such as ``jaxmg``.

    Returns
    -------
    jax.sharding.Mesh
        Device mesh with axis name ``("x",)``.
    """
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(devices, ("x",))


class MeshManager:
    """Create a one-axis mesh and convenience sharding specs.

    Notes
    -----
    - Mesh axis name is fixed to ``"data"``.
    - Sharding policy is intentionally simple: shard only axis 0.
    """

    def __init__(self) -> None:
        self.devices = tuple(jax.devices())
        self.n_dev = len(self.devices)
        if self.n_dev < 1:
            raise ValueError("No JAX devices available to build a mesh.")

        self.axis_names = ("data",)
        self.mesh = Mesh(np.asarray(self.devices), self.axis_names)

    def get_sharding_for_spec(self, *parts) -> NamedSharding:
        """Return a sharding for an explicit ``PartitionSpec`` layout."""
        return NamedSharding(self.mesh, P(*parts))

    def get_sharding(self, rank: int = 2) -> NamedSharding:
        """Return a sharding that partitions only the leading axis.

        Parameters
        ----------
        rank : int, optional
            Rank of the target array.
        """
        if rank < 1:
            raise ValueError("rank must be >= 1")

        spec = P("data", *([None] * (rank - 1)))
        return NamedSharding(self.mesh, spec)

    def get_replicated_sharding(self, rank: int = 2) -> NamedSharding:
        """Return a replicated sharding over the current mesh."""
        if rank < 1:
            raise ValueError("rank must be >= 1")

        spec = P(*([None] * rank))
        return NamedSharding(self.mesh, spec)
