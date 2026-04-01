"""Simple JAX mesh/sharding helpers for BayesEoR-JAX workloads.

This module intentionally keeps a single mesh axis over all devices and shards
only the leading array dimension. That matches both the current matrix-build
usage and the row-sharded JAXMg solver layout used by the marginalized
likelihood.
"""

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


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
