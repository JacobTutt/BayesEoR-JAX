"""Simple JAX mesh/sharding helpers for matrix-build workloads.

This module intentionally keeps a single mesh axis over all devices and shards
only the leading array dimension. That matches the current BuildMatrices usage,
where frequency/time-like batch axes are the natural split dimension.
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


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
