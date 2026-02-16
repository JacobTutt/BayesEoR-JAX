"""
Utilities for constructing a JAX device mesh and sharding specs.

This module provides a MeshManager that builds a device mesh based on the
available hardware and returns NamedSharding objects for common 1D/2D layouts.
"""

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

class MeshManager:
    """
    Helper class to build a JAX device mesh and NamedSharding specs.

    The mesh layout is chosen from the available device count:
      - If n_dev is factorable into a non-trivial 2D grid, use
        (major, minor) with a near-square factorization.
      - Otherwise (e.g. prime counts), use a 1D pixels-only mesh.

    Attributes
    ----------
    devices : list
        List of available JAX devices.
    n_dev : int
        Number of devices.
    mesh : Mesh
        JAX Mesh built from the device grid.
    axis_names : tuple
        Axis names used in the mesh (e.g., ("major",) or ("major", "minor")).
    """
    def __init__(self):
        """
        Initialize the MeshManager by discovering devices and building the mesh.
        """
        self.devices = jax.devices()
        self.n_dev = len(self.devices)
        self.mesh, self.axis_names = self._create_mesh()

    def _create_mesh(self):
        """
        Build the device mesh based on the number of devices.

        Returns
        -------
        mesh : Mesh
            JAX Mesh object with the chosen grid shape.
        axis_names : tuple
            Axis names used by the mesh.
        """
        if self.n_dev < 1:
            raise ValueError("No JAX devices available to build a mesh.")

        # Find a near-square factorization n_dev = a * b with b > 1.
        # This supports odd composite counts (e.g. 9 -> 3x3, 15 -> 5x3).
        pair = None
        root = int(np.sqrt(self.n_dev))
        for b in range(root, 1, -1):
            if self.n_dev % b == 0:
                a = self.n_dev // b
                pair = (a, b)
                break

        if pair is None:
            # Prime device counts fall back to 1D sharding.
            shape, names = (self.n_dev,), ("major",)
        else:
            shape, names = pair, ("major", "minor")

        # Arrange devices into the requested grid shape.
        device_grid = np.array(self.devices).reshape(shape)
        return Mesh(device_grid, axis_names=names), names

    def get_sharding(
        self,
        rank: int = 2,
        shard_minor: bool = False,
        shard_uv: bool = False,
    ) -> NamedSharding:
        """
        Return a NamedSharding matching the requested array rank.

        Parameters
        ----------
        rank : int, optional
            Rank (number of dimensions) of the array to shard. Defaults to 2.
        shard_minor : bool, optional
            If True and the mesh has a 'minor' axis, shard the second axis by
            minor. Otherwise only shard the first axis. Defaults to False.
        shard_uv : bool, optional
            Backward-compatible alias for ``shard_minor``.

        Returns
        -------
        sharding : NamedSharding
            Sharding spec compatible with the requested rank.
        """
        shard_minor = shard_minor or shard_uv
        has_minor = "minor" in self.axis_names and shard_minor

        if rank == 1:
            return NamedSharding(self.mesh, P("major"))
        if rank == 2:
            return NamedSharding(self.mesh, P("major", "minor" if has_minor else None))
        if rank == 3:
            return NamedSharding(self.mesh, P("major", None, None))
        if rank == 4:
            return NamedSharding(self.mesh, P("major", None, None, None))

        # Fallback: shard the leading axis only, leave others unsharded.
        return NamedSharding(self.mesh, P("major", *([None] * (rank - 1))))
