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

    The mesh layout is chosen based on the number of devices:
      - 4  -> (4,)      : pixels only
      - 8  -> (8,)      : pixels only
      - 12 -> (6, 2)    : pixels x uv_modes
      - 16 -> (8, 2)    : pixels x uv_modes
      - otherwise -> (n_dev,) : pixels only

    Attributes
    ----------
    devices : list
        List of available JAX devices.
    n_dev : int
        Number of devices.
    mesh : Mesh
        JAX Mesh built from the device grid.
    axis_names : tuple
        Axis names used in the mesh (e.g., ("pixels",) or ("pixels", "uv_modes")).
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
        if self.n_dev == 4:
            shape, names = (4,), ("pixels",)
        elif self.n_dev == 8:
            shape, names = (8,), ("pixels",)
        elif self.n_dev == 12:
            shape, names = (6, 2), ("pixels", "uv_modes")
        elif self.n_dev == 16:
            shape, names = (8, 2), ("pixels", "uv_modes")
        else:
            # Fallback for laptop or unexpected counts.
            shape, names = (self.n_dev,), ("pixels",)
        
        # Arrange devices into the requested grid shape.
        device_grid = np.array(self.devices).reshape(shape)
        return Mesh(device_grid, axis_names=names), names

    def get_sharding(self, rank: int = 2, shard_uv: bool = False) -> NamedSharding:
        """
        Return a NamedSharding matching the requested array rank.

        Parameters
        ----------
        rank : int, optional
            Rank (number of dimensions) of the array to shard. Defaults to 2.
        shard_uv : bool, optional
            If True and the mesh has a 'uv_modes' axis, shard the second axis
            by uv_modes. Otherwise only shard the first axis. Defaults to False.

        Returns
        -------
        sharding : NamedSharding
            Sharding spec compatible with the requested rank.
        """
        has_uv = "uv_modes" in self.axis_names and shard_uv

        if rank == 1:
            return NamedSharding(self.mesh, P("pixels"))
        if rank == 2:
            return NamedSharding(self.mesh, P("pixels", "uv_modes" if has_uv else None))
        if rank == 3:
            return NamedSharding(self.mesh, P("pixels", None, None))
        if rank == 4:
            return NamedSharding(self.mesh, P("pixels", None, None, None))

        # Fallback: shard the leading axis only, leave others unsharded.
        return NamedSharding(self.mesh, P("pixels", *([None] * (rank - 1))))
