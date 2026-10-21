"""JAX mesh helper for the row-sharded BayesEoR-JAX solver."""

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jax.sharding import Mesh


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
