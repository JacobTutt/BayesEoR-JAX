"""Arrange JAX devices for distributing rows of the inference system."""

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
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(devices, ("x",))
