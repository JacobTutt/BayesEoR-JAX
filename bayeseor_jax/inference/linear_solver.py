"""Choose CPU or GPU factorisation without changing the Gaussian target."""

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from jax.sharding import PartitionSpec as P


def build_gaussian_solver(mesh, tile_size=256):
    """Build a solver for Sigma x = dbar and log(det(Sigma)).

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Single CPU device for local inference, or the existing GPU solver mesh.
    tile_size : int, optional
        JAXMG factorisation tile width; ignored on CPU.

    Returns
    -------
    callable
        Takes a positive-definite Hermitian matrix and right-hand side and
        returns the solution and real log determinant from one factorisation.
        The callable can be used inside the JIT-compiled likelihood and MAP.
    """
    platforms = {device.platform for device in mesh.devices.flat}
    if platforms == {"cpu"}:
        if mesh.size != 1:
            raise ValueError("CPU inference requires a single-device solver mesh")

        def solve(matrix, rhs):
            # Reuse one Cholesky factor for both the solve and determinant.
            factor = jnp.linalg.cholesky(matrix)
            solution = cho_solve((factor, True), rhs)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(factor))))
            return solution, logdet

        return solve

    if platforms != {"gpu"}:
        raise ValueError("Inference supports a CPU device or a GPU solver mesh")
    try:
        import jaxmg
    except ImportError as exc:
        raise ImportError(
            "GPU inference requires JAXMG. Use the HPC installation described "
            "in docs/installation.md, or set JAX_PLATFORMS=cpu for local inference."
        ) from exc

    def solve(matrix, rhs):
        return jaxmg.potrs(
            matrix, rhs, T_A=tile_size, mesh=mesh,
            in_specs=(P("x", None),), return_logdet=True,
        )

    return solve
