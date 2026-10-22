import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def build_nudft_matrix(
    coords_in: jnp.ndarray, 
    coords_out: jnp.ndarray, 
    sign: int = -1
) -> jnp.ndarray:
    """
    Construct a NUDFT Operator for Forward or Inverse transforms.
    
    This function generates a kernel matrix that represents a Non-Uniform 
    Discrete Fourier Transform. It is designed to be bi-directional:
    
    1. Forward Transform (Sky -> Visibilities):
       Set sign=-1. Maps spatial coordinates (l, m, n) to baselines (u, v, w).
    
    2. Inverse Transform (Visibilities -> Sky):
       Set sign=+1. Maps baselines (u, v, w) to spatial coordinates (l, m, n).
       Note: For a mathematically exact Inverse DFT, the result may require 
       normalization by (1 / N).

    Parameters
    ----------
    coords_in : jnp.ndarray
        Source coordinates. Shape (N_axes, N_in). 
        e.g., Sky positions (l, m, n) for a forward transform.
    coords_out : jnp.ndarray
        Target coordinates. Shape (N_axes, N_out). 
        e.g., Baselines (u, v, w) for a forward transform.
    sign : int, optional
        Direction of the Fourier phase wrap. Default is -1.
        - -1 : Forward (e^-2πiθ)
        - +1 : Inverse/Reverse (e^+2πiθ)

    Returns
    -------
    nudft_matrix : jnp.ndarray
        A Complex128 matrix of shape (N_out, N_in).
    """
    # Shape Normalization
    # Ensures inputs are (N_axes, N_points).
    c_in = jnp.atleast_2d(jnp.array(coords_in, dtype=jnp.float64))
    c_out = jnp.atleast_2d(jnp.array(coords_out, dtype=jnp.float64))

    # Equivalent to the dot product of coordinate vectors for every pair.
    # (N_out, N_axes) @ (N_axes, N_in) -> (N_out, N_in)
    phase = jnp.matmul(c_out.T, c_in)

    # Complex Exponentiation
    # The 'sign' determines the direction of the phase wrap.
    # 2j * pi * phase converts the geometric distance into a phase angle.
    return jnp.exp(sign * (2.0 * jnp.pi * 1j) * phase)
