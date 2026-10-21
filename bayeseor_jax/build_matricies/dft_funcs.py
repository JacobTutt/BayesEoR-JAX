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


def inverse_udft_1d_rect(
    n_freq: int,
    n_eta: int,
    include_eta0: bool = True
) -> jnp.ndarray:
    """
    Generate the 1D DFT matrix mapping EoR delay modes to frequency.

    Parameters
    ----------
    n_freq : int
        Number of frequency channels.
    n_eta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    include_eta0 : bool
        If True, include eta=0 in the DFT, otherwise exclude it.  Defaults
        to True.

    Returns
    -------
    idft_array : jnp.ndarray of complex floats
        1D DFT matrix with shape `(n_freq, n_eta - (not include_eta0))`.
    """
    # 1. Define coordinate grids as 2D arrays (N_axes=1, N_points)
    # Frequency channel indices (the "output" space for an IDFT)
    index_f = (jnp.arange(n_freq, dtype=jnp.float64) - (n_freq // 2))
    index_eta = (jnp.arange(n_eta, dtype=jnp.float64) - (n_eta // 2))

    # Optionally exclude the DC (eta=0) mode via static slicing.
    # With this indexing convention, eta=0 is always at n_eta//2.
    if not include_eta0:
        eta0_idx = n_eta // 2
        index_eta = jnp.concatenate([index_eta[:eta0_idx], index_eta[eta0_idx + 1 :]])

    # Different normalisations for the eta grid are applied downstream.
    phase = (index_f[:, jnp.newaxis] @ index_eta[jnp.newaxis, :]) / float(n_freq)
    
    # Sign is -1 for an Inverse Transform (Delay -> Frequency)
    idft_array = jnp.exp(-2j * jnp.pi * phase).astype(jnp.complex128)

    return idft_array
