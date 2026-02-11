import jax
jax.config.update("jax_enable_x128", True)
import jax.numpy as jnp
import numpy as np

from typing import Optional, Sequence, Union


from .lssm_model_space import build_lssm_basis

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


def reorder_fuv_to_uvf(
    n_uv: int,
    n_freq: int,
) -> jnp.ndarray:
    """
    Build the visibility-ordered -> channel-ordered gridding matrix.

    This operator reorders a vector arranged by UV-pixel blocks into one
    arranged by frequency blocks:

    Vis-ordered (UV-major):
        [ (u0,v0,f0..fN), (u1,v1,f0..fN), ... ]

    Chan-ordered (Freq-major):
        [ (f0, all uv), (f1, all uv), ... ]

    In matrix terms, this is a pure permutation matrix with exactly one
    `1` per row and column.

    Parameters
    ----------
    n_uv : int
        Number of UV modes.
    n_freq : int
        Number of frequency channels.

    Returns
    -------
    gridding_matrix : jnp.ndarray
        Mapping matrix of shape (n_uv*n_freq, n_uv*n_freq).
    """
    # Row/column index construction without Python loops.
    # Row index: (freq-major) linear index = f * n_uv + uv
    chan = jnp.arange(n_freq, dtype=jnp.int32)[:, jnp.newaxis]  # (n_freq, 1)
    vis = jnp.arange(n_uv, dtype=jnp.int32)[jnp.newaxis, :]     # (1, n_uv)

    rows = (chan * n_uv + vis).reshape(-1)                     # (n_freq*n_uv,)
    # Column index: (uv-major) linear index = uv * n_freq + f
    cols = (chan + vis * n_freq).reshape(-1)                   # (n_freq*n_uv,)

    matrix_dims = (n_uv * n_freq, n_uv * n_freq)
    return jnp.zeros(matrix_dims, dtype=jnp.float32).at[rows, cols].set(1.0)



def inverse_udft_1d_rect(
    n_freq: int,
    n_eta: int,
    n_modes: int = 0,
    n_powerlaws: Optional[int] = None,
    beta: Optional[Union[float, Sequence[float], jnp.ndarray]] = None,
    f0_mhz: Optional[float] = None,
    df_mhz: Optional[float] = None,
    include_eta0: bool = True
) -> jnp.ndarray:
    """
    Generate a 1D Uniform DFT matrix transforming delay to frequency space as well as
    optional large spectral scale model (LSSM) basis vectors.

    Used in the construction of `Fz` for the:
    - Rectilinear EoR Model
    - Rectilinear Foreground Model
    
    For the subharmonic grid, see `inverse_udft_1d_sub`.

    Parameters
    ----------
    n_freq : int
        Number of frequency channels.
    n_eta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    n_modes : int
        Number of quadratic Large Spectral Scale Model (LSSM) basis vectors.
    n_powerlaws : int
        Number of power law LSSM basis vectors.  Overrides nq, i.e. replaces
        npl <= nq quadratic basis vectors with power law basis vectors.
    beta : float or array-like
        Spectral index(indices) for the `npl` power law LSSM basis vectors.
    f0_mhz : float
        Minimum frequency in megahertz.
    df_mhz : float
        Frequency channel width in megahertz.
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

    # Optionally exclude the DC (eta=0) mode
    if not include_eta0:
        index_eta = index_eta[index_eta != 0.0]

    # Define the 1D IDFT matrix for the Fourier modes. 
    # As different normalisations are applied to the:
    # LSSM, full eta grid and central eta=0 mode - these are applied downstream 
    phase = (index_f[:, jnp.newaxis] @ index_eta[jnp.newaxis, :]) / float(n_freq)
    
    # Sign is -1 for an Inverse Transform (Delay -> Frequency)
    idft_array = jnp.exp(-2j * jnp.pi * phase).astype(jnp.complex128)

    # Handle Large Spectral Scale Model (LSSM)
    if n_modes > 0:
        n_powerlaws = 0 if n_powerlaws is None else n_powerlaws
        lssm_basis_vectors = build_lssm_basis(
            n_freq=n_freq,
            n_modes=n_modes,
            n_powerlaws=n_powerlaws,
            f0_mhz=f0_mhz,
            df_mhz=df_mhz,
            beta=beta,
        ).astype(jnp.complex128)
        # Combine the Fourier basis and LSSM basis
        idft_array = jnp.hstack((idft_array, lssm_basis_vectors))

    return idft_array
