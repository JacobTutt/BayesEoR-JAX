"""Linear operators connecting the model, image, and visibility spaces."""

from __future__ import annotations

from collections.abc import Sequence

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def build_nudft_matrix(
    coords_in: jnp.ndarray,
    coords_out: jnp.ndarray,
    sign: int = -1,
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


def build_transverse_fourier_operator(
    uv: jax.Array,
    lm: jax.Array,
    uv_cell_area: float,
) -> jax.Array:
    r"""Build :math:`F'`, mapping transverse Fourier modes to image pixels."""
    if uv_cell_area <= 0:
        raise ValueError("uv_cell_area must be positive")
    return build_nudft_matrix(uv, lm, sign=-1) * uv_cell_area


def build_spectral_fourier_operator(
    eta: jax.Array,
    frequency_offsets_hz: jax.Array,
    d_eta: float,
) -> jax.Array:
    r"""Build :math:`F_z`, mapping line-of-sight modes to frequency."""
    if d_eta <= 0:
        raise ValueError("d_eta must be positive")
    eta = jnp.asarray(eta, dtype=jnp.float64)
    frequency_offsets_hz = jnp.asarray(
        frequency_offsets_hz,
        dtype=jnp.float64,
    )
    if eta.ndim != 1:
        raise ValueError("eta must be one-dimensional")
    if frequency_offsets_hz.ndim != 1:
        raise ValueError("frequency_offsets_hz must be one-dimensional")
    return (
        build_nudft_matrix(
            eta[jnp.newaxis, :],
            frequency_offsets_hz[jnp.newaxis, :],
            sign=-1,
        )
        * d_eta
    ).astype(jnp.complex128)


def build_sky_to_visibility_operator(
    lmn: jax.Array,
    uvw_lambda: jax.Array,
    beam: jax.Array,
    pixel_area_sr: float,
) -> jax.Array:
    r"""Build :math:`R=F^{-1}P` for a block of time integrations.

    Parameters
    ----------
    lmn
        Topocentric sky coordinates with shape ``(n_time, 3, n_pixel)``.
    uvw_lambda
        Baselines in observing wavelengths with shape
        ``(n_frequency, 3, n_baseline)``. BayesEoR follows the PyUVData
        ``ant2 - ant1`` baseline convention, hence the positive phase used
        below is equivalent to the negative phase in the paper's
        ``r_i - r_j`` convention.
    beam
        Horizon-limited primary-beam power response with shape
        ``(n_time, n_frequency, n_pixel)``. Pixels at or below the horizon
        must already have zero gain.
    pixel_area_sr
        Solid angle of one equal-area sky pixel in steradians.

    Returns
    -------
    jax.Array
        Beam-weighted sky-to-visibility response with shape
        ``(n_time, n_frequency, n_baseline, n_pixel)``.
    """
    lmn = jnp.asarray(lmn, dtype=jnp.float64)
    uvw_lambda = jnp.asarray(uvw_lambda, dtype=jnp.float64)
    beam = jnp.asarray(beam)

    if lmn.ndim != 3 or lmn.shape[1] != 3:
        raise ValueError("lmn must have shape (n_time, 3, n_pixel)")
    if uvw_lambda.ndim != 3 or uvw_lambda.shape[1] != 3:
        raise ValueError(
            "uvw_lambda must have shape (n_frequency, 3, n_baseline)"
        )
    expected_beam_shape = (lmn.shape[0], uvw_lambda.shape[0], lmn.shape[2])
    if beam.shape != expected_beam_shape:
        raise ValueError(
            f"beam must have shape {expected_beam_shape}, got {beam.shape}"
        )

    phase = jnp.einsum("fcb,tcp->tfbp", uvw_lambda, lmn)
    fringe = jnp.exp(2j * jnp.pi * phase)
    return (
        jnp.asarray(pixel_area_sr, dtype=jnp.float64)
        * fringe
        * beam[:, :, None, :]
    )


def build_T_21_core(
    sky_to_visibility: jax.Array,
    transverse_operator: jax.Array,
    spectral_operator: jax.Array,
) -> jax.Array:
    r"""Build :math:`T_{21}=R F' F_z` for one visibility-row block."""
    sky_to_visibility = jnp.asarray(sky_to_visibility)
    transverse_operator = jnp.asarray(transverse_operator)
    spectral_operator = jnp.asarray(spectral_operator)

    if sky_to_visibility.ndim != 4:
        raise ValueError(
            "sky_to_visibility must have shape "
            "(n_time, n_frequency, n_baseline, n_pixel)"
        )
    n_time, n_frequency, n_baseline, n_pixel = sky_to_visibility.shape
    if (
        transverse_operator.ndim != 2
        or transverse_operator.shape[0] != n_pixel
    ):
        raise ValueError(
            "transverse_operator must have shape (n_pixel, n_uv_mode)"
        )
    if (
        spectral_operator.ndim != 2
        or spectral_operator.shape[0] != n_frequency
    ):
        raise ValueError(
            "spectral_operator must have shape (n_frequency, n_eta_mode)"
        )

    transverse_response = jnp.einsum(
        "tfbp,pu->tfbu",
        sky_to_visibility,
        transverse_operator,
    )
    operator = jnp.einsum(
        "tfbu,fe->tfbue",
        transverse_response,
        spectral_operator,
    )
    return operator.reshape(
        n_time * n_frequency * n_baseline,
        transverse_operator.shape[1] * spectral_operator.shape[1],
    )


def build_T_fg_core(
    sky_to_visibility: jax.Array,
    spectral_operator: jax.Array,
) -> jax.Array:
    r"""Build :math:`T_{\rm FG}=R Q_z` for one visibility-row block.

    Columns are ordered by spectral component and then image-domain pixel,
    matching ``q = (q_0.T, q_1.T, ...).T`` in the physical model.
    """
    sky_to_visibility = jnp.asarray(sky_to_visibility)
    spectral_operator = jnp.asarray(spectral_operator)

    if sky_to_visibility.ndim != 4:
        raise ValueError(
            "sky_to_visibility must have shape "
            "(n_time, n_frequency, n_baseline, n_pixel)"
        )
    n_time, n_frequency, n_baseline, n_pixel = sky_to_visibility.shape
    if (
        spectral_operator.ndim != 2
        or spectral_operator.shape[0] != n_frequency
    ):
        raise ValueError(
            "spectral_operator must have shape "
            "(n_frequency, n_spectral_component)"
        )

    operator = jnp.einsum(
        "tfbp,fs->tfbsp",
        sky_to_visibility,
        spectral_operator,
    )
    return operator.reshape(
        n_time * n_frequency * n_baseline,
        spectral_operator.shape[1] * n_pixel,
    )


def build_foreground_spectral_operator(
    frequencies_mhz: jax.Array,
    reference_frequency_mhz: float,
    spectral_indices: Sequence[float] = (-2.63, -2.82),
) -> jax.Array:
    """Return the constant and power-law columns defining ``Q_z``."""
    frequencies_mhz = jnp.asarray(frequencies_mhz, dtype=jnp.float64)
    if frequencies_mhz.ndim != 1:
        raise ValueError("frequencies_mhz must be one-dimensional")
    if reference_frequency_mhz <= 0:
        raise ValueError("reference_frequency_mhz must be positive")

    indices = jnp.asarray(tuple(spectral_indices), dtype=jnp.float64)
    power_laws = (
        frequencies_mhz[:, None] / float(reference_frequency_mhz)
    ) ** indices[None, :]
    return jnp.concatenate(
        [
            jnp.ones((frequencies_mhz.size, 1), dtype=jnp.float64),
            power_laws,
        ],
        axis=1,
    ).astype(jnp.complex128)
