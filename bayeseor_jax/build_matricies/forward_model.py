"""Equation-level kernels for the interferometric forward model."""

from __future__ import annotations

from collections.abc import Sequence

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def build_sky_to_visibility_core(
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
        Primary-beam power response with shape
        ``(n_time, n_frequency, n_pixel)``.
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
    visible_beam = jnp.where(lmn[:, None, 2, :] > 0.0, beam, 0.0)
    return (
        jnp.asarray(pixel_area_sr, dtype=jnp.float64)
        * fringe
        * visible_beam[:, :, None, :]
    )


def build_T_eor_core(
    sky_to_visibility: jax.Array,
    transverse_basis: jax.Array,
    spectral_basis: jax.Array,
) -> jax.Array:
    r"""Build :math:`T_{\rm EoR}=R F' F_z` for one visibility-row block."""
    sky_to_visibility = jnp.asarray(sky_to_visibility)
    transverse_basis = jnp.asarray(transverse_basis)
    spectral_basis = jnp.asarray(spectral_basis)

    if sky_to_visibility.ndim != 4:
        raise ValueError(
            "sky_to_visibility must have shape "
            "(n_time, n_frequency, n_baseline, n_pixel)"
        )
    n_time, n_frequency, n_baseline, n_pixel = sky_to_visibility.shape
    if transverse_basis.ndim != 2 or transverse_basis.shape[0] != n_pixel:
        raise ValueError(
            "transverse_basis must have shape (n_pixel, n_uv_mode)"
        )
    if spectral_basis.ndim != 2 or spectral_basis.shape[0] != n_frequency:
        raise ValueError(
            "spectral_basis must have shape (n_frequency, n_eta_mode)"
        )

    transverse_response = jnp.einsum(
        "tfbp,pu->tfbu",
        sky_to_visibility,
        transverse_basis,
    )
    operator = jnp.einsum(
        "tfbu,fe->tfbue",
        transverse_response,
        spectral_basis,
    )
    return operator.reshape(
        n_time * n_frequency * n_baseline,
        transverse_basis.shape[1] * spectral_basis.shape[1],
    )


def build_T_fg_core(
    sky_to_visibility: jax.Array,
    spectral_basis: jax.Array,
) -> jax.Array:
    r"""Build :math:`T_{\rm FG}=R Q_z` for one visibility-row block.

    Columns are ordered by spectral component and then image-domain pixel,
    matching ``q = (q_0.T, q_1.T, ...).T`` in the physical model.
    """
    sky_to_visibility = jnp.asarray(sky_to_visibility)
    spectral_basis = jnp.asarray(spectral_basis)

    if sky_to_visibility.ndim != 4:
        raise ValueError(
            "sky_to_visibility must have shape "
            "(n_time, n_frequency, n_baseline, n_pixel)"
        )
    n_time, n_frequency, n_baseline, n_pixel = sky_to_visibility.shape
    if spectral_basis.ndim != 2 or spectral_basis.shape[0] != n_frequency:
        raise ValueError(
            "spectral_basis must have shape "
            "(n_frequency, n_spectral_component)"
        )

    operator = jnp.einsum(
        "tfbp,fs->tfbsp",
        sky_to_visibility,
        spectral_basis,
    )
    return operator.reshape(
        n_time * n_frequency * n_baseline,
        spectral_basis.shape[1] * n_pixel,
    )


def build_foreground_spectral_basis(
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
