r"""Linear operators used in BayesEoR's interferometric forward model.

The component models share the beam-weighted measurement operator
:math:`R = F^{-1}P` and differ in how their coefficients define sky brightness:

.. math::

    m_{21} = T_{21}a, \qquad T_{21} = R F' F_z,
    \qquad m_{\rm FG} = T_{\rm FG}q, \qquad T_{\rm FG} = R Q_z.

``build_transverse_fourier_operator`` constructs ``F'`` (uv to image),
``build_spectral_fourier_operator`` constructs ``F_z`` (eta to frequency),
and ``build_foreground_spectral_operator`` supplies the spectral part of
``Q_z``. ``build_sky_to_visibility_operator`` constructs ``R``, including
the pixel solid angle and primary beam. The two ``build_T_*_core`` functions
combine these factors for supplied blocks of visibility rows.

These functions perform numerical construction only. Geometry modules define
the coordinates; ``BuildMatrices`` in ``construction/builder.py`` chooses the blocks,
distributes their construction, and saves the assembled matrices. The compact
factors below implement these products without forming full Kronecker
matrices. No unit conversion, prior weighting, or KL compression is applied.
"""

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
    r"""Map amplitudes between supplied coordinate sets with a Fourier kernel.

    Parameters
    ----------
    coords_in : jnp.ndarray
        Source coordinates. Shape (N_axes, N_in).
        e.g., Sky positions (l, m, n) for a forward transform.
    coords_out : jnp.ndarray
        Target coordinates. Shape (N_axes, N_out).
        e.g., Baselines (u, v, w) for a forward transform.
    sign : int, optional
        Fourier exponent sign: -1 gives ``exp(-2 pi i phase)`` and +1 gives
        its complex conjugate. This is independent of the baseline convention
        documented in ``build_sky_to_visibility_operator``.

    Returns
    -------
    nudft_matrix : jnp.ndarray
        A Complex128 matrix of shape (N_out, N_in).

    Notes
    -----
    For input coordinates ``x_i`` and output coordinates ``y_j``, return

    .. math::

        K_{ji} = \exp\!\left(\mathrm{sign}\,2\pi i\,y_j\cdot x_i\right).

    Multiplication by ``K`` maps input amplitudes to the output samples.
    The coordinate pairs determine the spaces being connected; ``sign``
    selects the Fourier phase convention. Reversing the sign alone does not
    construct a matrix inverse for arbitrary non-uniform sampling. The
    physical operator builders supply their own integration weights.
    """
    # Treat one-dimensional coordinates as a single coordinate axis.
    c_in = jnp.atleast_2d(jnp.array(coords_in, dtype=jnp.float64))
    c_out = jnp.atleast_2d(jnp.array(coords_out, dtype=jnp.float64))

    # Equivalent to the dot product of coordinate vectors for every pair.
    # (N_out, N_axes) @ (N_axes, N_in) -> (N_out, N_in)
    phase = jnp.matmul(c_out.T, c_in)

    # Coordinate dot products are in cycles; 2 pi converts them to radians.
    return jnp.exp(sign * (2.0 * jnp.pi * 1j) * phase)


def build_transverse_fourier_operator(
    uv: jax.Array,
    lm: jax.Array,
    uv_cell_area: float,
) -> jax.Array:
    r"""Map cosmological (u, v) coefficients to brightness at HEALPix pixel centres.

    Parameters
    ----------
    uv : jax.Array
        Retained model coordinates, shape ``(2, n_uv_mode)``. These are the
        cosmological Fourier grid, not the instrument's sampled baselines.
    lm : jax.Array
        Image direction cosines, shape ``(2, n_pixel)``. The caller selects
        the cosmological pixels at the reference time, defining a fixed
        transverse chart for the intrinsic brightness field.
    uv_cell_area : float
        Fourier-cell integration weight ``du * dv``.

    Returns
    -------
    jax.Array
        Complex matrix of shape ``(n_pixel, n_uv_mode)``. Applied alongside
        ``F_z`` in ``build_T_21_core``; drift-scan time dependence enters the
        separate sky-to-visibility operator.

    Notes
    -----
    This transverse Fourier operator is denoted :math:`F'`.

    .. math::

        F'_{pu} = \Delta u\,\Delta v\,
        \exp[-2\pi i(u_u l_p + v_u m_p)].
    """
    if uv_cell_area <= 0:
        raise ValueError("uv_cell_area must be positive")
    return build_nudft_matrix(uv, lm, sign=-1) * uv_cell_area


def build_spectral_fourier_operator(
    eta: jax.Array,
    frequency_offsets_hz: jax.Array,
    d_eta: float,
) -> jax.Array:
    r"""Map cosmological eta coefficients to brightness at observing frequencies.

    Parameters
    ----------
    eta : jax.Array
        Retained line-of-sight Fourier coordinates in inverse Hz (seconds),
        shape ``(n_eta_mode,)``. Geometry construction excludes ``eta=0``.
    frequency_offsets_hz : jax.Array
        Frequencies relative to the chosen reference channel, in Hz, shape
        ``(n_frequency,)``. ``BuildMatrices`` uses channel ``n_freq // 2``.
    d_eta : float
        Fourier integration weight in inverse Hz, normally ``1 / bandwidth``.

    Returns
    -------
    jax.Array
        Complex matrix of shape ``(n_frequency, n_eta_mode)``. The same
        spectral mapping applies to every transverse cosmological mode.

    Notes
    -----
    This line-of-sight Fourier operator is denoted :math:`F_z`.

    .. math::

        (F_z)_{fe} = \Delta\eta\,
        \exp[-2\pi i(\nu_f-\nu_{\rm ref})\eta_e].
    """
    # Validate the line-of-sight integration weight before constructing F_z.
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
    # Fourier phases map eta coefficients to frequency, weighted by the eta spacing.
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
    r"""Map individual HEALPix pixel brightnesses to measured visibilities.

    Parameters
    ----------
    lmn : jax.Array
        Topocentric sky coordinates with shape ``(n_time, 3, n_pixel)``.
        Pixel identities remain fixed on the celestial sky while their
        topocentric direction cosines change with observing time.
    uvw_lambda : jax.Array
        Baselines in observing wavelengths with shape
        ``(n_frequency, 3, n_baseline)``. BayesEoR follows the PyUVData
        ``ant2 - ant1`` baseline convention, hence the positive phase used
        below is equivalent to a negative phase with the opposite
        ``ant1 - ant2`` baseline convention.
        Baselines are fixed in the instrument frame; sky rotation is already
        represented by ``lmn``.
    beam : jax.Array
        Horizon-limited primary-beam power response with shape
        ``(n_time, n_frequency, n_pixel)``. Pixels at or below the horizon
        must already have zero gain.
    pixel_area_sr : float
        Solid angle of one equal-area sky pixel in steradians.

    Returns
    -------
    jax.Array
        Beam-weighted sky-to-visibility response with shape
        ``(n_time, n_frequency, n_baseline, n_pixel)``.

    Notes
    -----
    At each observing time, propagate the pixels' topocentric (l, m, n)
    directions through the full three-dimensional discrete Fourier response,
    including primary-beam weighting and pixel solid angle. The combined
    operator is :math:`R=F^{-1}P`.

    .. math::

        R_{tfbp} = \Delta\Omega\,P_{tfp}\,
        \exp[+2\pi i\,\boldsymbol u_b(\nu_f)\cdot\hat{\boldsymbol s}_p(t)].

    Summing ``R[t, f, b, p] * brightness[f, p]`` over pixels gives a model
    visibility. Both sky components use this response, restricted to their
    respective pixel selections. ``F^{-1}`` denotes the sky-to-visibility
    Fourier operator; it does not imply a numerical matrix inversion here.
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

    # Contract all three direction components, retaining the full-sky fringe.
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
    r"""Map cosmological (u, v, eta) amplitudes to a block of visibilities.

    Parameters
    ----------
    sky_to_visibility : jax.Array
        ``R`` restricted to cosmological image pixels, with shape
        ``(n_time, n_frequency, n_baseline, n_pixel)``.
    transverse_operator : jax.Array
        ``F'`` with shape ``(n_pixel, n_uv_mode)``. Its pixel order must match
        the last axis of ``sky_to_visibility``.
    spectral_operator : jax.Array
        ``F_z`` with shape ``(n_frequency, n_eta_mode)``, selected for the
        same frequency rows as ``sky_to_visibility``.

    Returns
    -------
    jax.Array
        Matrix of shape ``(n_time * n_frequency * n_baseline,
        n_uv_mode * n_eta_mode)``. Rows are ordered by time, frequency, then
        baseline; columns by transverse mode, then eta (fastest varying).
        Multiplication by the correspondingly ordered ``a`` gives ``m_21``.

    Notes
    -----
    Reconstruct their brightness on the cosmological HEALPix selection and
    propagate it through the beam-weighted three-dimensional fringe response,
    implementing :math:`T_{21}=R F' F_z`.

    .. math::

        (T_{21})_{(tfb),(ue)} =
        \left[\sum_p R_{tfbp} F'_{pu}\right](F_z)_{fe}.

    ``core`` denotes the equation-level calculation for supplied arrays.
    ``BuildMatrices.build_T_21`` handles distributed construction and saving.
    All integration weights are already included in the input operators.
    """
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

    # Sum over sky pixels first, avoiding a full image-by-(uv, eta) matrix.
    transverse_response = jnp.einsum(
        "tfbp,pu->tfbu",
        sky_to_visibility,
        transverse_operator,
    )
    # At each frequency, attach the eta response to every transverse mode.
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
    r"""Map spectral foreground amplitudes at each HEALPix pixel to visibilities.

    Parameters
    ----------
    sky_to_visibility : jax.Array
        ``R`` for the foreground pixel selection, with shape
        ``(n_time, n_frequency, n_baseline, n_pixel)``.
    spectral_operator : jax.Array
        Spectral factors ``Q`` from ``build_foreground_spectral_operator``,
        shape ``(n_frequency, n_spectral_component)``. Frequency rows must
        match those in ``sky_to_visibility``.

    Returns
    -------
    jax.Array
        Matrix of shape ``(n_time * n_frequency * n_baseline,
        n_spectral_component * n_pixel)``. Rows are ordered by time,
        frequency, then baseline. Columns are ordered by spectral component,
        then pixel, matching ``q = (q_0.T, q_1.T, ...).T``.

    Notes
    -----
    Evaluate each pixel's brightness at the observing frequencies and apply
    the beam-weighted three-dimensional fringe response, implementing
    :math:`T_{\rm FG}=R Q_z`.

    .. math::

        (T_{\rm FG})_{(tfb),(sp)} = R_{tfbp} Q_{fs}.

    Each foreground coefficient belongs to one image pixel and one spectral
    component. ``Q`` supplies the spectral dependence at every pixel;
    ``R`` propagates the resulting brightness through the instrument.

    ``BuildMatrices.build_T_fg`` distributes and saves blocks from this core.
    The pixel identity in ``Q_z`` is applied implicitly, so no dense
    Kronecker product or additional pixel-area factor is needed.
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

    # Each spectral component weights every pixel without mixing spatial indices.
    operator = jnp.einsum(
        "tfbp,fs->tfbsp",
        sky_to_visibility,
        spectral_operator,
    )
    # Flatten rows as time/frequency/baseline and columns as component/pixel.
    return operator.reshape(
        n_time * n_frequency * n_baseline,
        spectral_operator.shape[1] * n_pixel,
    )


def build_foreground_spectral_operator(
    frequencies_mhz: jax.Array,
    reference_frequency_mhz: float,
    spectral_indices: Sequence[float] = (-2.63, -2.82),
) -> jax.Array:
    r"""Map a pixel's foreground amplitudes to its brightness spectrum.

    Parameters
    ----------
    frequencies_mhz : jax.Array
        Observing frequencies in MHz, shape ``(n_frequency,)``.
    reference_frequency_mhz : float
        Positive reference frequency ``nu_0`` in MHz. Power-law amplitudes
        are defined at this frequency.
    spectral_indices : sequence of float
        Signed exponents for the power-law components, following the constant
        component. For example, ``-2.63`` gives a spectrum decreasing with
        frequency. ``BuildMatrices`` converts its positive ``beta_fg``
        configuration values to this signed convention.

    Returns
    -------
    jax.Array
        Dimensionless complex matrix of shape
        ``(n_frequency, 1 + len(spectral_indices))``. It contains only the
        spectral factors; ``build_T_fg_core`` applies them independently to
        each image pixel to implement the full ``Q_z`` mapping.

    Notes
    -----
    These constant and power-law factors define the spectral part of
    :math:`Q_z`, shared by every foreground HEALPix pixel.

    .. math::

        Q_{f0}=1,\qquad Q_{fs}=(\nu_f/\nu_0)^{\beta_s}\ (s>0),
        \qquad s_p(\nu_f)=\sum_s Q_{fs}q_{s,p}.
    """
    frequencies_mhz = jnp.asarray(frequencies_mhz, dtype=jnp.float64)
    if frequencies_mhz.ndim != 1:
        raise ValueError("frequencies_mhz must be one-dimensional")
    if reference_frequency_mhz <= 0:
        raise ValueError("reference_frequency_mhz must be positive")

    # Power laws are normalised to unity at the foreground reference frequency.
    indices = jnp.asarray(tuple(spectral_indices), dtype=jnp.float64)
    power_laws = (
        frequencies_mhz[:, None] / float(reference_frequency_mhz)
    ) ** indices[None, :]
    # Prepend the constant spectrum, matching the component-major coefficient order.
    return jnp.concatenate(
        [
            jnp.ones((frequencies_mhz.size, 1), dtype=jnp.float64),
            power_laws,
        ],
        axis=1,
    ).astype(jnp.complex128)
