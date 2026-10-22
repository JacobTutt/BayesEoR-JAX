"""Define the (u, v, eta) coordinates of cosmological brightness coefficients.

These model coordinates supply both the Fourier-to-image operators and the
conversion to cosmological k bins; they are distinct from measured baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from ..instrument.beam import Beam_Model

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FourierGeometry:
    r"""Retained Fourier coordinates and their integration spacings.

    Parameters
    ----------
    uv : jax.Array
        Retained angular Fourier coordinates, shape (2, n_uv_mode), excluding
        the transverse monopole.
    eta : jax.Array
        Retained line-of-sight Fourier coordinates in inverse Hz, shape
        (n_eta_mode,), excluding eta=0.
    du : float
        Transverse u spacing in inverse radians; must be positive.
    dv : float
        Transverse v spacing in inverse radians; must be positive.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).

    Notes
    -----
    ``uv`` has shape ``(2, n_uv_mode)`` and contains angular spatial
    frequencies; ``eta`` has shape ``(n_eta_mode,)`` in inverse Hz.
    ``du``, ``dv``, and ``d_eta`` supply the corresponding integration weights.
    A coefficient belongs to each (uv, eta) pair, with eta varying fastest
    when coefficients are flattened into a vector.
    """

    uv: jax.Array
    eta: jax.Array
    du: float
    dv: float
    d_eta: float


def _build_uv_modes(
    n_u: int,
    n_v: int,
    du: float,
    dv: float,
) -> jax.Array:
    r"""Place transverse brightness modes on a centred rectangular uv grid.

    Parameters
    ----------
    n_u : int
        Number of transverse u modes before removing the angular monopole.
    n_v : int
        Number of transverse v modes before removing the angular monopole.
    du : float
        Transverse u spacing in inverse radians; must be positive.
    dv : float
        Transverse v spacing in inverse radians; must be positive.

    Returns
    -------
    uv : jax.Array
        Centred coordinates, shape (2, n_u*n_v - 1), ordered with u varying
        fastest.

    Notes
    -----
    Axis coordinates are ``(arange(n) - n//2) * spacing``. Return shape
    ``(2, n_u*n_v - 1)`` with u varying fastest within each v row and remove
    the angular monopole ``(u,v)=(0,0)``.
    """
    # Centre both angular Fourier axes on the zero mode.
    u = (jnp.arange(n_u, dtype=jnp.float64) - n_u // 2) * du
    v = (jnp.arange(n_v, dtype=jnp.float64) - n_v // 2) * dv
    # Row-major flattening makes u the fast axis within each v row.
    u_grid, v_grid = jnp.meshgrid(u, v, indexing="xy")
    uv = jnp.stack([u_grid.reshape(-1), v_grid.reshape(-1)])
    # Remove only the transverse monopole, retaining the other uv pairs.
    monopole = (n_v // 2) * n_u + (n_u // 2)
    return jnp.concatenate(
        [uv[:, :monopole], uv[:, monopole + 1 :]],
        axis=1,
    )


def _build_eta_modes(n_eta: int, d_eta: float) -> jax.Array:
    r"""Place line-of-sight brightness modes on a centred eta axis.

    Parameters
    ----------
    n_eta : int
        Number of line-of-sight Fourier modes before excluding eta=0.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).

    Returns
    -------
    eta : jax.Array
        Centred nonzero eta modes in inverse Hz, shape (n_eta - 1,).

    Notes
    -----
    Return ``(arange(n_eta) - n_eta//2) * d_eta`` in inverse Hz with the
    frequency-independent ``eta=0`` mode removed, giving ``n_eta - 1`` modes.
    """
    # Use the same centred indexing as the transverse grid.
    eta = (jnp.arange(n_eta, dtype=jnp.float64) - n_eta // 2) * d_eta
    # Exclude the frequency-independent mode from the cosmological component.
    eta0 = n_eta // 2
    return jnp.concatenate([eta[:eta0], eta[eta0 + 1 :]])


def build_fourier_geometry(
    *,
    n_u: int,
    n_v: int,
    n_eta: int,
    du: float,
    dv: float,
    d_eta: float,
) -> FourierGeometry:
    r"""Define the rectilinear Fourier grid of the cosmological 21-cm brightness field.

    Parameters
    ----------
    n_u : int
        Number of transverse u modes before removing the angular monopole.
    n_v : int
        Number of transverse v modes before removing the angular monopole.
    n_eta : int
        Number of line-of-sight Fourier modes before excluding eta=0.
    du : float
        Transverse u spacing in inverse radians; must be positive.
    dv : float
        Transverse v spacing in inverse radians; must be positive.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).

    Returns
    -------
    geometry : FourierGeometry
        Retained coordinates and spacings for (n_u*n_v - 1)*(n_eta - 1)
        coefficients.

    Notes
    -----
    The intrinsic cosmological brightness field is parameterised by complex
    amplitudes on a three-dimensional rectilinear ``(u, v, eta)`` grid, rather
    than by independent image-pixel amplitudes. The transverse coordinates
    ``u`` and ``v`` are Fourier duals of the reference-time image direction
    cosines ``(l, m)``, while ``eta`` is the Fourier dual of observing
    frequency. These are model coordinates, not the baselines sampled by the
    instrument. The transverse representation uses a local tangent-plane
    description of the cosmological field.

    This basis provides a natural parameterisation for power-spectrum
    inference. At a reference redshift, the modes map to comoving wavenumbers as

    .. math::

        k_x = 2\pi u / D_M, \qquad
        k_y = 2\pi v / D_M, \qquad
        k_z = 2\pi\eta / |d\chi/d\nu|.

    Here ``D_M`` is the transverse comoving distance and ``|dchi/dnu|`` is
    the line-of-sight distance per unit frequency. Under the statistically
    homogeneous and isotropic Gaussian signal model, distinct Fourier modes
    are uncorrelated and their prior variances are determined by the power
    spectrum at ``|k|``. Grouping modes into spherical k bins therefore links
    the coefficient prior directly to the power-spectrum amplitudes inferred
    downstream, while the individual amplitudes can be analytically marginalised.
    Coordinate conversion, binning, and prior construction are separate from
    this geometry builder.

    Every retained transverse pair is combined with every retained eta mode.
    The ``(u, v)=(0, 0)`` pair and ``eta=0`` plane are excluded, leaving
    ``(n_u*n_v - 1) * (n_eta - 1)`` coefficients. Flattening orders v first,
    then u, with eta varying fastest. The spacings ``du``, ``dv``, and
    ``d_eta`` also supply the Fourier integration weights; they are normally
    chosen from the reciprocal transverse model extents and observing bandwidth.

    Downstream, ``F_z`` maps eta modes to frequency and ``F'`` maps transverse
    modes to brightness at the selected HEALPix pixel centres. The primary
    beam and full three-dimensional fringe response then propagate this field
    into the measured visibilities, where instrumental coupling mixes the
    mode signatures. This function defines only coordinates and spacings; it
    does not construct a brightness realisation, prior covariance, or any of
    those operators.
    """
    # Reject invalid dimensions and integration weights before building axes.
    if min(n_u, n_v, n_eta) < 1:
        raise ValueError("Fourier-grid dimensions must all be positive")
    if min(du, dv, d_eta) <= 0:
        raise ValueError("Fourier-grid spacings must all be positive")

    # Share these retained coordinates between operator construction and k binning.
    return FourierGeometry(
        uv=_build_uv_modes(n_u, n_v, du, dv),
        eta=_build_eta_modes(n_eta, d_eta),
        du=float(du),
        dv=float(dv),
        d_eta=float(d_eta),
    )


def _resolve_uv_spacing_from_image_extents(
    *,
    extent_u: float,
    extent_v: float,
    du: float | None,
    dv: float | None,
    label: str,
    rtol: float = 5e-2,
) -> tuple[float, float]:
    """
    Resolve anisotropic uv-grid spacings from local angular chart widths.

    The input extents are expected to be angular widths in radians along the
    two transverse axes of the fixed image-domain chart. In the current
    wide-field setup these are built from the full ``(l, m, n)`` vectors using
    ``atan2(l, n)`` and ``atan2(m, n)`` so that the widths remain true angular
    offsets even when the raw direction cosines become strongly non-linear.
    """
    extent_u = float(extent_u)
    extent_v = float(extent_v)
    if extent_u <= 0 or extent_v <= 0:
        raise ValueError(
            f"{label}: extent_u and extent_v must both be > 0, got "
            f"{extent_u} and {extent_v}"
        )

    derived_du = float(1.0 / extent_u)
    derived_dv = float(1.0 / extent_v)
    du_val = derived_du if du is None else float(du)
    dv_val = derived_dv if dv is None else float(dv)

    if not np.isclose(du_val, derived_du, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: du={du_val:.6g} is inconsistent with the local-image "
            f"extent-derived value {derived_du:.6g} (=1/{extent_u:.6g}).",
            RuntimeWarning,
            stacklevel=2,
        )
    if not np.isclose(dv_val, derived_dv, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: dv={dv_val:.6g} is inconsistent with the local-image "
            f"extent-derived value {derived_dv:.6g} (=1/{extent_v:.6g}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return du_val, dv_val


def _resolve_model_grid_size(
    *,
    value: int | None,
    required: int,
    label: str,
) -> int:
    """Infer one uv-axis size or retain an override, warning if undersized."""
    if required < 1:
        required = 1

    if value is None:
        return int(required)

    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{label} must be >= 1, got {resolved}")
    if resolved < required:
        warnings.warn(
            f"{label}={resolved} is smaller than the auto-resolved requirement "
            f"{required} for the selected data, FoV, and beam model.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved


def _resolve_uv_grid_from_extents(
    *,
    extent_u_eor: float,
    extent_v_eor: float,
    freqs_hz: np.ndarray,
    uvws_m: np.ndarray,
    beam_type: str | None,
    fwhm_deg: float | None,
    diam: float | None,
    n_u_eor: int | None,
    n_v_eor: int | None,
    du_eor: float | None,
    dv_eor: float | None,
) -> dict[str, float | int]:
    """
    Resolve anisotropic uv-grid spacings and required axis sizes.

    The fixed central-time angular chart widths set the uv sampling through
    ``du = 1 / extent_u`` and ``dv = 1 / extent_v``. The grid extents are then
    chosen to cover the instrument support plus half the beam-aperture width on
    each side. Omitted axis sizes use these estimates independently; smaller
    explicit sizes are retained with a warning.
    """
    du_eor_val, dv_eor_val = _resolve_uv_spacing_from_image_extents(
        extent_u=extent_u_eor,
        extent_v=extent_v_eor,
        du=du_eor,
        dv=dv_eor,
        label="EoR uv grid spacing",
    )
    uvws_m = np.asarray(uvws_m, dtype=float).reshape(-1, 3)
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if uvws_m.size == 0:
        raise ValueError("uvws_m must contain at least one baseline sample")

    wavelength_min = float(299792458.0 / freqs_hz.max())
    uvs = uvws_m[:, :2] / wavelength_min
    u_max_inst_model = float(np.abs(uvs[:, 0]).max())
    v_max_inst_model = float(np.abs(uvs[:, 1]).max())

    beam = Beam_Model(
        beam_type=beam_type,
        fwhm_deg=fwhm_deg,
        diam=diam,
    )
    aperture_width = beam.aperture_width_uv(freqs_hz=freqs_hz)
    u_extent = u_max_inst_model + 0.5 * aperture_width
    v_extent = v_max_inst_model + 0.5 * aperture_width

    n_u_eor_required = int(np.ceil(u_extent / du_eor_val)) * 2 + 1
    n_v_eor_required = int(np.ceil(v_extent / dv_eor_val)) * 2 + 1
    return {
        "du_eor": float(du_eor_val),
        "dv_eor": float(dv_eor_val),
        "n_u_eor_required": int(n_u_eor_required),
        "n_v_eor_required": int(n_v_eor_required),
        "n_u_eor": _resolve_model_grid_size(
            value=n_u_eor,
            required=n_u_eor_required,
            label="n_u_eor",
        ),
        "n_v_eor": _resolve_model_grid_size(
            value=n_v_eor,
            required=n_v_eor_required,
            label="n_v_eor",
        ),
        "u_max_inst_model": float(u_max_inst_model),
        "v_max_inst_model": float(v_max_inst_model),
        "beam_aperture_width_uv": float(aperture_width),
        "u_extent_with_buffer": float(u_extent),
        "v_extent_with_buffer": float(v_extent),
    }
