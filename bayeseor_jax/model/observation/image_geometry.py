"""Select celestial HEALPix pixels and follow their directions during a drift scan.

Pixel identities remain fixed on the celestial sky. Their topocentric
direction cosines drive the three-dimensional fringe response, while their
azimuth and zenith angle determine the primary-beam weighting at each time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import astropy.units as u
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

jax.config.update("jax_enable_x64", True)


HERA_LOCATION = (
    -30.72152777777791,
    21.428305555555557,
    1073.0000000093132,
)


def required_healpix_nside(
    longest_baseline_m: float,
    highest_frequency_hz: float,
    *,
    samples_per_fringe: float = 2.0,
) -> int:
    r"""Estimate the HEALPix resolution needed to sample the shortest fringe.

    Parameters
    ----------
    longest_baseline_m : float
        Largest physical baseline length in metres, finite and non-negative.
        Use the full three-dimensional length, not just its uv projection.
    highest_frequency_hz : float
        Highest observing frequency in Hz, finite and positive. This sets the
        shortest wavelength and hence the finest instrumental fringe scale.
    samples_per_fringe : float, optional
        Number of nominal pixels per fringe, at least two. Defaults to the
        two-sample Nyquist estimate; larger values provide finer sampling.

    Returns
    -------
    int
        Smallest power-of-two ``nside`` satisfying the nominal pixel-size
        criterion, with a minimum of one. Values outside the supported
        HEALPix range (``nside < 2**30``) raise ValueError.

    Notes
    -----
    The largest angular phase gradient is :math:`2\pi b_{\max}/\lambda`.
    Using the nominal HEALPix pixel width :math:`\sqrt{\Omega_{\rm pix}}`
    therefore gives

    .. math::

        \frac{\sqrt{\pi/3}}{N_{\rm side}}
        \leq \frac{c}{s\,b_{\max}\nu_{\max}},

    where :math:`s` is ``samples_per_fringe``. For a zero baseline there is
    no fringe-resolution requirement and the function returns one.

    This is a baseline-based sampling estimate, not a strict spherical
    sampling theorem: HEALPix pixels have different shapes, and beam, sky
    structure and sharp horizon masks can require finer quadrature. Check
    visibility convergence at higher resolution when accuracy matters. The
    helper does not change a run's configured resolution or foreground prior.
    See https://healpy.readthedocs.io/en/latest/generated/healpy.pixelfunc.nside2resol.html
    for the nominal pixel-width convention.
    """
    baseline = float(longest_baseline_m)
    frequency = float(highest_frequency_hz)
    samples = float(samples_per_fringe)
    if not np.isfinite(baseline) or baseline < 0:
        raise ValueError("longest_baseline_m must be finite and non-negative")
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("highest_frequency_hz must be finite and positive")
    if not np.isfinite(samples) or samples < 2:
        raise ValueError("samples_per_fringe must be finite and at least two")

    # Round upward, never to the nearest resolution, to preserve the criterion.
    minimum = np.sqrt(np.pi / 3.0) * samples * baseline * (frequency / 299792458.0)
    if not np.isfinite(minimum) or minimum > 2**29:
        raise ValueError("Required resolution exceeds the supported HEALPix nside range")
    return 1 << max(0, (max(1, int(np.ceil(minimum))) - 1).bit_length())


@dataclass(frozen=True)
class ImageGeometry:
    r"""Coordinates of the HEALPix pixels used to represent sky brightness.

    Parameters
    ----------
    time_ind_center : int
        Index of the reference integration within the observing-time axis.
    eor_fg_shared_axis : jax.Array
        Boolean mask of shape (n_pix_fg,) selecting cosmological pixels from the
        foreground pixel axis.
    n_pix_fg : int
        Number of selected foreground HEALPix pixels.
    pixel_area_sr : float
        Solid angle of one HEALPix pixel in steradians.
    lmn_fg : jax.Array
        East, north, and zenith direction cosines, shape (n_time, 3, n_pix_fg).
    az_za_fg : jax.Array
        Azimuth and zenith angle in radians, shape (n_time, 2, n_pix_fg).
    fov_l_angle_eor : float
        Pixel-padded cosmological angular extent along the local l chart in
        radians.
    fov_m_angle_eor : float
        Pixel-padded cosmological angular extent along the local m chart in
        radians.
    fov_l_angle_fg : float
        Pixel-padded foreground angular extent along the local l chart in
        radians.
    fov_m_angle_fg : float
        Pixel-padded foreground angular extent along the local m chart in
        radians.

    Notes
    -----
    ``lmn_fg`` has shape ``(n_time, 3, n_pix_fg)`` with east, north, and
    zenith direction cosines. ``az_za_fg`` has shape
    ``(n_time, 2, n_pix_fg)`` and contains azimuth and zenith angle in radians.
    Both arrays retain the same pixel ordering throughout the observation.

    ``eor_fg_shared_axis`` selects cosmological pixels from the foreground
    pixel axis. ``time_ind_center`` selects the reference-time chart used by
    the transverse Fourier operator. The four ``fov_*_angle_*`` values are
    pixel-padded angular widths in radians in that chart; ``pixel_area_sr``
    is the equal solid angle used when integrating brightness over the sky.
    """

    time_ind_center: int
    eor_fg_shared_axis: jax.Array
    n_pix_fg: int
    pixel_area_sr: float
    lmn_fg: jax.Array
    az_za_fg: jax.Array
    fov_l_angle_eor: float
    fov_m_angle_eor: float
    fov_l_angle_fg: float
    fov_m_angle_fg: float


def _construct_time_array(
    jd_center: float,
    n_t: int,
    d_t: float,
) -> tuple[np.ndarray, int]:
    r"""Place integration centres around a reference Julian date in UTC.

    Parameters
    ----------
    jd_center : float
        Reference integration time as a UTC Julian date.
    n_t : int
        Number of observing-time samples.
    d_t : float
        Separation of integration centres in seconds.

    Returns
    -------
    times_jd : numpy.ndarray
        UTC Julian dates of the integration centres, shape (n_t,).
    reference_index : int
        Index n_t // 2 corresponding to jd_center.

    Notes
    -----
    Return ``(times_jd, reference_index)`` for ``n_t`` integrations spaced
    by ``d_t`` seconds. The reference date is at index ``n_t // 2``; for an
    even count there is one more integration before it than after it.
    """
    # Place the reference integration at index n_t//2 for odd and even counts.
    if n_t % 2:
        time_inds = np.arange(-(n_t // 2), n_t // 2 + 1)
    else:
        time_inds = np.arange(-(n_t // 2), n_t // 2)
    # Julian dates are measured in days, while integration spacing is in seconds.
    times_jd = jd_center + time_inds * d_t / (24 * 60 * 60)
    return times_jd, n_t // 2


def _observation_field_center(
    jd: float,
    telescope_location: EarthLocation,
) -> tuple[float, float]:
    r"""Locate the zenith-pointing field on the celestial sky at one time.

    Parameters
    ----------
    jd : float
        UTC Julian date at which to locate the zenith field.
    telescope_location : astropy.coordinates.EarthLocation
        Geodetic telescope location used for the ICRS-to-topocentric
        transformation.

    Returns
    -------
    ra_deg : float
        ICRS right ascension of zenith in degrees.
    dec_deg : float
        ICRS declination of zenith in degrees.

    Notes
    -----
    Transform zenith at the supplied EarthLocation and UTC Julian date to
    ICRS, returning ``(right_ascension, declination)`` in degrees. Repeating
    this over the observation defines the path of the drift-scan field.
    """
    # Locate the instrument zenith on the celestial sphere at this integration.
    altaz = AltAz(
        obstime=Time(jd, scale="utc", format="jd"),
        location=telescope_location,
    )
    zenith = SkyCoord(az=0 * u.deg, alt=90 * u.deg, frame=altaz)
    zenith_icrs = zenith.transform_to("icrs")
    return float(zenith_icrs.ra.deg), float(zenith_icrs.dec.deg)


def _query_disc_union(
    field_centers: Sequence[tuple[float, float]],
    radius_rad: float,
    *,
    nside: int,
) -> np.ndarray:
    r"""Find HEALPix pixel centres inside the observed spherical field.

    Parameters
    ----------
    field_centers : sequence of tuple of float
        ICRS (right ascension, declination) pairs in degrees, one per
        integration.
    radius_rad : float
        Angular radius of the spherical selection in radians.
    nside : int
        HEALPix resolution of the equal-area celestial pixel grid.

    Returns
    -------
    pixel_indices : numpy.ndarray
        Sorted unique HEALPix RING indices, shape (n_selected_pixel,).

    Notes
    -----
    ``field_centers`` contains ICRS (RA, Dec) pairs in degrees and
    ``radius_rad`` is the angular field radius. Return sorted, unique RING
    pixel indices at resolution ``nside``, including pixels inside the field
    at any integration. Pixel overlap with the boundary is not included.
    """
    npix = 12 * nside**2
    indices = []
    # Query a spherical disc around each pointing using a Cartesian unit vector.
    for ra_deg, dec_deg in field_centers:
        ra_rad = jnp.radians(ra_deg)
        dec_rad = jnp.radians(dec_deg)
        centre = jnp.asarray(
            [
                jnp.cos(dec_rad) * jnp.cos(ra_rad),
                jnp.cos(dec_rad) * jnp.sin(ra_rad),
                jnp.sin(dec_rad),
            ]
        )
        pixels = jhp.query_disc(
            nside=nside,
            vec=centre,
            radius=radius_rad,
            inclusive=False,
        )
        # Discard padded query entries that lie outside the HEALPix index range.
        indices.append(np.asarray(pixels[pixels < npix], dtype=np.int64))

    if not indices:
        return np.asarray([], dtype=np.int64)
    # One celestial pixel can enter several snapshots but appears only once.
    return np.unique(np.concatenate(indices))


def _select_image_pixels(
    field_centers: Sequence[tuple[float, float]],
    *,
    fov_eor: float,
    fov_fg: float,
    nside: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Select sky pixels entering either model field during the observation.

    Parameters
    ----------
    field_centers : sequence of tuple of float
        ICRS (right ascension, declination) pairs in degrees, one per
        integration.
    fov_eor : float
        Full angular diameter of the cosmological model field in degrees.
    fov_fg : float or None
        Full angular diameter of the foreground model field in degrees; None
        uses fov_eor.
    nside : int
        HEALPix resolution of the equal-area celestial pixel grid.

    Returns
    -------
    pixel_indices_21 : jax.Array
        Cosmological HEALPix RING indices, shape (n_pixel_21,).
    pixel_indices_fg : jax.Array
        Foreground HEALPix RING indices, shape (n_pixel_fg,).
    shared_mask : jax.Array
        Boolean selection of cosmological pixels within the foreground axis,
        shape (n_pixel_fg,).

    Notes
    -----
    Interpret ``fov_eor`` and ``fov_fg`` as full angular diameters in degrees.
    Return the cosmological indices, foreground indices, and a Boolean mask
    locating cosmological pixels within the foreground array. The shared
    response construction assumes the foreground selection contains the
    cosmological selection.
    """
    # Convert full field diameters into selection radii for each component.
    pix_indices_eor_np = _query_disc_union(
        field_centers,
        float(np.deg2rad(fov_eor / 2)),
        nside=nside,
    )
    # Reuse an identical selection instead of querying the same field twice.
    if fov_fg == fov_eor:
        pix_indices_fg_np = pix_indices_eor_np
    else:
        pix_indices_fg_np = _query_disc_union(
            field_centers,
            float(np.deg2rad(fov_fg / 2)),
            nside=nside,
        )
    # The membership mask selects cosmological columns from the wider sky response.
    pix_indices_eor = jnp.asarray(pix_indices_eor_np, dtype=jnp.int64)
    pix_indices_fg = jnp.asarray(pix_indices_fg_np, dtype=jnp.int64)
    return (
        pix_indices_eor,
        pix_indices_fg,
        jnp.isin(pix_indices_fg, pix_indices_eor),
    )


def _topocentric_coordinates(
    times_jd: np.ndarray,
    pix_indices_fg: jax.Array,
    *,
    nside: int,
    telescope_location: EarthLocation,
) -> tuple[jax.Array, jax.Array]:
    r"""Rotate fixed celestial HEALPix directions into the instrument frame.

    Parameters
    ----------
    times_jd : numpy.ndarray
        UTC Julian dates, shape (n_time,), at which to evaluate sky directions.
    pix_indices_fg : jax.Array
        Selected foreground HEALPix RING indices, shape (n_pixel,).
    nside : int
        HEALPix resolution of the equal-area celestial pixel grid.
    telescope_location : astropy.coordinates.EarthLocation
        Geodetic telescope location used for the ICRS-to-topocentric
        transformation.

    Returns
    -------
    lmn : jax.Array
        East, north, and zenith direction cosines, shape (n_time, 3, n_pixel).
    az_za : jax.Array
        Azimuth and zenith angle in radians, shape (n_time, 2, n_pixel).

    Notes
    -----
    At each UTC Julian date, transform the selected ICRS pixel centres to
    azimuth and zenith angle at the telescope location, then evaluate

    .. math::

        (l,m,n)=(\sin z\sin A,\sin z\cos A,\cos z).

    Return arrays of shape ``(n_time, 3, n_pixel)`` and
    ``(n_time, 2, n_pixel)`` respectively. Angles are in radians; the
    direction cosines point east, north, and up. Below-horizon directions
    remain in these arrays; the beam model sets their gain to zero.
    """
    # Celestial pixel centres stay fixed while the instrument frame rotates.
    ra_deg, dec_deg = jhp.pix2ang(nside, pix_indices_fg, lonlat=True)
    sky = SkyCoord(
        np.asarray(ra_deg) * u.deg,
        np.asarray(dec_deg) * u.deg,
        frame="icrs",
    )

    lmn_by_time = []
    az_za_by_time = []
    # Transform the same sky pixels at each integration, preserving their ordering.
    for jd in times_jd:
        altaz = sky.transform_to(
            AltAz(
                obstime=Time(jd, scale="utc", format="jd"),
                location=telescope_location,
            )
        )
        az = jnp.asarray(altaz.az.rad)
        za = jnp.pi / 2 - jnp.asarray(altaz.alt.rad)
        # East/north/up direction cosines enter the full three-dimensional fringe.
        lmn_by_time.append(
            jnp.stack(
                [
                    jnp.sin(za) * jnp.sin(az),
                    jnp.sin(za) * jnp.cos(az),
                    jnp.cos(za),
                ]
            )
        )
        # Angular coordinates independently supply the primary-beam evaluation.
        az_za_by_time.append(jnp.stack([az, za]))

    return jnp.stack(lmn_by_time), jnp.stack(az_za_by_time)


def _angular_lm_extent(
    lmn_coords: jax.Array,
    pixel_area_sr: float,
) -> tuple[float, float]:
    r"""Measure the angular extent of a pixel selection in a local sky chart.

    Parameters
    ----------
    lmn_coords : array-like
        Topocentric direction cosines, shape (3, n_pixel) or its transpose.
    pixel_area_sr : float
        Solid angle of one HEALPix pixel in steradians.

    Returns
    -------
    l_extent : float
        Pixel-padded width of atan2(l, n), in radians.
    m_extent : float
        Pixel-padded width of atan2(m, n), in radians.

    Notes
    -----
    Use the ranges of ``atan2(l, n)`` and ``atan2(m, n)``, adding
    ``sqrt(pixel_area_sr)`` to account approximately for pixel width.
    Return two widths in radians. Their reciprocals set the default
    transverse Fourier-grid spacings for the cosmological model.
    """
    lmn = np.asarray(lmn_coords, dtype=float)
    if lmn.ndim != 2:
        raise ValueError(
            f"lmn_coords must be two-dimensional, got {lmn.shape}"
        )
    if lmn.shape[0] != 3 and lmn.shape[1] == 3:
        lmn = lmn.T
    if lmn.shape[0] != 3:
        raise ValueError(
            f"lmn_coords must have shape (3, n_pix), got {lmn.shape}"
        )

    # Project into the reference tangent chart and include one pixel width.
    l_angle = np.arctan2(lmn[0], lmn[2])
    m_angle = np.arctan2(lmn[1], lmn[2])
    pixel_angle = float(np.sqrt(pixel_area_sr))
    return (
        float(np.ptp(l_angle) + pixel_angle),
        float(np.ptp(m_angle) + pixel_angle),
    )


def build_image_geometry(
    *,
    fov_eor: float,
    fov_fg: float | None = None,
    nside: int = 128,
    telescope_latlonalt: tuple[float, float, float] = HERA_LOCATION,
    jd_center: float = 2461040.5,
    n_t: int = 1,
    d_t: float = 60.0,
) -> ImageGeometry:
    r"""Construct the spherical sky coordinates used by both forward models.

    Parameters
    ----------
    fov_eor, fov_fg : float
        Full cosmological and foreground field diameters in degrees.
        ``fov_fg=None`` uses the cosmological field. The cosmological field
        must be contained in the foreground field for the shared response.
    nside : int
        HEALPix resolution; all selected pixels have area ``4*pi/(12*nside**2)``.
    telescope_latlonalt : tuple of float
        Geodetic latitude and longitude in degrees, followed by altitude in metres.
    jd_center : float
        Reference UTC Julian date.
    n_t : int
        Number of integrations.
    d_t : float
        Separation of integration centres in seconds.

    Returns
    -------
    ImageGeometry
        Fixed pixel selections represented by their time-dependent directions,
        shared cosmological-pixel mask, solid angle, and reference-time extents.

    Notes
    -----
    Select the union of HEALPix pixels entering each zenith drift-scan field
    across all integrations, follow their topocentric directions, and measure
    their angular extents at the reference time. Return an ``ImageGeometry`` for beam
    evaluation and the mapping from image brightness to visibilities.
    """
    fov_fg = fov_eor if fov_fg is None else fov_fg

    # Establish the observing times and terrestrial location before selecting sky pixels.
    times_jd, time_ind_center = _construct_time_array(jd_center, n_t, d_t)
    lat, lon, alt = telescope_latlonalt
    telescope_location = EarthLocation.from_geodetic(
        lon=lon * u.deg,
        lat=lat * u.deg,
        height=alt * u.m,
    )
    # Follow zenith through celestial coordinates to define the drift-scan footprint.
    field_centers = [
        _observation_field_center(float(jd), telescope_location)
        for jd in times_jd
    ]
    pix_indices_eor, pix_indices_fg, eor_fg_shared_axis = (
        _select_image_pixels(
            field_centers,
            fov_eor=fov_eor,
            fov_fg=fov_fg,
            nside=nside,
        )
    )
    # Keep one celestial pixel set and evaluate its moving instrumental coordinates.
    lmn_fg, az_za_fg = _topocentric_coordinates(
        times_jd,
        pix_indices_fg,
        nside=nside,
        telescope_location=telescope_location,
    )
    # Equal-area pixel weights enter the fringe operator; angular extents set uv spacing.
    pixel_area_sr = float(jhp.nside2pixarea(nside))
    fov_l_angle_fg, fov_m_angle_fg = _angular_lm_extent(
        lmn_fg[time_ind_center],
        pixel_area_sr,
    )
    fov_l_angle_eor, fov_m_angle_eor = _angular_lm_extent(
        lmn_fg[time_ind_center, :, eor_fg_shared_axis],
        pixel_area_sr,
    )

    return ImageGeometry(
        time_ind_center=time_ind_center,
        eor_fg_shared_axis=eor_fg_shared_axis,
        n_pix_fg=int(pix_indices_fg.size),
        pixel_area_sr=pixel_area_sr,
        lmn_fg=lmn_fg,
        az_za_fg=az_za_fg,
        fov_l_angle_eor=fov_l_angle_eor,
        fov_m_angle_eor=fov_m_angle_eor,
        fov_l_angle_fg=fov_l_angle_fg,
        fov_m_angle_fg=fov_m_angle_fg,
    )
