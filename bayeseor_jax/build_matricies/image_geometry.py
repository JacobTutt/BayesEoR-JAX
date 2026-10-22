"""Spherical image geometry used by the interferometric forward model."""

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


@dataclass(frozen=True)
class ImageGeometry:
    """Fixed sky pixels and their time-dependent topocentric coordinates."""

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
    """Return integration times centred on ``jd_center`` and its index."""
    if n_t % 2:
        time_inds = np.arange(-(n_t // 2), n_t // 2 + 1)
    else:
        time_inds = np.arange(-(n_t // 2), n_t // 2)
    times_jd = jd_center + time_inds * d_t / (24 * 60 * 60)
    return times_jd, n_t // 2


def _observation_field_center(
    jd: float,
    telescope_location: EarthLocation,
) -> tuple[float, float]:
    """Return the ICRS right ascension and declination of zenith."""
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
    sky_mask_time_mode: str,
    time_ind_center: int,
) -> np.ndarray:
    """Return pixels entering a spherical field at the selected times."""
    if sky_mask_time_mode == "center":
        field_centers = [field_centers[time_ind_center]]

    npix = 12 * nside**2
    indices = []
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
        indices.append(np.asarray(pixels[pixels < npix], dtype=np.int64))

    if not indices:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.concatenate(indices))


def _select_image_pixels(
    field_centers: Sequence[tuple[float, float]],
    *,
    fov_eor: float,
    fov_fg: float,
    nside: int,
    sky_mask_time_mode: str,
    time_ind_center: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select the fixed 21-cm and foreground HEALPix pixel sets."""
    pix_indices_eor_np = _query_disc_union(
        field_centers,
        float(np.deg2rad(fov_eor / 2)),
        nside=nside,
        sky_mask_time_mode=sky_mask_time_mode,
        time_ind_center=time_ind_center,
    )
    if fov_fg == fov_eor:
        pix_indices_fg_np = pix_indices_eor_np
    else:
        pix_indices_fg_np = _query_disc_union(
            field_centers,
            float(np.deg2rad(fov_fg / 2)),
            nside=nside,
            sky_mask_time_mode=sky_mask_time_mode,
            time_ind_center=time_ind_center,
        )
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
    """Evaluate foreground-pixel ``(l,m,n)`` and ``(az,za)`` over time."""
    ra_deg, dec_deg = jhp.pix2ang(nside, pix_indices_fg, lonlat=True)
    sky = SkyCoord(
        np.asarray(ra_deg) * u.deg,
        np.asarray(dec_deg) * u.deg,
        frame="icrs",
    )

    lmn_by_time = []
    az_za_by_time = []
    for jd in times_jd:
        altaz = sky.transform_to(
            AltAz(
                obstime=Time(jd, scale="utc", format="jd"),
                location=telescope_location,
            )
        )
        az = jnp.asarray(altaz.az.rad)
        za = jnp.pi / 2 - jnp.asarray(altaz.alt.rad)
        lmn_by_time.append(
            jnp.stack(
                [
                    jnp.sin(za) * jnp.sin(az),
                    jnp.sin(za) * jnp.cos(az),
                    jnp.cos(za),
                ]
            )
        )
        az_za_by_time.append(jnp.stack([az, za]))

    return jnp.stack(lmn_by_time), jnp.stack(az_za_by_time)


def _angular_lm_extent(
    lmn_coords: jax.Array,
    pixel_area_sr: float,
) -> tuple[float, float]:
    """Return pixel-padded angular widths along the local l and m axes."""
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
    sky_mask_time_mode: str = "union",
    telescope_latlonalt: tuple[float, float, float] = HERA_LOCATION,
    jd_center: float = 2461040.5,
    n_t: int = 1,
    d_t: float = 60.0,
) -> ImageGeometry:
    """Construct the image geometry shared by the 21-cm and foreground models."""
    fov_fg = fov_eor if fov_fg is None else fov_fg
    sky_mask_time_mode = str(sky_mask_time_mode).lower()
    if sky_mask_time_mode not in {"union", "center"}:
        raise ValueError(
            "sky_mask_time_mode must be one of {'union', 'center'}, "
            f"got {sky_mask_time_mode!r}."
        )

    times_jd, time_ind_center = _construct_time_array(jd_center, n_t, d_t)
    lat, lon, alt = telescope_latlonalt
    telescope_location = EarthLocation.from_geodetic(
        lon=lon * u.deg,
        lat=lat * u.deg,
        height=alt * u.m,
    )
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
            sky_mask_time_mode=sky_mask_time_mode,
            time_ind_center=time_ind_center,
        )
    )
    lmn_fg, az_za_fg = _topocentric_coordinates(
        times_jd,
        pix_indices_fg,
        nside=nside,
        telescope_location=telescope_location,
    )
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
