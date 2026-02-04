"""
To model wide-field interferometric data in the image domain, we build model
spaces in (l, m, n) coordinates (not just (l, m)). This module defines a
HEALPix-backed interface for constructing and filtering sky pixels within
field-of-view constraints.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

import astropy_healpix as ahp
from astropy_healpix import healpy as hp
from astropy.coordinates import EarthLocation, AltAz, ICRS, Angle, SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.constants import c
from pyuvdata import UVBeam
from pyuvdata import utils as uvutils


SECS_PER_HOUR = 60 * 60
SECS_PER_DAY = SECS_PER_HOUR * 24
DAYS_PER_SEC = 1.0 / SECS_PER_DAY
HERA_LAT_LON_ALT = (
    -30.72152777777791,  # deg
    21.428305555555557,  # deg
    1073.0000000093132,  # meters
)


class HealpixLMNModel:
    """
    Interface for the HEALPix image-domain model used to define (l, m, n) grids.

    This class focuses on professional, explicit configuration with clear
    separation between input validation and computed state. Heavy operations
    (e.g., coordinate transforms) are done via astropy/astropy-healpix.

    Parameters
    ----------
    fov_ra_eor : float
        Field of view in degrees of the RA axis for the EoR model.
    fov_dec_eor : float, optional
        Field of view in degrees of the DEC axis for the EoR model.
        Defaults to `fov_ra_eor`.
    fov_ra_fg : float, optional
        Field of view in degrees of the RA axis for the FG model.
    fov_dec_fg : float, optional
        Field of view in degrees of the DEC axis for the FG model.
        Defaults to `fov_ra_fg`.
    simple_za_filter : bool, optional
        If True, filter pixels using a zenith-angle cut.
        Otherwise, use a rectangular RA/DEC box.
    nside : int, optional
        HEALPix Nside resolution. Defaults to 256.
    telescope_latlonalt : tuple, optional
        (lat_deg, lon_deg, alt_m). Defaults to HERA location.
    jd_center : float
        Central time step of the observation in JD.
    nt : int, optional
        Number of time integrations. Defaults to 1.
    dt : float, optional
        Integration time in seconds. Required if ``nt > 1``.
    beam_type : str, optional
        Beam model identifier or path to a UVBeam file.
    peak_amp : float, optional
        Peak amplitude of the beam. Defaults to 1.0.
    fwhm_deg : float, optional
        Full width at half maximum (degrees) for analytic beam models.
    diam : float, optional
        Antenna diameter in meters for Airy beam models.
    cosfreq : float, optional
        Cosine frequency for gausscosine beams.
    tanh_freq : float, optional
        Exponential frequency for tanhairy beams.
    tanh_sl_red : float, optional
        Airy sidelobe reduction for tanhairy beams.
    pol : str, optional
        Polarization string for UVBeam selection. Defaults to "xx".
    freq_interp_kind : str, optional
        UVBeam frequency interpolation kind. Defaults to "cubic".
    """

    def __init__(
        self,
        *,
        fov_ra_eor: float,
        fov_dec_eor: Optional[float] = None,
        fov_ra_fg: Optional[float] = None,
        fov_dec_fg: Optional[float] = None,
        simple_za_filter: bool = False,
        nside: int = 256,
        telescope_latlonalt: Tuple[float, float, float] = HERA_LAT_LON_ALT,
        jd_center: Optional[float] = None,
        nt: int = 1,
        dt: Optional[float] = None,
        beam_type: Optional[str] = None,
        peak_amp: float = 1.0,
        fwhm_deg: Optional[float] = None,
        diam: Optional[float] = None,
        cosfreq: Optional[float] = None,
        tanh_freq: Optional[float] = None,
        tanh_sl_red: Optional[float] = None,
        pol: str = "xx",
        freq_interp_kind: str = "cubic",
    ) -> None:
        self._require_astropy()

        self.fov_ra_eor = fov_ra_eor
        self.fov_dec_eor = fov_dec_eor or fov_ra_eor

        if fov_ra_fg is None:
            self.fov_ra_fg = self.fov_ra_eor
            self.fov_dec_fg = self.fov_dec_eor
            self.fovs_match = True
        else:
            if fov_ra_fg < self.fov_ra_eor:
                raise ValueError("fov_ra_fg must be >= fov_ra_eor.")
            self.fov_ra_fg = fov_ra_fg
            self.fov_dec_fg = fov_dec_fg or fov_ra_fg
            self.fovs_match = bool(
                (self.fov_ra_eor == self.fov_ra_fg)
                and (self.fov_dec_eor == self.fov_dec_fg)
            )

        self.simple_za_filter = simple_za_filter
        self.nside = nside
        self.jd_center = jd_center
        self.nt = nt
        self.dt = dt
        self.peak_amp = peak_amp
        self.beam_type = beam_type or "uniform"
        self.fwhm_deg = fwhm_deg
        self.diam = diam
        self.cosfreq = cosfreq
        self.tanh_freq = tanh_freq
        self.tanh_sl_red = tanh_sl_red
        self.pol = pol
        self.freq_interp_kind = freq_interp_kind

        self._healpix = ahp.HEALPix(nside, frame=ICRS())
        self.pixel_area_sr = self._healpix.pixel_area.to("sr").value

        self.tele_lat, self.tele_lon, self.tele_alt = telescope_latlonalt
        self.telescope_location = self._build_telescope_location(
            self.tele_lat, self.tele_lon, self.tele_alt
        )

        self.field_center = self._compute_field_center()
        self.time_inds, self.jds = self._build_time_axis()
        self.pointing_centers = self._compute_pointing_centers()

        self._configure_beam()
        self._compute_pixel_filters()

    @staticmethod
    def _require_astropy() -> None:
        if ahp is None or hp is None or Time is None:
            raise ImportError(
                "astropy-healpix/astropy/pyuvdata are required for HealpixLMNModel."
            )

    @staticmethod
    def _build_telescope_location(lat_deg: float, lon_deg: float, alt_m: float) -> EarthLocation:
        telescope_xyz = uvutils.XYZ_from_LatLonAlt(
            lat_deg * np.pi / 180.0,
            lon_deg * np.pi / 180.0,
            alt_m,
        )
        return EarthLocation.from_geocentric(*telescope_xyz, unit="m")

    def _compute_field_center(self) -> Tuple[float, float]:
        if self.jd_center is None:
            raise ValueError("jd_center must be provided to compute field center.")
        t = Time(self.jd_center, scale="utc", format="jd")
        zen = AltAz(
            alt=Angle("90d"),
            az=Angle("0d"),
            obstime=t,
            location=self.telescope_location,
        )
        zen_radec = zen.transform_to(ICRS())
        return (zen_radec.ra.deg, zen_radec.dec.deg)

    def _build_time_axis(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.nt % 2:
            time_inds = np.arange(-(self.nt // 2), self.nt // 2 + 1)
        else:
            time_inds = np.arange(-(self.nt // 2), self.nt // 2)
        if self.dt is None:
            jds = np.array([self.jd_center])
        else:
            jds = self.jd_center + time_inds * self.dt * DAYS_PER_SEC
        return time_inds, jds

    def _compute_pointing_centers(self) -> Sequence[Tuple[float, float]]:
        centers = []
        for jd in self.jds:
            t = Time(jd, scale="utc", format="jd")
            zen = AltAz(
                alt=Angle("90d"),
                az=Angle("0d"),
                obstime=t,
                location=self.telescope_location,
            )
            zen_radec = zen.transform_to(ICRS())
            centers.append((zen_radec.ra.deg, zen_radec.dec.deg))
        return centers

    def _configure_beam(self) -> None:
        if self.beam_type is None or self.beam_type == "uniform":
            self.beam_type = "uniform"
            self.uvb = None
            return

        if "." not in str(self.beam_type):
            self.beam_type = str(self.beam_type).lower()
            self.uvb = None
            return

        uvb = UVBeam()
        uvb.read_beamfits(self.beam_type)
        if uvb.beam_type == "efield" and self.pol in ["xx", "yy"]:
            uvb.efield_to_power()
        elif uvb.beam_type == "efield" and self.pol == "pI":
            uvb.efield_to_pstokes()
        uvb.select(polarizations=[uvutils.polstr2num(self.pol)])
        uvb.freq_interp_kind = self.freq_interp_kind
        uvb.interpolation_function = (
            "healpix_simple" if uvb.pixel_coordinate_system == "healpix" else "az_za_simple"
        )
        self.beam_type = "uvbeam"
        self.uvb = uvb

    def _compute_pixel_filters(self) -> None:
        pix_eor, ra_eor, dec_eor = self.get_pixel_filter(
            self.fov_ra_eor,
            self.fov_dec_eor,
            return_radec=True,
            simple_za_filter=self.simple_za_filter,
        )
        self.pix_eor = pix_eor
        self.ra_eor = ra_eor
        self.dec_eor = dec_eor
        self.npix_fov_eor = self.pix_eor.size

        if self.fovs_match:
            self.pix_fg = self.pix_eor.copy()
            self.ra_fg = self.ra_eor.copy()
            self.dec_fg = self.dec_eor.copy()
            self.npix_fov_fg = self.pix_fg.size
        else:
            pix_fg, ra_fg, dec_fg = self.get_pixel_filter(
                self.fov_ra_fg,
                self.fov_dec_fg,
                return_radec=True,
                simple_za_filter=self.simple_za_filter,
            )
            self.pix_fg = pix_fg
            self.ra_fg = ra_fg
            self.dec_fg = dec_fg
            self.npix_fov_fg = self.pix_fg.size

        self.pix = self.pix_fg
        self.ra = self.ra_fg
        self.dec = self.dec_fg
        self.npix_fov = self.npix_fov_fg
        self.fov_ra = self.fov_ra_fg
        self.fov_dec = self.fov_dec_fg
        self.eor_to_fg_pix = np.in1d(self.pix_fg, self.pix_eor)

    def get_pixel_filter(
        self,
        fov_ra: float,
        fov_dec: float,
        *,
        return_radec: bool = False,
        inverse: bool = False,
        simple_za_filter: bool = True,
    ):
        """
        Return HEALPix pixel indices lying inside the observed region.
        """
        lons, lats = hp.pix2ang(self.nside, np.arange(self._healpix.npix), lonlat=True)
        if simple_za_filter:
            _, _, _, _, za = self.calc_lmn_from_radec(
                self.jd_center, lons, lats, return_azza=True
            )
            max_za = np.deg2rad(fov_ra) / 2
            pix = np.where(za <= max_za)[0]
        else:
            thetas = (90 - lats) * np.pi / 180
            if self.field_center[0] - fov_ra / 2 < 0:
                lons[lons > 180] -= 360
            lons_inds = np.logical_and(
                (lons - self.field_center[0]) * np.sin(thetas) >= -fov_ra / 2,
                (lons - self.field_center[0]) * np.sin(thetas) <= fov_ra / 2,
            )
            lats_inds = np.logical_and(
                lats >= self.field_center[1] - fov_dec / 2,
                lats <= self.field_center[1] + fov_dec / 2,
            )
            if inverse:
                pix = np.where(np.logical_not(lons_inds * lats_inds))[0]
            else:
                pix = np.where(lons_inds * lats_inds)[0]
            lons[lons < 0] += 360

        if not return_radec:
            return pix
        return pix, lons[pix], lats[pix]

    def calc_lmn_from_radec(
        self, time: float, ra: np.ndarray, dec: np.ndarray, *, return_azza: bool = False
    ):
        """
        Return arrays of (l, m, n) coordinates in radians for all (RA, DEC).
        """
        if not isinstance(time, Time):
            time = Time(time, format="jd")

        skycoord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        altaz = skycoord.transform_to(AltAz(obstime=time, location=self.telescope_location))
        az = altaz.az.rad
        za = np.pi / 2 - altaz.alt.rad

        ls = np.sin(za) * np.sin(az)
        ms = np.sin(za) * np.cos(az)
        ns = np.cos(za)

        if return_azza:
            return ls, ms, ns, az, za
        return ls, ms, ns
