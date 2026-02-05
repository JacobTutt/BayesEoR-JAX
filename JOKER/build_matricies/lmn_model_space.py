"""
HEALPix-based image-domain model for constructing (l, m, n) coordinates over time.

For wide-field interferometric data, sky pixels must be mapped into direction
cosines for each time step. This module builds HEALPix pixel centers, applies
field-of-view masks, and computes (l, m, n) for EoR and foreground domains.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u

import jax_healpy as jhp

class LMN_ModelSpace:
    """
    HEALPix image-domain model used to define (l, m, n) grids for a given FoV.

    Parameters
    ----------
    fov_eor : float
        Circular field of view in degrees for the EoR model.
    fov_fg : float, optional
        Circular field of view in degrees for the foreground model.
        Defaults to `fov_eor`.
    nside : int, optional
        HEALPix Nside resolution. Defaults to 256.
    telescope_latlonalt : tuple, optional
        (lat_deg, lon_deg, alt_m). Defaults to HERA location.
    jd_start : float
        Start time step of the observation in JD units.
        Defaults to 2461040.5 (2026-01-01 00:00:00 UTC).
    n_t : int, optional
        Number of time integrations. Defaults to 1.
    d_t : float, optional
        Integration time in seconds. Required if ``n_t > 1``.

    Attributes
    ----------
    times_jd : np.ndarray
        Observation times in Julian Date with shape (n_t,).
    lmn_eor : list[jnp.ndarray]
        Per-time (l, m, n) arrays for EoR pixels, each shape (N_pix_eor_t, 3).
    lmn_fg : list[jnp.ndarray]
        Per-time (l, m, n) arrays for FG pixels, each shape (N_pix_fg_t, 3).
    eor_fg_shared_axis : list[jnp.ndarray]
        Per-time boolean masks over FG pixels indicating which are also in the EoR FoV,
        each shape (N_pix_fg_t,).
    """
    # Default Location for Telescope (lat_deg, lon_deg, alt_m): HERA site in South Africa
    hera_location = (-30.72152777777791, 21.428305555555557, 1073.0000000093132)  

    def __init__(
        self,
        fov_eor: float,
        fov_fg: Optional[float] = None,
        nside: int = 256,
        telescope_latlonalt: Tuple[float, float, float] = hera_location,
        jd_start: float = 2461040.5,
        n_t: int = 1,
        d_t: float = 60,
    ) -> None:
        
        # Store the field of view parameters for EoR and foreground models.
        self.fov_eor = fov_eor
        # If not foreground FoV is provided, use the EoR FoV as default.
        self.fov_fg = fov_fg if fov_fg is not None else fov_eor
        # Check if the EoR and foreground FoVs are the same.
        self.fovs_match = bool(self.fov_fg == self.fov_eor)

        # Store the HEALPix resolution and telescope location parameters.
        self.nside = nside
        self.npix = 12 * nside**2
        self.telescope_latlonalt = telescope_latlonalt

        # Store the time parameters for constructing the time array and pointing centers.
        self.jd_start = jd_start
        self.n_t = n_t
        self.d_t = d_t

        # Telescope location in ICRS transforms.
        lat, lon, alt = telescope_latlonalt
        self.telescope_location = EarthLocation.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=alt * u.m)

        # Construct the array of observation time stamps in Julian Date (JD).
        self.times_jd = self._construct_time_array(jd_start, n_t, d_t)
        # Compute the (l, m, n) coordinates for pixels within the EoR/FG FoVs.
        self.lmn_eor, self.lmn_fg, self.eor_fg_shared_axis = self._lmn_in_fov(self.times_jd)



    def _construct_time_array(self, jd_start: float, n_t: int, d_t: float) -> np.ndarray:
        """
        Create Julian Date time stamps from start time, count, and cadence.
        """
        # If only one time step is requested, return an array with the single start time.
        if n_t == 1:
            jd_array = np.array([jd_start], dtype=float)
        # If multiple time steps are requested, compute the array of time stamps.
        else:
            # Convert integration time from seconds to days and construct the array of Julian Dates.
            dt_days = d_t / (24 * 60 * 60)
            jd_array = jd_start + np.arange(n_t, dtype=float) * dt_days

        return jd_array

    def _lmn_in_fov(
        self, jds: np.ndarray
    ) -> Tuple[Sequence[jnp.ndarray], Sequence[jnp.ndarray], Sequence[jnp.ndarray]]:
        """
        Return (l, m, n) arrays within circular FoVs for each time step.

        Returns
        -------
        lmn_eor : list of jnp.ndarray
            Each element has shape (N_pix_eor_t, 3).
        lmn_fg : list of jnp.ndarray
            Each element has shape (N_pix_fg_t, 3).
        """
        # Get the RA and Dec coordinates of all HEALPix pixel centers.
        ra_deg, dec_deg = jhp.pix2ang(self.nside, jnp.arange(self.npix), lonlat=True)
        # Define these coordinates as a SkyCoord object in the ICRS frame for transformation to AltAz.
        skycoord = SkyCoord(np.array(ra_deg) * u.deg, np.array(dec_deg) * u.deg, frame="icrs")

        # Loop over each time step and compute (l, m, n) for pixels within FoV.
        lmn_eor: list[jnp.ndarray] = []
        lmn_fg: list[jnp.ndarray] = []
        eor_fg_shared_axis: list[jnp.ndarray] = []

        for jd in jds:
            # Compute the AltAz coordinates for the given time step and telescope location.
            t = Time(jd, scale="utc", format="jd")
            altaz = skycoord.transform_to(AltAz(obstime=t, location=self.telescope_location))
            # Extract the azimuth and zenith angle (or equivalently, altitude) in radians for all pixels.
            az = jnp.array(altaz.az.rad)
            za = jnp.pi/2 - jnp.array(altaz.alt.rad)
            # Define a two masks for the pixels which fall within the EoR and foreground FoVs, respectively.
            fov_eor_mask = za <= jnp.radians(self.fov_eor / 2)
            fov_fg_mask = za <= jnp.radians(self.fov_fg / 2)
            
            # Apply the masks the the azimuth and zenith angle arrays to select only the pixels within the respective FoVs.
            az_eor = az[fov_eor_mask]
            za_eor = za[fov_eor_mask]
            az_fg = az[fov_fg_mask]
            za_fg = za[fov_fg_mask]

            # Boolean mask over FG pixels showing which are also in the EoR FoV.
            shared_mask_fg = fov_eor_mask[fov_fg_mask]

            # Convert from (az, za) to (l, m, n) for the EoR and foreground pixels, respectively.
            l_eor = jnp.sin(za_eor) * jnp.sin(az_eor)
            m_eor = jnp.sin(za_eor) * jnp.cos(az_eor)
            n_eor = jnp.cos(za_eor)
            l_fg = jnp.sin(za_fg) * jnp.sin(az_fg)
            m_fg = jnp.sin(za_fg) * jnp.cos(az_fg)
            n_fg = jnp.cos(za_fg)

            # Wrap in two arrays of shape (N_pix_in_fov, 3) for the EoR and foreground pixels, respectively.
            lmn_eor_singletime = jnp.stack([l_eor, m_eor, n_eor], axis=-1)
            lmn_fg_singletime = jnp.stack([l_fg, m_fg, n_fg], axis=-1) 

            lmn_eor.append(lmn_eor_singletime)
            lmn_fg.append(lmn_fg_singletime)
            eor_fg_shared_axis.append(shared_mask_fg)

        return lmn_eor, lmn_fg, eor_fg_shared_axis
