"""
HEALPix-based image-domain model for constructing (l, m, n) and (az, za) over time.

For wide-field interferometric data, sky pixels are mapped into direction cosines
for each time step. This module selects HEALPix pixels within EoR/FG fields of view,
then computes (l, m, n) and (az, za) for those fixed pixel sets across time.
"""

import jax
jax.config.update("jax_enable_x64", True)

from typing import Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u

import jax_healpy as jhp

class LMN_ModelSpace:
    """
    HEALPix image-domain model used to define (l, m, n) and (az, za) grids for a given FoV.

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
    lmn_eor : jnp.ndarray
        (l, m, n) array for EoR pixels with shape (n_t, n_pix_eor, 3).
    lmn_fg : jnp.ndarray
        (l, m, n) array for FG pixels with shape (n_t, n_pix_fg, 3).
    az_za_eor : jnp.ndarray
        (az, za) array for EoR pixels with shape (n_t, n_pix_eor, 2).
    az_za_fg : jnp.ndarray
        (az, za) array for FG pixels with shape (n_t, n_pix_fg, 2).
    eor_fg_shared_axis : jnp.ndarray
        Boolean mask over FG pixels indicating which are also in the EoR FoV,
        shape (n_pix_fg,).
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
        # Construct the array of observation time stamps in Julian Date (JD).
        self.times_jd, self.jd_center = self._construct_time_array(jd_start, n_t, d_t)

        # Telescope location in ICRS transforms.
        lat, lon, alt = telescope_latlonalt
        self.telescope_location = EarthLocation.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=alt * u.m)
        # Determine the RA and Dec of the pointing center for the central time step (used for defining the FoV).
        self.field_center = self._observation_field_center(self.jd_center)
        # Compute the HEALPix pixel indices within the EoR and foreground FoVs
        self.pix_indices_eor, self.pix_indices_fg, self.eor_fg_shared_axis = self._healpix_pixels_fov(
            self.field_center, (self.fov_eor, self.fov_fg)
        )
        self.n_pix_eor = len(self.pix_indices_eor)
        self.n_pix_fg = len(self.pix_indices_fg)

        # Compute the (l, m, n) coordinates and (az, za) for pixels within the EoR/FG FoVs.
        self.lmn_eor, self.lmn_fg, self.az_za_eor, self.az_za_fg = self._lmn_in_fov(self.times_jd)




    def _construct_time_array(self, jd_start: float, n_t: int, d_t: float) -> Tuple[np.ndarray, float]:
        """
        Create Julian Date time stamps and return the center time.

        Returns
        -------
        jd_array : np.ndarray
            Time stamps in JD with shape (n_t,).
        jd_center : float
            Center time (middle value if odd, midpoint if even).
        """
        # If only one time step is requested, return an array with the single start time.
        if n_t == 1:
            jd_array = np.array([jd_start], dtype=float)
        # If multiple time steps are requested, compute the array of time stamps.
        else:
            # Convert integration time from seconds to days and construct the array of Julian Dates.
            dt_days = d_t / (24 * 60 * 60)
            jd_array = jd_start + np.arange(n_t, dtype=float) * dt_days

        # Compute the center of the time array.
        # If odd: middle element is the center
        if jd_array.size % 2 == 1:
            jd_center = jd_array[jd_array.size // 2]
        # If even: average of two middle elements is the center.
        else:
            mid = jd_array.size // 2
            jd_center = 0.5 * (jd_array[mid - 1] + jd_array[mid])

        return jd_array, float(jd_center)

    def _observation_field_center(self, jd: float) -> Tuple[float, float]:
        """
        Compute the RA and Dec of the observation pointing center for a given time step.

        Returns
        -------
        ra_dec : tuple
            (ra_deg, dec_deg) in degrees.
        """
        t = Time(jd, scale="utc", format="jd")
        altaz = AltAz(obstime=t, location=self.telescope_location)
        pointing_center_altaz = SkyCoord(az=0 * u.deg, alt=90 * u.deg, frame=altaz)
        pointing_center_icrs = pointing_center_altaz.transform_to("icrs")
        return (float(pointing_center_icrs.ra.deg), float(pointing_center_icrs.dec.deg))

    def _healpix_pixels_fov(self, field_center: Tuple[float, float], fov_deg: Tuple[float, float]) -> jnp.ndarray:
        """
        Return HEALPix pixel indices within the EoR and FG circular FoVs.

        Parameters
        ----------
        field_center : tuple
            (ra_deg, dec_deg) center of the FoV.
        fov_deg : tuple
            (fov_eor_deg, fov_fg_deg) full diameters in degrees.

        Returns
        -------
        pix_indices_eor : jnp.ndarray
            HEALPix indices for EoR FoV.
        pix_indices_fg : jnp.ndarray
            HEALPix indices for FG FoV.
        eor_fg_shared_axis : jnp.ndarray
            Boolean mask over FG pixels indicating which are also in EoR FoV.
        """
        # Convert the field center from (RA, Dec) to a unit vector in Cartesian coordinates.
        ra_rad = jnp.radians(field_center[0])
        dec_rad = jnp.radians(field_center[1])
        x = jnp.cos(dec_rad) * jnp.cos(ra_rad)
        y = jnp.cos(dec_rad) * jnp.sin(ra_rad)
        z = jnp.sin(dec_rad)
        vec_center = jnp.array([x, y, z])

        # Compute the angular radius of the FoV in radians for the FG and EoR models, respectively.
        radius_rad_eor = jnp.radians(fov_deg[0] / 2)
        radius_rad_fg = jnp.radians(fov_deg[1] / 2)

        # Use healpy to query the pixel indices within the specified radius of the field center.
        pix_indices_eor = jhp.query_disc(nside=self.nside, vec=vec_center, radius=radius_rad_eor, inclusive=False)
        # If the EoR and foreground FoVs are the same, we can skip the second query and just reuse the pixel indices.
        # Also determine a boolean mask over the foreground pixels indicating which are also in the EoR FoV (i.e., shared axis).
        if self.fovs_match:
            pix_indices_fg = pix_indices_eor
            eor_fg_shared_axis = jnp.ones_like(pix_indices_fg, dtype=bool)
        else: 
            pix_indices_fg = jhp.query_disc(nside=self.nside, vec=vec_center, radius=radius_rad_fg, inclusive=False)
            eor_fg_shared_axis = jnp.isin(pix_indices_fg, pix_indices_eor)

        return pix_indices_eor, pix_indices_fg, eor_fg_shared_axis


    def _lmn_in_fov(
        self, jds: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return (l, m, n) and (az, za) arrays for the preselected pixels over time.

        Returns
        -------
        lmn_eor : jnp.ndarray
            Array with shape (n_t, n_pix_eor, 3) containing (l, m, n) for the EoR pixels.
        lmn_fg : jnp.ndarray
            Array with shape (n_t, n_pix_fg, 3) containing (l, m, n) for the foreground pixels.
        az_za_eor : jnp.ndarray
            Array with shape (n_t, n_pix_eor, 2) containing (az, za) for the EoR pixels.
        az_za_fg : jnp.ndarray
            Array with shape (n_t, n_pix_fg, 2) containing (az, za) for the foreground pixels.
        """
        # Get the RA and Dec coordinates for the preselected HEALPix pixel indices.
        ra_eor_deg, dec_eor_deg = jhp.pix2ang(self.nside, self.pix_indices_eor, lonlat=True)
        ra_fg_deg, dec_fg_deg = jhp.pix2ang(self.nside, self.pix_indices_fg, lonlat=True)

        # Create SkyCoord objects for the EoR and foreground pixels in the ICRS frame.
        skycoord_eor = SkyCoord(np.array(ra_eor_deg) * u.deg, np.array(dec_eor_deg) * u.deg, frame="icrs")
        if self.fovs_match:
            skycoord_fg = skycoord_eor
        else:
            skycoord_fg = SkyCoord(np.array(ra_fg_deg) * u.deg, np.array(dec_fg_deg) * u.deg, frame="icrs")

        # Initialize arrays to hold the (l, m, n) coordinates for each time step and pixel.
        n_t = len(jds)
        lmn_eor = jnp.zeros((n_t, self.n_pix_eor, 3), dtype=jnp.float64)
        lmn_fg = jnp.zeros((n_t, self.n_pix_fg, 3), dtype=jnp.float64)
        az_za_eor = jnp.zeros((n_t, self.n_pix_eor, 2), dtype=jnp.float64)
        az_za_fg = jnp.zeros((n_t, self.n_pix_fg, 2), dtype=jnp.float64)

        for i_t, jd in enumerate(jds):
            # Compute the AltAz coordinates for the given time step and telescope location.
            t = Time(jd, scale="utc", format="jd")
            # First for the EoR 
            altaz_eor = skycoord_eor.transform_to(AltAz(obstime=t, location=self.telescope_location))
            # Extract azimuth and zenith angle for the selected pixels.
            az_eor = jnp.array(altaz_eor.az.rad)
            za_eor = jnp.pi / 2 - jnp.array(altaz_eor.alt.rad)
            # Convert from (az, za) to (l, m, n) for the EoR and foreground pixels, respectively.
            l_eor = jnp.sin(za_eor) * jnp.sin(az_eor)
            m_eor = jnp.sin(za_eor) * jnp.cos(az_eor)
            n_eor = jnp.cos(za_eor)
            # Insert the (l, m, n) and (az, za) coordinates for this time step.
            lmn_eor = lmn_eor.at[i_t].set(jnp.stack([l_eor, m_eor, n_eor], axis=-1))
            az_za_eor = az_za_eor.at[i_t].set(jnp.stack([az_eor, za_eor], axis=-1))

            # If FoVs match, we can reuse the EoR (l, m, n) coordinates for the foreground model. 
            if self.fovs_match:
                lmn_fg = lmn_fg.at[i_t].set(jnp.stack([l_eor, m_eor, n_eor], axis=-1))
                az_za_fg = az_za_fg.at[i_t].set(jnp.stack([az_eor, za_eor], axis=-1))
            # Otherwise, we need to compute the (l, m, n) coordinates separately for the foreground pixels.
            else:
                altaz_fg = skycoord_fg.transform_to(AltAz(obstime=t, location=self.telescope_location))
                az_fg = jnp.array(altaz_fg.az.rad)
                za_fg = jnp.pi / 2 - jnp.array(altaz_fg.alt.rad)
                # Convert from (az, za) to (l, m, n) for the foreground pixels.
                l_fg = jnp.sin(za_fg) * jnp.sin(az_fg)
                m_fg = jnp.sin(za_fg) * jnp.cos(az_fg)
                n_fg = jnp.cos(za_fg)
                # Insert the (l, m, n) coordinates for this time step into the corresponding array.
                lmn_fg = lmn_fg.at[i_t].set(jnp.stack([l_fg, m_fg, n_fg], axis=-1))
                az_za_fg = az_za_fg.at[i_t].set(jnp.stack([az_fg, za_fg], axis=-1))

        return lmn_eor, lmn_fg, az_za_eor, az_za_fg
