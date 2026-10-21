"""
HEALPix image-domain geometry for BayesEoR-JAX.

This module builds the fixed sky pixel sets used by the image-domain portions of
the pipeline. The key design is:

- select HEALPix pixels that enter the requested EoR/FG fields of view at
  **any** configured time
- hold those selected sky pixels fixed thereafter
- evaluate their time-dependent ``(l, m, n)`` and ``(az, za)`` coordinates for
  the visibility response

The same class also supports an optional multiresolution foreground geometry for
image-domain foreground modeling:

- a central high-resolution core FoV remains at the native HEALPix ``nside``
- the outer foreground region can be grouped onto a lower-resolution HEALPix
  parent grid
- grouping is still defined on the full high-resolution union mask, so the
  resulting grouped model remains drift-scan aware
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
    HEALPix image-domain model used to define ``(l, m, n)`` and ``(az, za)`` grids.

    Parameters
    ----------
    fov_eor : float
        Circular field of view in degrees for the EoR model.
    fov_fg : float, optional
        Circular field of view in degrees for the foreground model.
        Defaults to `fov_eor`.
    fg_core_fov : float, optional
        Optional central high-resolution foreground FoV in degrees. When set
        together with ``fg_outer_nside``, the foreground image model becomes
        multiresolution: pixels that ever enter this core FoV remain at full
        resolution, while the outer region may be grouped.
    fg_outer_nside : int, optional
        Optional lower HEALPix ``nside`` used to group the outer foreground
        region. Grouping is only active when both ``fg_core_fov`` and
        ``fg_outer_nside`` are provided and satisfy
        ``fg_core_fov < fov_fg`` and ``fg_outer_nside < nside``.
    nside : int, optional
        HEALPix Nside resolution. Defaults to 128.
    sky_mask_time_mode : {"union", "center"}, optional
        How the fixed HEALPix sky mask is selected across time. ``"union"``
        keeps pixels that enter the requested FoV at any configured time.
        ``"center"`` uses only the central timestep FoV, even if ``n_t > 1``.
    telescope_latlonalt : tuple, optional
        (lat_deg, lon_deg, alt_m). Defaults to HERA location.
    jd_center : float
        Central time step of the observation in JD units.
        Defaults to 2461040.5 (2026-01-01 00:00:00 UTC).
    n_t : int, optional
        Number of time integrations. Defaults to 1.
    d_t : float, optional
        Integration time in seconds. Required if ``n_t > 1``.

    Attributes
    ----------
    times_jd : np.ndarray
        Observation times in Julian Date with shape (n_t,).
    lmn_fg : jnp.ndarray
        (l, m, n) array for FG pixels with shape (n_t, n_pix_fg, 3).
    az_za_fg : jnp.ndarray
        (az, za) array for FG pixels with shape (n_t, n_pix_fg, 2).
    eor_fg_shared_axis : jnp.ndarray
        Boolean mask over FG pixels indicating which are also in the EoR FoV,
        shape (n_pix_fg,).
    fg_group_labels : jnp.ndarray
        Integer group label for every high-resolution FG pixel. Equal labels
        indicate that those child pixels share one image-domain foreground
        coefficient in the multiresolution model.
    fg_group_pixel_counts : jnp.ndarray
        Number of native high-resolution FG pixels represented by each spatial
        group. This records the grouped geometry for diagnostics and provides
        the spatial-group layout used when constructing the image-domain FG
        prior. It does not scale the prior precision by group size.
    fg_group_is_highres : jnp.ndarray
        Boolean flag per spatial group. ``True`` means this group corresponds to
        a single retained high-resolution FG pixel. ``False`` means it is a
        grouped outer low-resolution parent built from multiple high-resolution
        children.
    n_fg_groups : int
        Number of effective foreground spatial groups. This equals ``n_pix_fg``
        in the old full-resolution image-domain model and is smaller whenever
        multiresolution grouping is active.
    angular_lm_extent_eor, angular_lm_extent_fg : dict
        Central-time local angular-offset extent summaries on the fixed
        HEALPix pixel sets. These are built from the full ``(l, m, n)``
        vectors using ``atan2(l, n)`` and ``atan2(m, n)`` so that wide-field
        support is measured as an angular width rather than a direction-cosine
        width.
    """
    # Default Location for Telescope (lat_deg, lon_deg, alt_m): HERA site in South Africa
    hera_location = (-30.72152777777791, 21.428305555555557, 1073.0000000093132)  

    def __init__(
        self,
        fov_eor: float,
        fov_fg: Optional[float] = None,
        fg_core_fov: Optional[float] = None,
        fg_outer_nside: Optional[int] = None,
        nside: int = 128,
        sky_mask_time_mode: str = "union",
        telescope_latlonalt: Tuple[float, float, float] = hera_location,
        jd_center: float = 2461040.5,
        n_t: int = 1,
        d_t: float = 60,
    ) -> None:
        
        # Store the field of view parameters for EoR and foreground models.
        self.fov_eor = fov_eor
        # If not foreground FoV is provided, use the EoR FoV as default.
        self.fov_fg = fov_fg if fov_fg is not None else fov_eor
        self.fg_core_fov = fg_core_fov
        self.fg_outer_nside = fg_outer_nside
        # Check if the EoR and foreground FoVs are the same.
        self.fovs_match = bool(self.fov_fg == self.fov_eor)

        # Store the HEALPix resolution and telescope location parameters.
        self.nside = nside
        self.npix = 12 * nside**2
        self.pixel_area_sr = jhp.nside2pixarea(nside)
        self.sky_mask_time_mode = str(sky_mask_time_mode).lower()
        if self.sky_mask_time_mode not in {"union", "center"}:
            raise ValueError(
                "sky_mask_time_mode must be one of {'union', 'center'}, "
                f"got {sky_mask_time_mode!r}."
            )
        self.telescope_latlonalt = telescope_latlonalt

        # Store the time parameters for constructing the time array and pointing centers.
        self.jd_center = jd_center
        self.n_t = n_t
        self.d_t = d_t
        # Construct the array of observation time stamps in Julian Date (JD).
        self.times_jd, self.time_ind_center = self._construct_time_array(jd_center, n_t, d_t)

        # Telescope location in ICRS transforms.
        lat, lon, alt = telescope_latlonalt
        self.telescope_location = EarthLocation.from_geodetic(lon=lon * u.deg, lat=lat * u.deg, height=alt * u.m)
        # Determine the RA and Dec of the pointing center for the central time step.
        self.field_center = self._observation_field_center(self.jd_center)
        self.field_centers = [self._observation_field_center(float(jd)) for jd in self.times_jd]
        # Compute the HEALPix pixel indices that enter the EoR and foreground
        # FoVs at any configured time. This gives a single fixed sky pixel set
        # for basis definitions while Finv handles the time-dependent response.
        self.pix_indices_eor, self.pix_indices_fg, self.eor_fg_shared_axis = self._healpix_pixels_fov(
            self.field_centers, (self.fov_eor, self.fov_fg)
        )
        self.n_pix_eor = len(self.pix_indices_eor)
        self.n_pix_fg = len(self.pix_indices_fg)

        # Compute the (l, m, n) coordinates and (az, za) for FG pixels.
        self.lmn_fg, self.az_za_fg = self._lmn_in_fov(self.times_jd)
        self._refresh_fg_multiresolution()

        self.angular_lm_extent_fg = self._angular_lm_extent(
            self.lmn_fg[self.time_ind_center]
        )
        self.angular_lm_extent_eor = self._angular_lm_extent(
            self.lmn_fg[self.time_ind_center, :, self.eor_fg_shared_axis]
        )
        self.fov_l_angle_eor = self.angular_lm_extent_eor["l_angle_range_padded"]
        self.fov_m_angle_eor = self.angular_lm_extent_eor["m_angle_range_padded"]
        self.fov_l_angle_fg = self.angular_lm_extent_fg["l_angle_range_padded"]
        self.fov_m_angle_fg = self.angular_lm_extent_fg["m_angle_range_padded"]
        self.n_fg_pixels_trimmed = 0


    def _construct_time_array(self, jd_center: float, n_t: int, d_t: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Create Julian Date time stamps centered on the supplied JD.

        Returns
        -------
        jd_array : np.ndarray
            Time stamps in JD with shape (n_t,).
        """
        # Construct time indices centered on zero (left-of-center when even).
        if n_t % 2:
            time_inds = np.arange(-(n_t // 2), n_t // 2 + 1)
        else:
            time_inds = np.arange(-(n_t // 2), n_t // 2)

        dt_days = d_t / (24 * 60 * 60)
        jd_array = jd_center + time_inds * dt_days

        # Index of the center time (where time_inds == 0).
        center_idx = n_t // 2

        return jd_array, center_idx

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

    def _query_disc_union(
        self,
        field_centers: Sequence[Tuple[float, float]],
        radius_rad: jnp.ndarray,
    ) -> np.ndarray:
        """
        Return HEALPix pixels inside a spherical disc under the configured time-mask rule.

        Under ``sky_mask_time_mode="union"``, the returned set contains every
        pixel whose centre enters the supplied disc at any configured time.
        Under ``"center"``, only the central timestep disc is used. This is
        the basic operation used both for the full FG/EoR FoVs and for the
        optional high-resolution FG core.
        """
        if self.sky_mask_time_mode == "center":
            field_centers = [field_centers[self.time_ind_center]]
        indices: list[np.ndarray] = []
        for field_center in field_centers:
            ra_rad = jnp.radians(field_center[0])
            dec_rad = jnp.radians(field_center[1])
            x = jnp.cos(dec_rad) * jnp.cos(ra_rad)
            y = jnp.cos(dec_rad) * jnp.sin(ra_rad)
            z = jnp.sin(dec_rad)
            vec_center = jnp.array([x, y, z])
            pix = jhp.query_disc(
                nside=self.nside,
                vec=vec_center,
                radius=radius_rad,
                inclusive=False,
            )
            pix = np.asarray(pix[pix < self.npix], dtype=np.int64)
            indices.append(pix)
        if not indices:
            return np.asarray([], dtype=np.int64)
        return np.unique(np.concatenate(indices))

    def _healpix_pixels_fov(
        self,
        field_centers: Sequence[Tuple[float, float]],
        fov_deg: Tuple[float, float],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return HEALPix pixel indices within the time-union EoR and FG FoVs.

        Parameters
        ----------
        field_centers : sequence of tuple
            ``(ra_deg, dec_deg)`` FoV centers for each configured time.
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
        # Compute the angular radius of the FoV in radians for the EoR and FG models.
        radius_rad_eor = jnp.radians(fov_deg[0] / 2)
        radius_rad_fg = jnp.radians(fov_deg[1] / 2)

        pix_indices_eor_np = self._query_disc_union(field_centers, radius_rad_eor)
        if self.fovs_match:
            pix_indices_fg_np = pix_indices_eor_np
        else: 
            pix_indices_fg_np = self._query_disc_union(field_centers, radius_rad_fg)

        pix_indices_eor = jnp.asarray(pix_indices_eor_np, dtype=jnp.int64)
        pix_indices_fg = jnp.asarray(pix_indices_fg_np, dtype=jnp.int64)
        eor_fg_shared_axis = jnp.isin(pix_indices_fg, pix_indices_eor)
        return pix_indices_eor, pix_indices_fg, eor_fg_shared_axis

    def _refresh_fg_multiresolution(self) -> None:
        """
        Build multiresolution FG grouping labels on the current high-res FG mask.

        The grouping rule is conservative at the core boundary:

        - pixels that ever enter the high-resolution core FoV stay high
          resolution
        - if a lower-resolution parent contains **any** high-resolution core
          child, then all FG children of that parent remain high resolution
        - only fully outer parents, which are untouched by the core, are
          collapsed into grouped outer coefficients

        At the outer FG boundary the grouped parents may be only partially
        populated by the time-union mask. In that case the grouped coefficient
        simply represents the subset of high-resolution children that are
        actually present in the FG union mask.

        The output of this routine is a grouping map from native high-res FG
        pixels to effective image-domain spatial coefficients:

        - ``fg_group_labels[p]`` gives the effective coefficient index for
          high-resolution FG pixel ``p``
        - ``fg_group_pixel_counts[g]`` is the number of native FG pixels
          represented by grouped coefficient ``g``

        Downstream, the image-domain FG block of ``T`` is reduced by summing
        the high-resolution child columns that share a group label.
        """
        pix_fg = np.asarray(self.pix_indices_fg, dtype=np.int64)
        n_pix_fg = int(pix_fg.size)
        self.pix_indices_fg_core = np.asarray([], dtype=np.int64)
        self.fg_group_labels = np.arange(n_pix_fg, dtype=np.int64)
        self.fg_group_pixel_counts = np.ones((n_pix_fg,), dtype=np.int32)
        self.fg_group_is_highres = np.ones((n_pix_fg,), dtype=bool)
        self.n_fg_groups = int(n_pix_fg)
        self.use_multires_fg = False

        if (
            self.fg_core_fov is None
            or self.fg_outer_nside is None
            or int(self.fg_outer_nside) >= int(self.nside)
            or float(self.fg_core_fov) >= float(self.fov_fg)
        ):
            return

        nside_lo = int(self.fg_outer_nside)
        nside_hi = int(self.nside)
        if nside_hi % nside_lo != 0:
            raise ValueError(
                f"fg_outer_nside={nside_lo} must divide nside={nside_hi}."
            )
        ratio = nside_hi // nside_lo
        if ratio & (ratio - 1):
            raise ValueError(
                "fg_outer_nside must differ from nside by a power-of-two factor "
                f"(got nside={nside_hi}, fg_outer_nside={nside_lo})."
            )

        radius_rad_core = jnp.radians(float(self.fg_core_fov) / 2.0)
        pix_indices_core_np = self._query_disc_union(self.field_centers, radius_rad_core)
        core_mask = np.isin(pix_fg, pix_indices_core_np)
        self.pix_indices_fg_core = np.asarray(
            pix_fg[core_mask],
            dtype=np.int64,
        )

        child_per_parent = int(ratio * ratio)
        pix_fg_nested = np.asarray(jhp.ring2nest(nside_hi, pix_fg), dtype=np.int64)
        parent_nested = pix_fg_nested // child_per_parent
        parent_ids_with_core = np.unique(parent_nested[core_mask])
        # Any low-resolution parent touched by the high-resolution core is
        # promoted back to fully high-resolution treatment. This avoids a
        # grouped parent straddling the core boundary.
        core_touched_mask = np.isin(parent_nested, parent_ids_with_core)

        labels = np.full((n_pix_fg,), -1, dtype=np.int64)
        group_is_highres: list[bool] = []
        next_label = 0

        hi_axis = np.flatnonzero(core_touched_mask)
        if hi_axis.size:
            labels[hi_axis] = np.arange(next_label, next_label + hi_axis.size, dtype=np.int64)
            group_is_highres.extend([True] * int(hi_axis.size))
            next_label += int(hi_axis.size)

        outer_axis = np.flatnonzero(~core_touched_mask)
        if outer_axis.size:
            outer_parents = parent_nested[outer_axis]
            unique_outer_parents = np.unique(outer_parents)
            parent_to_label = {
                int(parent): int(next_label + i)
                for i, parent in enumerate(unique_outer_parents.tolist())
            }
            labels[outer_axis] = np.asarray(
                [parent_to_label[int(parent)] for parent in outer_parents],
                dtype=np.int64,
            )
            group_is_highres.extend([False] * int(unique_outer_parents.size))
            next_label += int(unique_outer_parents.size)

        if np.any(labels < 0):
            raise ValueError("Failed to assign all FG pixels to multiresolution groups.")

        group_counts = np.bincount(labels, minlength=next_label).astype(np.int32)
        self.fg_group_labels = np.asarray(labels, dtype=np.int64)
        self.fg_group_pixel_counts = np.asarray(group_counts, dtype=np.int32)
        self.fg_group_is_highres = np.asarray(group_is_highres, dtype=bool)
        self.n_fg_groups = int(next_label)
        self.use_multires_fg = True

    def _angular_lm_extent(self, lmn_coords: jnp.ndarray) -> dict[str, float]:
        """
        Summarize central-time angular support along the local ``l`` and ``m`` axes.

        The current wide-field image chart is still anchored to the central-time
        local frame, but the raw direction cosines ``l`` and ``m`` compress large
        angular offsets through the sine projection. To recover angular widths
        that remain comparable to the spherical FoV, we define angular offsets
        along the two local transverse axes as:

        ``l_angle = atan2(l, n)``
        ``m_angle = atan2(m, n)``

        where ``n`` is the local line-of-sight component. For small fields these
        reduce to the usual flat-sky coordinates, while for wide fields they
        retain the correct angular scaling. A single HEALPix pixel length in
        radians is added in total, equivalent to padding by half a pixel on
        either side.
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

        l = lmn[0]
        m = lmn[1]
        n = lmn[2]
        l_angle = np.arctan2(l, n)
        m_angle = np.arctan2(m, n)
        pixel_angle = float(np.sqrt(float(self.pixel_area_sr)))
        half_pixel_angle = 0.5 * pixel_angle

        l_angle_min = float(l_angle.min())
        l_angle_max = float(l_angle.max())
        m_angle_min = float(m_angle.min())
        m_angle_max = float(m_angle.max())

        return {
            "pixel_angle": pixel_angle,
            "l_angle_min": l_angle_min,
            "l_angle_max": l_angle_max,
            "l_angle_range": float(l_angle_max - l_angle_min),
            "l_angle_min_padded": float(l_angle_min - half_pixel_angle),
            "l_angle_max_padded": float(l_angle_max + half_pixel_angle),
            "l_angle_range_padded": float((l_angle_max - l_angle_min) + pixel_angle),
            "m_angle_min": m_angle_min,
            "m_angle_max": m_angle_max,
            "m_angle_range": float(m_angle_max - m_angle_min),
            "m_angle_min_padded": float(m_angle_min - half_pixel_angle),
            "m_angle_max_padded": float(m_angle_max + half_pixel_angle),
            "m_angle_range_padded": float((m_angle_max - m_angle_min) + pixel_angle),
        }


    def _lmn_in_fov(
        self, jds: np.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return (l, m, n) and (az, za) arrays for the preselected pixels over time.

        Returns
        -------
        lmn_fg : jnp.ndarray
            Array with shape (n_t, 3, n_pix_fg) containing (l, m, n) for the foreground pixels.
        az_za_fg : jnp.ndarray
            Array with shape (n_t, 2, n_pix_fg) containing (az, za) for the foreground pixels.
        """
        # Get the RA and Dec coordinates for the preselected FG HEALPix pixel indices.
        ra_fg_deg, dec_fg_deg = jhp.pix2ang(self.nside, self.pix_indices_fg, lonlat=True)

        # Create SkyCoord objects for the foreground pixels in the ICRS frame.
        skycoord_fg = SkyCoord(np.array(ra_fg_deg) * u.deg, np.array(dec_fg_deg) * u.deg, frame="icrs")

        # Initialize arrays to hold the (l, m, n) coordinates for each time step and pixel.
        n_t = len(jds)
        lmn_fg = jnp.zeros((n_t, 3, self.n_pix_fg), dtype=jnp.float64) # Shape (n_t, 3, n_pix_fg)
        az_za_fg = jnp.zeros((n_t, 2, self.n_pix_fg), dtype=jnp.float64) # Shape (n_t, 2, n_pix_fg)

        for i_t, jd in enumerate(jds):
            # Compute the AltAz coordinates for the given time step and telescope location.
            t = Time(jd, scale="utc", format="jd")
            # Foreground pixels only.
            altaz_fg = skycoord_fg.transform_to(AltAz(obstime=t, location=self.telescope_location))
            az_fg = jnp.array(altaz_fg.az.rad)
            za_fg = jnp.pi / 2 - jnp.array(altaz_fg.alt.rad)
            # Convert from (az, za) to (l, m, n) for the foreground pixels.
            l_fg = jnp.sin(za_fg) * jnp.sin(az_fg)
            m_fg = jnp.sin(za_fg) * jnp.cos(az_fg)
            n_fg = jnp.cos(za_fg)
            # Insert the (l, m, n) and (az, za) coordinates for this time step.
            lmn_fg = lmn_fg.at[i_t].set(jnp.stack([l_fg, m_fg, n_fg], axis=0)) 
            az_za_fg = az_za_fg.at[i_t].set(jnp.stack([az_fg, za_fg], axis=0))

        return lmn_fg, az_za_fg
    

    def _refresh_fg_state(self) -> None:
        """
        Recompute all FG-dependent cached quantities after a pixel trim.

        This refreshes both the ordinary FG geometry and the optional
        multiresolution grouping metadata so that later T-building code always
        sees a self-consistent FG pixel/group description. In particular,
        trimming pixels changes:

        - the FG HEALPix mask itself
        - the time-dependent ``(l, m, n)`` and ``(az, za)`` coordinates
        - the EoR/FG shared axis
        - any multiresolution group labels/counts
        - the angular support summaries later used for auto-sizing
        """
        self.n_pix_fg = len(self.pix_indices_fg)
        self.eor_fg_shared_axis = jnp.isin(self.pix_indices_fg, self.pix_indices_eor)
        self.lmn_fg, self.az_za_fg = self._lmn_in_fov(self.times_jd)
        self._refresh_fg_multiresolution()
        self.angular_lm_extent_fg = self._angular_lm_extent(
            self.lmn_fg[self.time_ind_center]
        )
        self.angular_lm_extent_eor = self._angular_lm_extent(
            self.lmn_fg[self.time_ind_center, :, self.eor_fg_shared_axis]
        )
        self.fov_l_angle_eor = self.angular_lm_extent_eor["l_angle_range_padded"]
        self.fov_m_angle_eor = self.angular_lm_extent_eor["m_angle_range_padded"]
        self.fov_l_angle_fg = self.angular_lm_extent_fg["l_angle_range_padded"]
        self.fov_m_angle_fg = self.angular_lm_extent_fg["m_angle_range_padded"]

    def trim_fg_outer_pixels(self, n_trim: int) -> int:
        """
        Remove ``n_trim`` FG pixels furthest from the central FoV centre.

        Pixels are ranked in the central-time image chart by ``l^2 + m^2`` and
        the largest values are removed. This gives the simple "trim the outer
        edge of the circular FoV" behavior desired for image-domain FG runs.
        After trimming, the full FG geometry and any multiresolution grouping
        metadata are rebuilt on the reduced mask.

        Notes
        -----
        This intentionally uses the current image-domain chart rather than a
        HEALPix neighbour-shell construction. The goal is simply to remove the
        outermost FG image pixels first so the image-domain parameter count can
        be adjusted without changing the interior support.
        """
        n_trim = int(n_trim)
        if n_trim <= 0:
            return 0
        if n_trim >= int(self.n_pix_fg):
            raise ValueError(
                f"Cannot trim {n_trim} FG pixels from a mask with only {int(self.n_pix_fg)} pixels."
            )

        lmn_center = np.asarray(self.lmn_fg[self.time_ind_center], dtype=float)
        r2 = np.square(lmn_center[0]) + np.square(lmn_center[1])
        order = np.argsort(r2)[::-1]
        drop_axis = np.sort(order[:n_trim])

        keep_mask = np.ones((int(self.n_pix_fg),), dtype=bool)
        keep_mask[drop_axis] = False
        self.pix_indices_fg = jnp.asarray(
            np.asarray(self.pix_indices_fg, dtype=np.int64)[keep_mask],
            dtype=jnp.int64,
        )
        self.n_fg_pixels_trimmed += n_trim
        self._refresh_fg_state()
        return n_trim

    def trim_fg_outer_pixels_for_total_param_divisibility(
        self,
        *,
        n_modes_fg: int,
        n_uv_eor: int,
        n_eta: int,
        divisor: int = 4,
    ) -> int:
        """
        Trim outer FG pixels until the total linear parameter count is divisible.

        Parameters
        ----------
        n_modes_fg : int
            Number of FG/LSSM spectral modes coupled to each non-monopole FG
            pixel.
        n_uv_eor : int
            Number of EoR UV modes.
        n_eta : int
            Number of line-of-sight Fourier modes.
        divisor : int, optional
            Required divisor for the total linear parameter count. Defaults to 4.

        Notes
        -----
        This helper exists for the optional image-domain trimming path used to
        make the **full** linear parameter vector divisible by a given factor.
        The arithmetic is done here because the number of trimmable FG pixels,
        and the optional multiresolution regrouping after each trim, are both
        owned by ``LMN_ModelSpace``. The method still receives the external
        matrix-layout counts explicitly rather than reaching into
        ``BuildMatrices`` state.

        For image-domain FG, each retained spatial mode contributes
        ``1 + n_modes_fg`` non-monopole parameters:

        - one ``eta0`` coefficient
        - ``n_modes_fg`` LSSM coefficients

        The total parameter count checked here is therefore:

        ``n_uv_eor * (n_eta - 1)``
        ``+ n_fg_groups * (1 + n_modes_fg)``
        Because trimming is only needed to make the *final* parameter axis in
        ``T`` and ``T_Ninv_T`` divisible by ``divisor``, the check is performed
        after any multiresolution regrouping. If grouping is active, removing a
        single high-resolution outer pixel may or may not change
        ``n_fg_groups``. The routine therefore trims outer pixels iteratively
        until the grouped image-domain parameter count satisfies the required
        divisibility.
        """
        divisor = int(divisor)
        if divisor < 1:
            raise ValueError("divisor must be >= 1")

        per_group_width = 1 + int(n_modes_fg)
        eor_width = int(n_uv_eor) * (int(n_eta) - 1)
        initial_trimmed = int(self.n_fg_pixels_trimmed)

        while True:
            total_width = eor_width + int(self.n_fg_groups) * per_group_width
            if total_width % divisor == 0:
                return int(self.n_fg_pixels_trimmed) - initial_trimmed

            if int(self.n_pix_fg) <= 1:
                raise ValueError(
                    "Unable to make the grouped image-domain parameter count "
                    f"divisible by {divisor}. "
                    f"Current total={total_width}, grouped FG modes={int(self.n_fg_groups)}."
                )

            self.trim_fg_outer_pixels(1)
