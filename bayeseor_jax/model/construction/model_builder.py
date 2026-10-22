"""Prepare physical inputs and save independent cosmological/foreground operators.

This builder does not assemble a joint model or prepare likelihood products.
Those stages take explicit saved operator paths, after any foreground reduction.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import c

from ..cosmology.fourier_geometry import FourierGeometry, build_fourier_geometry
from ..observation.image_geometry import ImageGeometry, build_image_geometry
from ..instrument.beam import Beam_Model
from ..operators import (
    build_foreground_spectral_operator,
    build_spectral_fourier_operator,
    build_transverse_fourier_operator,
)
from .operator_construction import T_21_build_save, T_fg_build_save, T_fg_weighted_build_save

jax.config.update("jax_enable_x64", True)


class BuildMatrices:
    r"""Coordinate construction of the cosmological and foreground visibility models.

    Parameters
    ----------
    n_u_eor : int
        Number of cosmological u modes before monopole exclusion.
    du_eor : float
        Cosmological u spacing in inverse radians.
    n_v_eor : int
        Number of cosmological v modes before monopole exclusion.
    dv_eor : float
        Cosmological v spacing in inverse radians.
    beta_fg : float or sequence of float
        Positive foreground spectral-index magnitudes; a constant component is
        included separately.
    n_freq : int
        Number of observing-frequency channels.
    f0_mhz : float
        First observing frequency and foreground reference frequency in MHz.
    df_mhz : float
        Separation of frequency channels in MHz.
    n_eta : int
        Number of line-of-sight Fourier modes before excluding eta=0.
    d_eta : float
        Positive eta spacing and integration weight in inverse Hz (seconds).
    sky_mask_time_mode : str
        Pixel selection: 'union' across integrations or 'center' at the
        reference time.
    telescope_latlonalt : tuple of float
        Geodetic latitude and longitude in degrees, followed by altitude in
        metres.
    jd_center : float
        Reference integration time as a UTC Julian date.
    nside : int
        HEALPix resolution of the equal-area celestial pixel grid.
    fov_eor : float
        Full angular diameter of the cosmological model field in degrees.
    fov_fg : float or None
        Full angular diameter of the foreground model field in degrees; None
        uses fov_eor.
    n_t : int
        Number of observing-time samples.
    d_t : float
        Separation of integration centres in seconds.
    image_geometry : ImageGeometry or None
        Precomputed pixel directions and selections to reuse; None constructs
        them from the field and observing settings.
    fourier_geometry : FourierGeometry or None
        Precomputed (u, v, eta) coordinates and spacings to reuse; None
        constructs them from the grid settings.
    beam_type : str or None
        Analytic beam name or beamfits file path. None selects a uniform beam.
    peak_amp : float
        Peak power gain for Gaussian-based beam profiles; defaults to 1.
    fwhm_deg : float or None
        Configured beam full width at half maximum in degrees, when used instead
        of an aperture-derived width.
    diam : float or None
        Physical or effective aperture diameter in metres for Airy or Gaussian
        beam profiles.
    cosfreq : float or None
        Angular modulation frequency in inverse radians for the gausscosine
        profile.
    tanh_freq : float or None
        Angular taper rate in inverse radians for the tanhairy profile.
    tanh_sl_red : float or None
        Fractional far-off-axis power suppression for the tanhairy profile.
    pol : str
        Polarisation selected from a file-backed beam, for example xx or pI.
    freq_interp_kind : str
        Frequency interpolation method used by UVBeam; defaults to cubic.
    uvw_array_m : array-like
        Baseline coordinates in metres, shape (n_t, n_baseline, 3). The first
        integration supplies the fixed instrument-frame baselines.
    array_save_directory : str or pathlib.Path
        Directory holding the separate physical component matrices.

    Notes
    -----
    The cosmological contribution is represented on a rectilinear
    ``(u, v, eta)`` grid, while smooth foreground brightness is represented in
    the spherical image domain. Both are propagated through the same primary
    beam and full three-dimensional fringe response. Large operators are built
    in process-local CPU row blocks and written to shared ``.npy`` memmaps.

    ``T_21_build_save`` and ``T_fg_build_save`` save the separate coefficient-to-visibility
    mappings. Assembly takes explicit saved component paths, allowing a reduced
    foreground representation to replace the original before likelihood products
    are prepared. Geometry and beam values are prepared at construction; the
    large visibility matrices are created only when a build method runs.
    """

    def __init__(
        self,
        n_u_eor: int,
        du_eor: float,
        n_v_eor: int,
        dv_eor: float,
        beta_fg: Optional[Union[float, Sequence[float], jnp.ndarray]] = (2.63, 2.82),
        n_freq: int = 0,
        f0_mhz: Optional[float] = None,
        df_mhz: Optional[float] = None,
        n_eta: int = 0,
        d_eta: float = 0.0,
        sky_mask_time_mode: str = "union",
        telescope_latlonalt: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        jd_center: float = 2461040.5,
        nside: Optional[int] = None,
        fov_eor: Optional[float] = None,
        fov_fg: Optional[float] = None,
        n_t: int = 1,
        d_t: Optional[float] = None,
        image_geometry: Optional[ImageGeometry] = None,
        fourier_geometry: Optional[FourierGeometry] = None,
        beam_type: str = 'uniform',
        peak_amp: float = 1.0,
        fwhm_deg: Optional[float] = None,
        diam: Optional[float] = None,
        cosfreq: Optional[float] = None,
        tanh_freq: Optional[float] = None,
        tanh_sl_red: Optional[float] = None,
        pol: str = 'xx',
        freq_interp_kind: str = 'cubic',
        uvw_array_m: Optional[jnp.ndarray] = None,
        array_save_directory: str = "./matrices/",
    ) -> None:
        """Prepare the physical inputs documented on BuildMatrices.

        Fourier factors, foreground spectra and beam gains are shared by the
        component writers. Noise, priors, assembly and likelihood preparation
        are deliberately separate from this physical-model configuration.
        """
        # Store foreground spectral-model settings.
        if beta_fg is None:
            self.beta_fg = [2.63, 2.82]
        elif np.isscalar(beta_fg):
            self.beta_fg = [float(beta_fg)]
        else:
            self.beta_fg = list(beta_fg)
        self.n_fg_components = 1 + len(self.beta_fg)
        self.sky_mask_time_mode = str(sky_mask_time_mode)

        # Store frequency/eta grid parameters.
        self.n_freq = n_freq
        self.f0_mhz = f0_mhz
        self.df_mhz = df_mhz    
        # Frequency array in Hz (from MHz).
        self.freq_hz = (f0_mhz + df_mhz * jnp.arange(n_freq, dtype=jnp.float64)) * 1e6 
        c_ms = float(c.to("m/s").value)

        # Store HEALPix image-domain parameters.
        self.nside = nside
        self.fov_eor = fov_eor
        self.fov_fg = fov_fg
        self.telescope_latlonalt = telescope_latlonalt
        self.jd_center = jd_center
        self.n_t = n_t
        self.d_t = d_t


        # Store beam model parameters.
        self.beam_type = beam_type
        self.peak_amp = peak_amp
        self.fwhm_deg = fwhm_deg
        self.diam = diam
        self.cosfreq = cosfreq
        self.tanh_freq = tanh_freq
        self.tanh_sl_red = tanh_sl_red
        self.pol = pol
        self.freq_interp_kind = freq_interp_kind

        # Instrument data: convert uvw from meters to inverse radians for each frequency.
        uvw_array_m = jnp.array(uvw_array_m)  # (n_t, n_bls, 3)
        self.nbls = int(uvw_array_m.shape[1])
        # Baselines are fixed; sky directions and beam weights carry time dependence.
        uvw_array_m = uvw_array_m[0].T  # (3, n_bls)
        # Convert meters -> inverse radians by dividing by wavelength per frequency.
        wavelengths = c_ms / self.freq_hz
        self.inst_uvw_irad = uvw_array_m[jnp.newaxis, :, :] / wavelengths[:, jnp.newaxis, jnp.newaxis]

        self.array_save_directory = Path(array_save_directory)
        self.array_save_directory.mkdir(parents=True, exist_ok=True)

        # Reuse setup geometry when available; direct callers may construct it here.
        self.image_geometry = image_geometry or build_image_geometry(
            fov_eor=self.fov_eor,
            fov_fg=self.fov_fg,
            nside=self.nside,
            sky_mask_time_mode=self.sky_mask_time_mode,
            telescope_latlonalt=self.telescope_latlonalt,
            jd_center=self.jd_center,
            n_t=self.n_t,
            d_t=self.d_t,
        )
        self.fourier_geometry = fourier_geometry or build_fourier_geometry(
            n_u=n_u_eor,
            n_v=n_v_eor,
            n_eta=n_eta,
            du=du_eor,
            dv=dv_eor,
            d_eta=d_eta,
        )
        # F' uses the reference-time tangent chart within the cosmological field.
        lm_21 = self.image_geometry.lmn_fg[
            self.image_geometry.time_ind_center,
            :2,
        ]
        lm_21 = lm_21[:, self.image_geometry.eor_fg_shared_axis]
        self.transverse_fourier_operator = build_transverse_fourier_operator(
            self.fourier_geometry.uv,
            lm_21,
            uv_cell_area=self.fourier_geometry.du * self.fourier_geometry.dv,
        )
        # F_z uses offsets from the central channel to define line-of-sight phases.
        frequency_offsets_hz = self.freq_hz - self.freq_hz[n_freq // 2]
        self.spectral_fourier_operator = build_spectral_fourier_operator(
            self.fourier_geometry.eta,
            frequency_offsets_hz,
            d_eta=self.fourier_geometry.d_eta,
        )
        # Q_z's compact spectral factors use MHz and signed power-law exponents.
        self.foreground_spectral_operator = build_foreground_spectral_operator(
            self.freq_hz / 1e6,
            self.f0_mhz,
            spectral_indices=tuple(-beta for beta in self.beta_fg),
        )
        # Beam model: evaluate beam response for FG pixels across time/frequency.
        self.beam_model = Beam_Model(
            self.beam_type, 
            self.peak_amp, 
            self.fwhm_deg, 
            self.diam, 
            self.cosfreq, 
            self.tanh_freq, 
            self.tanh_sl_red, 
            self.pol, 
            self.freq_interp_kind,
        )
        self.beam_response = self.beam_model.get_beam_vals(
            self.image_geometry.az_za_fg, self.freq_hz
            )  # Shape (n_freq, n_t, n_pix_fg)

        self.n_fg_spatial_modes = int(self.image_geometry.n_pix_fg)

    def T_21_build_save(self, *, matrix_name: str = "T_21") -> Path:
        r"""Save the cosmological operator using the configured observation and Fourier grid.

        Parameters
        ----------
        matrix_name : str, optional
            Output file stem within array_save_directory; defaults to T_21.

        Returns
        -------
        path : pathlib.Path
            Complete complex128 operator with (uv, eta) coefficient columns.

        Notes
        -----
        Select the cosmological sky pixels from the shared observation geometry,
        then delegate CPU placement, block partitioning and saving to the
        standalone T_21_build_save function. No prior weighting or KL is applied.
        """
        # The cosmological image support is a subset of the foreground selection.
        mask = np.asarray(self.image_geometry.eor_fg_shared_axis)
        lmn = np.asarray(self.image_geometry.lmn_fg)[:, :, mask]
        beam = np.asarray(self.beam_response).transpose(1, 0, 2)[:, :, mask]
        return T_21_build_save(
            lmn, self.inst_uvw_irad, beam, self.image_geometry.pixel_area_sr,
            self.transverse_fourier_operator, self.spectral_fourier_operator,
            output_path=self.array_save_directory / f"{matrix_name}.npy",
        )

    def T_fg_build_save(self, *, matrix_name: str = "T_fg") -> Path:
        r"""Save the physical foreground operator using the configured sky and spectra.

        Parameters
        ----------
        matrix_name : str, optional
            Output file stem within array_save_directory; defaults to T_fg.

        Returns
        -------
        path : pathlib.Path
            Complete complex128 operator with spectral-component/pixel columns.

        Notes
        -----
        The standalone T_fg_build_save function distributes CPU evaluation
        and saving. The original spectral amplitude columns are retained;
        compression and foreground-prior transformations are separate stages.
        """
        # Align beam axes with the numerical operator's time/frequency/pixel order.
        beam = np.asarray(self.beam_response).transpose(1, 0, 2)
        return T_fg_build_save(
            self.image_geometry.lmn_fg, self.inst_uvw_irad, beam,
            self.image_geometry.pixel_area_sr, self.foreground_spectral_operator,
            output_path=self.array_save_directory / f"{matrix_name}.npy",
        )

    def T_fg_weighted_build_save(
        self, *, prior_sqrt, noise_whitener, matrix_name: str = "A_fg",
    ) -> Path:
        r"""Save the prior- and noise-weighted foreground operator directly on CPU.

        Parameters
        ----------
        prior_sqrt : array-like
            Nonnegative prior standard deviations in spectral-component/pixel
            order, one per foreground coefficient.
        noise_whitener : array-like
            Positive inverse noise RMS in time/frequency/baseline order,
            one per visibility. Both weights must be finite real vectors.
        matrix_name : str, optional
            Output file stem within array_save_directory; defaults to A_fg.

        Returns
        -------
        path : pathlib.Path
            Complete complex128 A = N^-1/2 T_fg Sigma_q^1/2 operator for KL
            decomposition. No unweighted foreground file is required or saved.

        Notes
        -----
        Weights are explicit inputs, separate from physical model configuration.
        The shared CPU writer applies them within each time-frequency block;
        decomposition and truncation are not performed here.
        """
        beam = np.asarray(self.beam_response).transpose(1, 0, 2)
        return T_fg_weighted_build_save(
            self.image_geometry.lmn_fg, self.inst_uvw_irad, beam,
            self.image_geometry.pixel_area_sr, self.foreground_spectral_operator,
            prior_sqrt=prior_sqrt, noise_whitener=noise_whitener,
            output_path=self.array_save_directory / f"{matrix_name}.npy",
        )
