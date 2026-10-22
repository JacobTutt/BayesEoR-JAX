"""Prepare physical inputs and save independent cosmological/foreground operators.

This builder does not assemble a joint model or prepare likelihood products.
Those stages take explicit saved operator paths, after any foreground reduction.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import c

from ...configuration import ModelConfig
from ..cosmology.fourier_geometry import FourierGeometry
from ..observation.image_geometry import ImageGeometry
from ..instrument.beam import Beam_Model
from ..operators import (
    build_foreground_spectral_operator,
    build_spectral_fourier_operator,
    build_transverse_fourier_operator,
)
from .operator_construction import T_21_build_save, T_fg_build_save, T_fg_weighted_build_save

jax.config.update("jax_enable_x64", True)


class BuildMatrices:
    r"""Prepare physical responses from model settings and resolved coordinates.

    Parameters
    ----------
    config : ModelConfig
        Requested spectral, beam and storage settings. A RunConfig is also
        accepted, but data-selection, prior and sampler fields are not used.
    frequencies_hz : array-like
        Resolved observing frequencies in Hz, shape (n_frequency,).
    uvw_array_m : array-like
        Resolved physical baselines, shape (n_time, n_baseline, 3), in metres.
        The first integration supplies the fixed instrument-frame baselines.
    image_geometry : ImageGeometry
        Prepared HEALPix pixel selections, time-dependent directions and areas.
        This fixes the actual image support; it is not rebuilt from config.
    fourier_geometry : FourierGeometry or None, optional
        Prepared rectilinear (u, v, eta) grid for the cosmological signal.
        Required for models containing the 21-cm component, unused for FG-only.

    Notes
    -----
    Geometry is resolved before construction. This object prepares the beam,
    Fourier factors and foreground spectra once, then its component writers
    construct CPU blocks and save separate operators. Joint assembly, noise
    weighting, KL reduction and inference remain separate stages.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        frequencies_hz: jax.Array,
        uvw_array_m: jax.Array,
        image_geometry: ImageGeometry,
        fourier_geometry: FourierGeometry | None = None,
    ) -> None:
        """Prepare physical responses without changing the requested settings."""
        if not isinstance(config, ModelConfig):
            raise TypeError("BuildMatrices requires a ModelConfig or RunConfig")
        if config.model not in ("eor", "fg", "eor+fg"):
            raise ValueError("model must be 'eor', 'fg' or 'eor+fg'")
        if config.model != "fg" and fourier_geometry is None:
            raise ValueError("A cosmological model requires resolved Fourier geometry")
        self.config = config
        self.model = config.model
        self.image_geometry = image_geometry
        self.fourier_geometry = fourier_geometry if self.model != "fg" else None

        # Resolved arrays, not nominal YAML dimensions, determine operator shapes.
        frequencies = np.asarray(frequencies_hz)
        if (frequencies.ndim != 1 or frequencies.size == 0 or not np.isrealobj(frequencies)
                or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0)):
            raise ValueError("frequencies_hz must be a nonempty positive finite vector")
        uvw = np.asarray(uvw_array_m)
        if (uvw.ndim != 3 or uvw.shape[2] != 3 or uvw.shape[1] == 0 or uvw.shape[0] == 0
                or not np.isrealobj(uvw)
                or uvw.shape[0] != image_geometry.lmn_fg.shape[0]
                or not np.all(np.isfinite(uvw))):
            raise ValueError("uvw_array_m must have shape (n_time, n_baseline, 3) matching image geometry")
        self.freq_hz = jnp.asarray(frequencies, dtype=jnp.float64)
        self.n_freq = int(self.freq_hz.size)
        self.n_t, self.nbls = uvw.shape[:2]
        self.f0_mhz = float(self.freq_hz[0] / 1e6)

        # Foreground amplitudes retain their first-channel spectral normalization.
        if config.beta_fg is None:
            self.beta_fg = list(ModelConfig().beta_fg)
        elif np.isscalar(config.beta_fg):
            self.beta_fg = [float(config.beta_fg)]
        else:
            self.beta_fg = list(config.beta_fg)
        self.n_fg_components = 1 + len(self.beta_fg)
        self.n_fg_spatial_modes = int(image_geometry.n_pix_fg)

        # Baselines are fixed; time dependence lives in sky directions and beam weights.
        wavelengths = float(c.to("m/s").value) / self.freq_hz
        self.inst_uvw_irad = jnp.asarray(uvw[0].T)[None, :, :] / wavelengths[:, None, None]
        self.array_save_directory = Path(config.array_save_directory).expanduser().resolve()
        self.array_save_directory.mkdir(parents=True, exist_ok=True)

        if self.model != "fg":
            # F' maps transverse modes into the reference-time cosmological image.
            lm_21 = image_geometry.lmn_fg[image_geometry.time_ind_center, :2]
            lm_21 = lm_21[:, image_geometry.eor_fg_shared_axis]
            self.transverse_fourier_operator = build_transverse_fourier_operator(
                self.fourier_geometry.uv, lm_21,
                uv_cell_area=self.fourier_geometry.du * self.fourier_geometry.dv,
            )
            # F_z maps line-of-sight modes to the resolved observing channels.
            frequency_offsets_hz = self.freq_hz - self.freq_hz[self.n_freq // 2]
            self.spectral_fourier_operator = build_spectral_fourier_operator(
                self.fourier_geometry.eta, frequency_offsets_hz,
                d_eta=self.fourier_geometry.d_eta,
            )
        if self.model != "eor":
            self.foreground_spectral_operator = build_foreground_spectral_operator(
                self.freq_hz / 1e6, self.f0_mhz,
                spectral_indices=tuple(-beta for beta in self.beta_fg),
            )

        # Apply the same beam to both image supports; component selection happens later.
        self.beam_model = Beam_Model(
            config.beam_type, config.peak_amp, config.fwhm_deg, config.diam,
            config.cosfreq, config.tanh_freq, config.tanh_sl_red,
            config.pol, config.freq_interp_kind,
        )
        self.beam_response = self.beam_model.get_beam_vals(
            image_geometry.az_za_fg, self.freq_hz,
        )

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
        if self.model == "fg":
            raise ValueError("The foreground-only model has no cosmological operator")
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
        if self.model == "eor":
            raise ValueError("The EoR-only model has no foreground operator")
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
        if self.model == "eor":
            raise ValueError("The EoR-only model has no foreground operator")
        beam = np.asarray(self.beam_response).transpose(1, 0, 2)
        return T_fg_weighted_build_save(
            self.image_geometry.lmn_fg, self.inst_uvw_irad, beam,
            self.image_geometry.pixel_area_sr, self.foreground_spectral_operator,
            prior_sqrt=prior_sqrt, noise_whitener=noise_whitener,
            output_path=self.array_save_directory / f"{matrix_name}.npy",
        )
