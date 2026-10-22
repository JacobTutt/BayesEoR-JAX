"""Command-line and YAML configuration for data, models and inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonargparse import ActionConfigFile, ArgumentParser


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    """Requested spectral, beam and storage settings for physical operators.

    Parameters
    ----------
    model : {"eor", "fg", "eor+fg"}
        Components to construct, independently of the simulated data contents.
    beta_fg : sequence of float
        Positive power-law indices following the constant foreground component.
    beam_type, peak_amp, fwhm_deg, diam, cosfreq, tanh_freq, tanh_sl_red,
    pol, freq_interp_kind : optional
        Primary-beam settings; angles are in degrees, diameter in metres.
        See ``Beam_Model`` for individual analytic and file-backed models.
    array_save_directory : pathlib.Path or str
        Directory for saved component operators and likelihood products.

    Notes
    -----
    Actual frequencies, baselines and image/Fourier coordinates are supplied
    separately to the builder. Requested settings are not overwritten when
    the observation determines those resolved inputs.
    """
    model: str = 'eor+fg'
    beta_fg: tuple[float, ...] = (2.63, 2.82)
    beam_type: str = 'uniform'
    peak_amp: float = 1.0
    fwhm_deg: float | None = None
    diam: float | None = None
    cosfreq: float | None = None
    tanh_freq: float | None = None
    tanh_sl_red: float | None = None
    pol: str = 'xx'
    freq_interp_kind: str = 'cubic'
    array_save_directory: Path | str = './matrices/'


@dataclass(frozen=True, kw_only=True)
class RunConfig(ModelConfig):
    """Requested settings for data preparation, model construction and inference.

    Parameters
    ----------
    data_path, inst_model, noise_data_path : pathlib.Path or str, optional
        Visibility input and optional instrument/noise products.
    n_u_eor, n_v_eor, du_eor, dv_eor, n_eta : optional
        Requested Fourier-grid dimensions and spacings; None permits setup to
        infer them from the selected observation and image geometry.
    n_freq, f0_mhz, df_mhz, n_t, d_t, jd_center : optional
        Requested observation axes (MHz, seconds and Julian dates). Raw data
        determine the actual axes; preprocessed vectors require these settings.
    nside, fov_eor, fov_fg, telescope_latlonalt : optional
        HEALPix resolution, field diameters in degrees and telescope location
        (latitude/longitude in degrees, altitude in metres).
    priors, uniform_priors : optional
        Log10 power bounds and the per-bin linear-uniform prior mask.
    foreground_prior_std_mk, foreground_kl_path : optional
        Physical spectral-component widths, or a preconstructed KL operator
        whose coefficients instead have unit prior variance.
    output_dir, save_dir, clobber : optional
        Metadata/preprocessed output directories and overwrite permission.
    quiet, run, resume, n_live_scale_factor, random_key : optional
        Driver and MultiNest controls. These are not physical model inputs.

    Notes
    -----
    Data selection and noise controls retain the YAML/CLI names documented in
    ``docs/configuration.md``. Construct directly in Python or obtain from
    ``BayesEoRParser.parse_args``. Use ``dataclasses.replace`` for a new run
    configuration; setup leaves this object unchanged. Sequence inputs are
    read-only by convention, even when supplied as lists or NumPy arrays.
    """
    data_path: Path | str | None = None
    inst_model: Path | str | None = None
    noise_data_path: Path | str | None = None
    n_u_eor: int | None = None
    n_v_eor: int | None = None
    du_eor: float | None = None
    dv_eor: float | None = None
    n_freq: int | None = None
    f0_mhz: float | None = None
    df_mhz: float | None = None
    n_eta: int | None = None
    n_t: int = 1
    d_t: float | None = None
    jd_center: float | None = None
    nside: int = 128
    fov_eor: float = 10.0
    fov_fg: float | None = None
    telescope_latlonalt: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0)
    antenna_selection: str = 'cross'
    bl_cutoff: float | None = None
    freq_idx_min: int | None = None
    freq_min: float | None = None
    freq_center: float | None = None
    n_freqs_select: int | None = None
    jd_idx_min: int | None = None
    jd_min: float | None = None
    n_times_select: int | None = None
    form_pi: bool = True
    pi_norm: float = 1.0
    redundant_avg: bool = False
    uniform_redundancy: bool = False
    calc_noise: bool = False
    sigma: float | None = None
    noise_seed: int = 742123
    save_vis: bool = False
    save_model: bool = False
    save_dir: Path | str = './'
    output_dir: Path | str = './'
    clobber: bool = False
    effective_noise_path: Path | str | None = None
    priors: Any = None
    uniform_priors: Any = None
    foreground_prior_std_mk: tuple[float, ...] = (
        8690.961932554002, 2479547.297490031, 1406322.413511489,
    )
    foreground_kl_path: Path | str | None = None
    use_intrinsic_noise_fitting: bool = False
    quiet: bool = False
    run: bool = False
    resume: bool = False
    n_live_scale_factor: float = 25.0
    random_key: int = 0


class BayesEoRParser(ArgumentParser):
    """Read the existing YAML/CLI options into a typed run configuration.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``jsonargparse.ArgumentParser``.
    parse_as_dict : bool, optional
        Return a plain dictionary instead of RunConfig when explicitly enabled.
        The standard workflow uses the typed result directly in ``run_setup``.

    Notes
    -----
    YAML keys use underscores and CLI flags use dashes. The options retain
    their existing names and required fields. The CLI output directory keeps
    its historical ``./BayesEoROutputs/`` default; direct RunConfig construction
    retains the Python setup default ``./``.

    Examples
    --------
    >>> config = BayesEoRParser().parse_args([
    ...     "--config", "test_examples/eor/config.yaml",
    ... ])
    >>> isinstance(config, RunConfig)
    True
    """

    def __init__(self, *args, parse_as_dict: bool = False, **kwargs):
        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)

        defaults = RunConfig()

        # General
        self.add_argument(
            "--config",
            action=ActionConfigFile,
            help="Path to a YAML configuration file containing parser arguments.",
        )
        self.add_argument("--quiet", action="store_true", help="Quiet mode.")
        self.add_argument("--clobber", action="store_true", help="Overwrite existing outputs.")
        self.add_argument("--run", action="store_true", help="Run a full analysis.")

        # Output / sampler / prior controls
        self.add_argument(
            "--array-save-directory",
            type=str,
            default=defaults.array_save_directory,
            help="Directory for matrix storage.",
        )
        self.add_argument("--output-dir", type=str, default="./BayesEoROutputs/", help="Directory for sampler outputs.")
        self.add_argument(
            "--model", choices=("eor", "fg", "eor+fg"), default=defaults.model,
            help="Components included in inference; independent of the contents of the data.",
        )
        self.add_argument("--priors", type=list[list[float]], help="Power-spectrum prior ranges [[min, max], ...].")
        self.add_argument(
            "--uniform-priors",
            type=list[int],
            default=defaults.uniform_priors,
            help="Optional boolean/int mask over k bins (1=True uses linear-uniform sampling).",
        )
        self.add_argument(
            "--foreground-prior-std-mk",
            type=list[float],
            default=defaults.foreground_prior_std_mk,
            help="Physical prior width for each foreground spectral field in mK.",
        )
        self.add_argument("--use-intrinsic-noise-fitting", action="store_true", help="Fit a global noise level.")
        self.add_argument(
            "--foreground-kl-path", type=str, default=defaults.foreground_kl_path,
            help="Use a saved unwhitened KL foreground operator with a standard-normal prior.",
        )
        self.add_argument(
            "--resume",
            action="store_true",
            help="Resume an existing MultiNest run from the current output_dir.",
        )
        self.add_argument("--n-live-scale-factor", type=float, default=defaults.n_live_scale_factor, help="Live points = factor * n_dim.")
        self.add_argument("--random-key", type=int, default=defaults.random_key, help="PRNG seed for nested sampling.")

        # UV model grids
        self.add_argument(
            "--n-u-eor",
            type=int,
            help="Number of EoR uv pixels along u. Defaults to the auto-resolved requirement.",
        )
        self.add_argument(
            "--n-v-eor",
            type=int,
            help="Number of EoR uv pixels along v. Defaults to the auto-resolved requirement.",
        )
        self.add_argument(
            "--du-eor",
            type=float,
            default=defaults.du_eor,
            help="EoR u spacing in inverse radians. Defaults to 1 / fov_l_angle_eor from the fixed local (l, m, n) chart.",
        )
        self.add_argument(
            "--dv-eor",
            type=float,
            default=defaults.dv_eor,
            help="EoR v spacing in inverse radians. Defaults to 1 / fov_m_angle_eor from the fixed local (l, m, n) chart.",
        )
        self.add_argument("--beta-fg", type=list[float], default=defaults.beta_fg, help="FG power-law indices.")

        # Frequency / eta
        self.add_argument("--n-freq", type=int, required=True, help="Number of frequency channels.")
        self.add_argument("--f0-mhz", type=float, required=True, help="Minimum frequency in MHz.")
        self.add_argument("--df-mhz", type=float, required=True, help="Channel width in MHz.")
        self.add_argument("--n-eta", type=int, help="Number of line-of-sight Fourier modes. Defaults to n_freq.")

        # Time / sky geometry
        self.add_argument("--n-t", type=int, default=defaults.n_t, help="Number of time integrations.")
        self.add_argument("--d-t", type=float, help="Integration time in seconds.")
        self.add_argument("--jd-center", type=float, default=defaults.jd_center, help="Central Julian date for centered time-window selection (raw mode) or centered time-axis anchor (.npy mode).")
        self.add_argument("--nside", type=int, required=True, help="HEALPix nside.")
        self.add_argument("--fov-eor", type=float, default=defaults.fov_eor, help="Circular EoR FoV in degrees; unused for FG-only inference.")
        self.add_argument("--fov-fg", type=float, help="Circular FG FoV in degrees. Defaults to fov_eor.")
        self.add_argument("--telescope-latlonalt", type=list[float], required=True, help="Telescope location [lat_deg, lon_deg, alt_m].")

        # Beam / instrument
        self.add_argument("--beam-type", type=str, default=defaults.beam_type, help="Beam model name or UVBeam file path.")
        self.add_argument("--peak-amp", type=float, default=defaults.peak_amp, help="Beam peak amplitude.")
        self.add_argument("--fwhm-deg", type=float, help="Beam FWHM in degrees.")
        self.add_argument("--diam", type=float, help="Antenna diameter in meters.")
        self.add_argument("--cosfreq", type=float, help="Cosine frequency for gausscosine beam.")
        self.add_argument("--tanh-freq", type=float, help="Rate parameter for tanhairy beam.")
        self.add_argument("--tanh-sl-red", type=float, help="Sidelobe reduction for tanhairy beam.")
        self.add_argument("--pol", type=str, default=defaults.pol, help="Polarization for UVBeam evaluation.")
        self.add_argument("--freq-interp-kind", type=str, default=defaults.freq_interp_kind, help="UVBeam frequency interpolation mode.")
        self.add_argument("--inst-model", type=str, help="Path to instrument-model directory.")

        # Data / noise
        self.add_argument("--data-path", type=str, help="Path to visibility data.")
        self.add_argument("--antenna-selection", type=str, default=defaults.antenna_selection, help="Antenna downselect string.")
        self.add_argument("--bl-cutoff", type=float, help="Baseline length cutoff in meters.")
        self.add_argument("--freq-idx-min", type=int, help="Minimum frequency index for downselect.")
        self.add_argument("--freq-min", type=float, help="Minimum frequency in Hz for downselect.")
        self.add_argument("--freq-center", type=float, help="Central frequency in Hz for downselect.")
        self.add_argument("--n-freqs-select", type=int, help="Number of frequencies to keep.")
        self.add_argument("--jd-idx-min", type=int, help="Minimum time index for downselect.")
        self.add_argument("--jd-min", type=float, help="Minimum Julian date for downselect (raw mode) or time-axis anchor (.npy mode).")
        self.add_argument("--n-times-select", type=int, help="Number of times to keep.")
        self.add_argument("--form-pi", action="store_true", dest="form_pi", default=defaults.form_pi, help="Form pseudo-Stokes I visibilities.")
        self.add_argument("--no-form-pi", action="store_false", dest="form_pi", help=argparse.SUPPRESS)
        self.add_argument("--pi-norm", type=float, default=defaults.pi_norm, help="Normalization for pseudo-Stokes I formation.")
        self.add_argument("--redundant-avg", action="store_true", help="Enable redundant averaging.")
        self.add_argument("--uniform-redundancy", action="store_true", help="Force uniform redundancy model.")
        self.add_argument("--calc-noise", action="store_true", help="Estimate noise via even/odd differencing.")
        self.add_argument("--save-vis", action="store_true", help="Save visibility vector to disk.")
        self.add_argument("--save-model", action="store_true", help="Save instrument model arrays to disk.")
        self.add_argument("--save-dir", type=str, default=defaults.save_dir, help="Output directory for saved vis/model arrays.")
        self.add_argument("--noise-data-path", type=str, help="Path to preprocessed noise vector.")
        self.add_argument("--sigma", type=float, help="Visibility-noise standard deviation.")
        self.add_argument("--noise-seed", type=int, default=defaults.noise_seed, help="Noise RNG seed.")
        self.add_argument("--effective-noise-path", type=str, help="Path to per-visibility effective-noise array.")

    def parse_args(self, args_str=None):
        """Read YAML/CLI settings and return a typed, requested RunConfig.

        Parameters
        ----------
        args_str : list of str or None
            CLI tokens, or None to read the process command line.

        Returns
        -------
        RunConfig or dict
            Typed settings by default; a plain mapping when the parser was
            explicitly created with ``parse_as_dict=True``.
        """
        args = super().parse_args(args_str)
        values = args if isinstance(args, dict) else args.as_dict()
        values = dict(values)
        values.pop("config", None)  # YAML source paths are parser bookkeeping.
        if values["beam_type"]:
            values["beam_type"] = values["beam_type"].lower()
        if values["pol"]:
            values["pol"] = values["pol"].lower()
        config = RunConfig(**values)
        return asdict(config) if isinstance(args, dict) else config
