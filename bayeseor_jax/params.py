"""Analysis settings for the BayesEoR-JAX pipeline."""

from __future__ import annotations

import argparse

from jsonargparse import ActionConfigFile, ArgumentParser


class BayesEoRParser(ArgumentParser):
    """
    Command line argument parser class for BayesEoR-JAX analysis parameters.

    For a minimal runnable example using this parser, see:

    .. code-block:: bash

        python examples/run_analysis_example.py

    Attributes
    ----------

    General Parameters

    config : str, optional
        Path to a YAML configuration file containing parser arguments.
    quiet : bool, optional
        Quiet mode where only a few statements are printed. Defaults to False.
    clobber : bool, optional
        Overwrite files if they exist. Defaults to False.
    run : bool, optional
        Run a full analysis. Defaults to False.

    Output / Sampler / Prior Parameters

    array_save_directory : str, optional
        Directory for matrix storage. Defaults to ``"./matrices/"``.
    output_dir : str, optional
        Directory for sampler outputs. Defaults to ``"./BlackJAXOutputs/"``.
    priors : list of list, optional
        Power-spectrum prior ranges ``[[min, max], ...]``.
    uniform_priors : list of bool/int, optional
        Boolean mask over k bins where True indicates a linear-uniform
        physical prior implemented through a Jacobian correction in the
        likelihood. Sampler coordinates remain in log10-power space for all
        bins.
    inverse_lw_power : float, optional
        Inverse prior power for FG/LSSM coefficients. Defaults to ``1e-16``.
    use_intrinsic_noise_fitting : bool, optional
        Fit for global intrinsic noise amplitude. Defaults to False.
    n_live_scale_factor : float, optional
        Multiplier converting ``n_dim`` to live-point count. Defaults to 25.
    num_inner_scale_factor : float, optional
        Multiplier converting ``n_dim`` to inner-step count. Defaults to 3.
    delete_scale_factor : float, optional
        Fraction of live points replaced per NS iteration. Defaults to 0.025.
    stop_threshold : float, optional
        Stop when ``logZ_live - logZ < stop_threshold``. Defaults to -3.
    random_key : int, optional
        Base PRNG seed for nested sampling. Defaults to 0.

    UV Model Parameters

    n_u_eor, n_v_eor : int
        Number of EoR uv pixels along u and v axes. ``n_v_eor`` defaults to
        ``n_u_eor``.
    du_eor, dv_eor : float, optional
        EoR uv-axis spacing in inverse radians. If omitted, each axis defaults
        to ``1 / deg2rad(fov_eor)``.
    n_u_fg, n_v_fg : int, optional
        Number of FG uv pixels along u and v axes. ``n_u_fg`` defaults to
        ``n_u_eor`` and ``n_v_fg`` defaults to ``n_u_fg``.
    du_fg, dv_fg : float, optional
        FG uv-axis spacing in inverse radians. If omitted, each axis defaults
        to ``1 / deg2rad(fov_fg)``.
    fit_for_monopole : bool, optional
        Include the uv monopole mode. Defaults to False.
    n_modes_fg : int, optional
        Number of foreground/LSSM modes. Defaults to 0.
    n_powerlaws_fg : int, optional
        Number of foreground power-law modes. If None, derived from
        ``beta_fg`` when foreground modes are enabled.
    beta_fg : list of float, optional
        Foreground power-law spectral indices. Defaults to ``[2.63, 2.82]``.
    fg_non_monopole_parameterization : {"uv", "image"}, optional
        Spatial basis used for the non-monopole foreground ``[eta0 | LSSM]``
        block. Defaults to ``"uv"``.

    Frequency / LoS Parameters

    n_freq : int
        Number of frequency channels.
    f0_mhz : float
        Minimum frequency in MHz.
    df_mhz : float
        Frequency channel width in MHz.
    n_eta : int, optional
        Number of line-of-sight Fourier modes. Defaults to ``n_freq``.

    Time / Sky Geometry Parameters

    n_t : int, optional
        Number of time integrations. Defaults to 1.
    d_t : float, optional
        Integration time in seconds. Defaults to None.
    jd_center : float, optional
        Central Julian date used for centered time-window selection in raw
        pyuvdata mode. In preprocessed ``.npy`` mode, use this or ``jd_min``
        to define the time axis. Defaults to ``None``.
    nside : int
        HEALPix ``nside`` resolution parameter.
    fov_eor : float
        Circular EoR field of view in degrees.
    fov_fg : float, optional
        Circular FG field of view in degrees. Defaults to ``fov_eor``.
    telescope_latlonalt : list of float
        Telescope location ``[latitude_deg, longitude_deg, altitude_m]``.

    Beam / Instrument Parameters

    beam_type : str, optional
        Beam model name or path to a UVBeam file.
    peak_amp : float, optional
        Peak beam amplitude. Defaults to 1.0.
    fwhm_deg : float, optional
        Beam full-width at half-maximum in degrees.
    diam : float, optional
        Antenna diameter in meters.
    cosfreq : float, optional
        Cosine frequency for gausscosine beam mode.
    tanh_freq : float, optional
        Rate parameter for tanhairy beam mode.
    tanh_sl_red : float, optional
        Sidelobe reduction for tanhairy beam mode.
    pol : str, optional
        Polarization for UVBeam evaluation. Defaults to ``"xx"``.
    freq_interp_kind : str, optional
        Frequency interpolation mode for UVBeam evaluation.
    inst_model : str, optional
        Path to instrument-model directory.

    Data / Noise Parameters

    data_path : str, optional
        Path to visibility data.
    antenna_selection : str, optional
        Antenna downselect string used by pyuvdata selection.
    bl_cutoff : float, optional
        Baseline length cutoff in meters.
    freq_idx_min, freq_min, freq_center : optional
        Frequency downselect controls for raw pyuvdata ingestion.
    n_freqs_select : int, optional
        Number of frequency channels to keep after downselect.
    jd_idx_min, jd_min, jd_center : optional
        Time controls. In raw pyuvdata mode they define downselect behavior;
        in preprocessed ``.npy`` mode use ``jd_min`` or ``jd_center`` to
        anchor the reconstructed time axis.
    n_times_select : int, optional
        Number of time integrations to keep after downselect.
    form_pi : bool, optional
        Form pseudo-Stokes I visibilities. Defaults to True.
    pi_norm : float, optional
        Normalization used for pseudo-Stokes I formation.
    redundant_avg : bool, optional
        Enable redundant averaging in preprocessing.
    uniform_redundancy : bool, optional
        Force uniform redundancy model when averaging.
    calc_noise : bool, optional
        Estimate empirical noise via even/odd differencing.
    save_vis, save_model : bool, optional
        Write preprocessed vis/model products to disk.
    save_dir : str, optional
        Output directory for saved vis/model products.
    noise_data_path : str, optional
        Path to preprocessed noise vector.
    sigma : float, optional
        Visibility-noise standard deviation.
    noise_seed : int, optional
        RNG seed for noise generation. Defaults to 742123.
    effective_noise_path : str, optional
        Path to per-visibility effective-noise array.
    """

    def __init__(self, *args, parse_as_dict: bool = False, **kwargs):
        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)

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
            default="./matrices/",
            help="Directory for matrix storage.",
        )
        self.add_argument("--output-dir", type=str, default="./BlackJAXOutputs/", help="Directory for sampler outputs.")
        self.add_argument("--priors", type=list[list[float]], help="Power-spectrum prior ranges [[min, max], ...].")
        self.add_argument(
            "--uniform-priors",
            type=list[int],
            default=None,
            help="Optional boolean/int mask over k bins (1=True uses linear-uniform sampling).",
        )
        self.add_argument("--inverse-lw-power", type=float, default=1e-16, help="Inverse FG prior power.")
        self.add_argument("--use-intrinsic-noise-fitting", action="store_true", help="Fit a global noise level.")
        self.add_argument("--n-live-scale-factor", type=float, default=25.0, help="Live points = factor * n_dim.")
        self.add_argument("--num-inner-scale-factor", type=float, default=3.0, help="Inner steps = factor * n_dim.")
        self.add_argument("--delete-scale-factor", type=float, default=0.025, help="Fraction of live points deleted per NS step.")
        self.add_argument("--stop-threshold", type=float, default=-3.0, help="Stop when logZ_live - logZ is below this threshold.")
        self.add_argument("--random-key", type=int, default=0, help="PRNG seed for nested sampling.")

        # UV model grids
        self.add_argument("--n-u-eor", type=int, required=True, help="Number of EoR uv pixels along u.")
        self.add_argument("--n-v-eor", type=int, help="Number of EoR uv pixels along v. Defaults to n_u_eor.")
        self.add_argument("--du-eor", type=float, default=None, help="EoR u spacing in inverse radians. Defaults to 1/deg2rad(fov_eor).")
        self.add_argument("--dv-eor", type=float, default=None, help="EoR v spacing in inverse radians. Defaults to 1/deg2rad(fov_eor).")
        self.add_argument("--n-u-fg", type=int, help="Number of FG uv pixels along u. Defaults to n_u_eor.")
        self.add_argument("--n-v-fg", type=int, help="Number of FG uv pixels along v. Defaults to n_u_fg.")
        self.add_argument("--du-fg", type=float, default=None, help="FG u spacing in inverse radians. Defaults to 1/deg2rad(fov_fg).")
        self.add_argument("--dv-fg", type=float, default=None, help="FG v spacing in inverse radians. Defaults to 1/deg2rad(fov_fg).")
        self.add_argument("--fit-for-monopole", action="store_true", help="Include the uv monopole mode.")
        self.add_argument("--n-modes-fg", type=int, default=0, help="Number of FG/LSSM modes.")
        self.add_argument("--n-powerlaws-fg", type=int, help="Number of FG power-law modes.")
        self.add_argument("--beta-fg", type=list[float], default=[2.63, 2.82], help="FG power-law indices.")
        self.add_argument(
            "--fg-non-monopole-parameterization",
            type=str,
            choices=("uv", "image"),
            default="uv",
            help="Spatial basis for the non-monopole FG [eta0 | LSSM] block.",
        )

        # Frequency / eta
        self.add_argument("--n-freq", type=int, required=True, help="Number of frequency channels.")
        self.add_argument("--f0-mhz", type=float, required=True, help="Minimum frequency in MHz.")
        self.add_argument("--df-mhz", type=float, required=True, help="Channel width in MHz.")
        self.add_argument("--n-eta", type=int, help="Number of line-of-sight Fourier modes. Defaults to n_freq.")

        # Time / sky geometry
        self.add_argument("--n-t", type=int, default=1, help="Number of time integrations.")
        self.add_argument("--d-t", type=float, help="Integration time in seconds.")
        self.add_argument("--jd-center", type=float, default=None, help="Central Julian date for centered time-window selection (raw mode) or centered time-axis anchor (.npy mode).")
        self.add_argument("--nside", type=int, required=True, help="HEALPix nside.")
        self.add_argument("--fov-eor", type=float, required=True, help="Circular EoR FoV in degrees.")
        self.add_argument("--fov-fg", type=float, help="Circular FG FoV in degrees. Defaults to fov_eor.")
        self.add_argument("--telescope-latlonalt", type=list[float], required=True, help="Telescope location [lat_deg, lon_deg, alt_m].")

        # Beam / instrument
        self.add_argument("--beam-type", type=str, default="uniform", help="Beam model name or UVBeam file path.")
        self.add_argument("--peak-amp", type=float, default=1.0, help="Beam peak amplitude.")
        self.add_argument("--fwhm-deg", type=float, help="Beam FWHM in degrees.")
        self.add_argument("--diam", type=float, help="Antenna diameter in meters.")
        self.add_argument("--cosfreq", type=float, help="Cosine frequency for gausscosine beam.")
        self.add_argument("--tanh-freq", type=float, help="Rate parameter for tanhairy beam.")
        self.add_argument("--tanh-sl-red", type=float, help="Sidelobe reduction for tanhairy beam.")
        self.add_argument("--pol", type=str, default="xx", help="Polarization for UVBeam evaluation.")
        self.add_argument("--freq-interp-kind", type=str, default="cubic", help="UVBeam frequency interpolation mode.")
        self.add_argument("--inst-model", type=str, help="Path to instrument-model directory.")

        # Data / noise
        self.add_argument("--data-path", type=str, help="Path to visibility data.")
        self.add_argument("--antenna-selection", type=str, default="cross", help="Antenna downselect string.")
        self.add_argument("--bl-cutoff", type=float, help="Baseline length cutoff in meters.")
        self.add_argument("--freq-idx-min", type=int, help="Minimum frequency index for downselect.")
        self.add_argument("--freq-min", type=float, help="Minimum frequency in Hz for downselect.")
        self.add_argument("--freq-center", type=float, help="Central frequency in Hz for downselect.")
        self.add_argument("--n-freqs-select", type=int, help="Number of frequencies to keep.")
        self.add_argument("--jd-idx-min", type=int, help="Minimum time index for downselect.")
        self.add_argument("--jd-min", type=float, help="Minimum Julian date for downselect (raw mode) or time-axis anchor (.npy mode).")
        self.add_argument("--n-times-select", type=int, help="Number of times to keep.")
        self.add_argument("--form-pi", action="store_true", dest="form_pi", default=True, help="Form pseudo-Stokes I visibilities.")
        self.add_argument("--no-form-pi", action="store_false", dest="form_pi", help=argparse.SUPPRESS)
        self.add_argument("--pi-norm", type=float, default=1.0, help="Normalization for pseudo-Stokes I formation.")
        self.add_argument("--redundant-avg", action="store_true", help="Enable redundant averaging.")
        self.add_argument("--uniform-redundancy", action="store_true", help="Force uniform redundancy model.")
        self.add_argument("--calc-noise", action="store_true", help="Estimate noise via even/odd differencing.")
        self.add_argument("--save-vis", action="store_true", help="Save visibility vector to disk.")
        self.add_argument("--save-model", action="store_true", help="Save instrument model arrays to disk.")
        self.add_argument("--save-dir", type=str, default="./", help="Output directory for saved vis/model arrays.")
        self.add_argument("--noise-data-path", type=str, help="Path to preprocessed noise vector.")
        self.add_argument("--sigma", type=float, help="Visibility-noise standard deviation.")
        self.add_argument("--noise-seed", type=int, default=742123, help="Noise RNG seed.")
        self.add_argument("--effective-noise-path", type=str, help="Path to per-visibility effective-noise array.")

    def parse_args(self, args_str=None):
        """Parse arguments and apply lightweight normalization."""
        args = super().parse_args(args_str)

        if args.beam_type:
            args.beam_type = args.beam_type.lower()
        if args.pol:
            args.pol = args.pol.lower()

        return args
