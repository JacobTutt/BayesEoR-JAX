"""High-level setup orchestration for BayesEoR-JAX analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import jax
import jax.numpy as jnp
# from jax.experimental import sparse
import numpy as np
from jax.experimental import multihost_utils

from .build_matricies import BuildMatrices, LMN_ModelSpace
from .inference.likelihood_marg import MarginalisedLikelihood
from .inference.phi import DEFAULT_FOREGROUND_PRIOR_STD_MK, PhiInverse
from .inference.prior import Prior
from .model.cosmology_transforms import Cosmology
from .model.k_cube import (
    calc_mean_binned_k_vals,
    compress_to_nonempty_k_bins,
    generate_k_cube_in_physical_coordinates,
    generate_k_cube_model_spherical_binning,
    mask_k_cube,
)
from .utils import (
    _resolve_uv_grid_from_extents,
    load_numpy_dict,
    write_analysis_metadata,
)
from .vis import get_vis_data

jax.config.update("jax_enable_x64", True)


class SetupResult:
    """Container for setup outputs consumed by downstream samplers."""

    def __init__(
        self,
        likelihood_class: Likelihood,
        prior_class: Prior,
        phi_inverse_class: PhiInverse,
        bm: BuildMatrices,
        vis_dict: dict[str, Any],
        k_vals: jax.Array,
        bin_id_per_k_mode: jax.Array,
        valid_mask: jax.Array,
        ps_box_size_ra_mpc: float,
        ps_box_size_dec_mpc: float,
        ps_box_size_para_mpc: float,
    ) -> None:
        self.likelihood_class = likelihood_class
        self.prior_class = prior_class
        self.phi_inverse_class = phi_inverse_class
        self.bm = bm
        self.vis_dict = vis_dict
        self.k_vals = k_vals
        self.bin_id_per_k_mode = bin_id_per_k_mode
        self.valid_mask = valid_mask
        self.ps_box_size_ra_mpc = ps_box_size_ra_mpc
        self.ps_box_size_dec_mpc = ps_box_size_dec_mpc
        self.ps_box_size_para_mpc = ps_box_size_para_mpc
def run_setup(
    *,
    # IO
    data_path: Path | str,
    inst_model: Path | str | None = None,
    noise_data_path: Path | str | None = None,
    array_save_directory: Path | str = "./matrices/",
    # UV model
    n_u_eor: int | None = None,
    n_v_eor: int | None = None,
    du_eor: float | None = None,
    dv_eor: float | None = None,
    n_u_fg: int | None = None,
    n_v_fg: int | None = None,
    du_fg: float | None = None,
    dv_fg: float | None = None,
    fit_for_monopole: bool = False,
    image_domain_trim: bool = False,
    sky_mask_time_mode: str = "union",
    n_modes_fg: int = 0,
    n_powerlaws_fg: int | None = None,
    beta_fg: list[float] = [2.63, 2.82],
    fg_model_domain: str = "image",
    # Frequency / time
    n_freq: int | None = None,
    f0_mhz: float | None = None,
    df_mhz: float | None = None,
    n_eta: int | None = None,
    n_t: int = 1,
    d_t: float | None = None,
    jd_center: float | None = None,
    # Sky / beam
    nside: int = 128,
    fov_eor: float = 10.0,
    fov_fg: float | None = None,
    fg_core_fov: float | None = None,
    fg_outer_nside: int | None = None,
    telescope_latlonalt: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    beam_type: str = "uniform",
    peak_amp: float = 1.0,
    fwhm_deg: float | None = None,
    diam: float | None = None,
    cosfreq: float | None = None,
    tanh_freq: float | None = None,
    tanh_sl_red: float | None = None,
    pol: str = "xx",
    freq_interp_kind: str = "cubic",
    # Data preprocessing
    antenna_selection: str = "cross",
    bl_cutoff: float | None = None,
    freq_idx_min: int | None = None,
    freq_min: float | None = None,
    freq_center: float | None = None,
    n_freqs_select: int | None = None,
    jd_idx_min: int | None = None,
    jd_min: float | None = None,
    n_times_select: int | None = None,
    form_pi: bool = True,
    pi_norm: float = 1.0,
    redundant_avg: bool = False,
    uniform_redundancy: bool = False,
    calc_noise: bool = False,
    sigma: float | None = None,
    noise_seed: int = 742123,
    save_vis: bool = False,
    save_model: bool = False,
    save_dir: Path | str = "./",
    output_dir: Path | str = "./",
    clobber: bool = False,
    effective_noise_path: Path | str | None = None,
    # Prior / likelihood
    priors: Any = None,
    uniform_priors: Any = None,
    foreground_prior_std_mk: tuple[float, ...] = DEFAULT_FOREGROUND_PRIOR_STD_MK,
    use_intrinsic_noise_fitting: bool = False,
    **_: Any,
) -> SetupResult:
    """
    Build all objects required to run posterior sampling in BayesEoR-JAX.

    This is the main setup entry point used by :mod:`bayeseor_jax.run_ns`. It:

    1. Loads or preprocesses visibilities/noise metadata.
    2. Builds matrix products (`T`, `Ninv`, `T_Ninv_T`) needed by the
       marginalized Gaussian likelihood.
    3. Builds k-bin metadata and power-spectrum normalization inputs.
    4. Instantiates sampler-space prior, latent-space precision config, and
       likelihood config objects.

    Parameters
    ----------
    data_path : pathlib.Path or str
        Input visibility path. Supports raw pyuvdata files (``.uvh5``,
        ``.uvfits``, ``.ms``) or preprocessed vectors (``.npy``).
    inst_model : pathlib.Path or str, optional
        Instrument model directory for ``.npy`` input mode.
    noise_data_path : pathlib.Path or str, optional
        Optional precomputed noise vector path.
    array_save_directory : pathlib.Path or str, optional
        Directory used to store/load matrix products.
    output_dir : pathlib.Path or str, optional
        Directory where analysis-side metadata files are written. The new
        pipeline saves old-style `k-vals*.txt` products here for downstream
        analysis compatibility.

    n_u_eor, n_v_eor, du_eor, dv_eor : int/float, optional
        EoR model UV grid shape and spacing. When omitted, these are
        auto-resolved from the configured sky-mask FoV widths and instrument support.
    n_u_fg, n_v_fg, du_fg, dv_fg : int/float, optional
        FG model UV grid shape and spacing. When omitted, these are
        auto-resolved from the configured sky-mask FoV widths and instrument support.
    fit_for_monopole : bool, optional
        Whether to include a dedicated FG monopole block.
    sky_mask_time_mode : {"union", "center"}, optional
        How the fixed EoR/FG HEALPix masks are selected across time.
        ``"union"`` uses the drift-scan union over all configured timesteps.
        ``"center"`` uses only the central timestep FoV when defining the
        fixed sky mask.
    n_modes_fg, n_powerlaws_fg, beta_fg : optional
        FG large spectral scale model controls.

    n_freq, f0_mhz, df_mhz : optional
        Frequency-axis setup (overridden by data-driven values in raw mode).
    n_eta : int, optional
        Number of line-of-sight Fourier modes. Defaults to ``n_freq``.
    n_t, d_t, jd_min, jd_center : optional
        Time-axis controls.

    nside, fov_eor, fov_fg, telescope_latlonalt : optional
        Sky geometry controls.
    beam_type, peak_amp, fwhm_deg, diam, cosfreq, tanh_freq, tanh_sl_red, pol,
    freq_interp_kind : optional
        Beam model controls.

    antenna_selection, bl_cutoff, freq_idx_min, freq_min, freq_center,
    n_freqs_select, jd_idx_min, jd_min, n_times_select, form_pi, pi_norm,
    redundant_avg, uniform_redundancy, calc_noise, sigma, noise_seed, save_vis,
    save_model, save_dir, clobber, effective_noise_path : optional
        Data selection/preprocessing and noise-model controls.

    priors : array-like
        Power-spectrum prior bounds with shape ``(n_power_params, 2)`` in
        log10 space.
    uniform_priors : array-like of bool, optional
        Boolean mask of length ``n_power_params`` where True bins use the
        old BayesEoR linear-uniform physical prior implemented via a Jacobian
        correction inside the likelihood. Sampler coordinates remain in
        log10-power space for all bins.
    inverse_lw_power : float, optional
        Inverse prior power assigned to FG/LSSM latent coefficients.
    use_intrinsic_noise_fitting : bool, optional
        Whether to include a global intrinsic noise amplitude parameter in the
        likelihood.
    **_ : Any
        Catch-all for extra parser keys that are not consumed here.

    Returns
    -------
    SetupResult
        Container with:

        - `likelihood_class`: configured marginalized Gaussian likelihood.
        - `prior_class`: sampler-space prior object.
        - `phi_inverse_class`: latent prior-precision object.
        - `bm`: matrix builder instance.
        - `vis_dict`: visibility/noise/instrument metadata.
        - `k_vals`, `bin_id_per_k_mode`, `valid_mask`: k-binning metadata.
        - `ps_box_size_ra_mpc`, `ps_box_size_dec_mpc`, `ps_box_size_para_mpc`:
          cosmological box extents used for power normalization.
    """
    # Resolve parser-like defaults that are independent of the actual selected
    # data product. The uv-grid shapes themselves are resolved later from the
    # time-union FoV extents and instrument uv coverage once vis metadata are
    # available.
    if fov_fg is None:
        fov_fg = fov_eor
    if len(telescope_latlonalt) != 3:
        raise ValueError("telescope_latlonalt must contain exactly 3 values: [lat, lon, alt].")
    fov_eor = float(fov_eor)
    fov_fg = float(fov_fg)

    if n_eta is None:
        if n_freq is None:
            raise ValueError("n_eta is None and n_freq is None; one must be provided")
        n_eta = n_freq
    if n_powerlaws_fg is None:
        n_powerlaws_fg = len(beta_fg) if (beta_fg is not None and n_modes_fg > 0) else 0
    if n_modes_fg > n_powerlaws_fg and n_powerlaws_fg > 0:
        n_modes_fg = n_powerlaws_fg
    if len(foreground_prior_std_mk) != 1 + len(beta_fg):
        raise ValueError(
            "foreground_prior_std_mk must contain one width for the constant "
            "field and one for each beta_fg power law"
        )

    output_dir = Path(output_dir).expanduser().resolve()
    array_save_directory = Path(array_save_directory).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve()

    vis_dict = get_vis_data(
        data_path=data_path,
        inst_model=inst_model,
        noise_data_path=noise_data_path,
        sigma=sigma,
        noise_seed=noise_seed,
        calc_noise=calc_noise,
        antenna_selection=antenna_selection,
        bl_cutoff=bl_cutoff,
        freq_idx_min=freq_idx_min,
        freq_min=freq_min,
        freq_center=freq_center,
        n_freqs_select=n_freqs_select,
        jd_idx_min=jd_idx_min,
        jd_min=jd_min,
        jd_center=jd_center,
        n_times_select=n_times_select,
        form_pi=form_pi,
        pi_norm=pi_norm,
        pol=pol,
        redundant_avg=redundant_avg,
        uniform_redundancy=uniform_redundancy,
        n_freq=n_freq,
        f0_mhz=f0_mhz,
        df_mhz=df_mhz,
        n_t=n_t,
        d_t=d_t,
        save_vis=save_vis,
        save_model=save_model,
        save_dir=save_dir,
        clobber=clobber,
    )

    # Data-driven values overwrite nominal values in raw-file mode.
    # This ensures matrix dimensions and cosmology transforms follow the actual
    # selected data product.
    freqs_hz = vis_dict["freqs_hz"]
    n_freq = int(freqs_hz.size)
    f0_mhz = float(freqs_hz[0] / 1e6)
    df_mhz = float(vis_dict["df_hz"] / 1e6)
    n_t = int(vis_dict["uvws"].shape[0])
    if d_t is None:
        d_t = float(vis_dict["dt"])
    if vis_dict["jds"].size > 0:
        jd_center = float(vis_dict["jds"][n_t // 2])

    if n_eta is None:
        n_eta = n_freq

    # Build the configured HEALPix FoV model once before the expensive matrix
    # construction so the spherical-aware local angular widths can define the
    # uv sampling and automatic n_u/n_v sizing.
    lmn_model_space = LMN_ModelSpace(
        fov_eor=float(fov_eor),
        fov_fg=float(fov_fg),
        nside=int(nside),
        sky_mask_time_mode=str(sky_mask_time_mode),
        telescope_latlonalt=tuple(float(x) for x in telescope_latlonalt),
        jd_center=float(jd_center),
        n_t=int(n_t),
        d_t=float(d_t),
    )
    resolved_grid = _resolve_uv_grid_from_extents(
        extent_u_eor=float(lmn_model_space.fov_l_angle_eor),
        extent_v_eor=float(lmn_model_space.fov_m_angle_eor),
        extent_u_fg=float(lmn_model_space.fov_l_angle_fg),
        extent_v_fg=float(lmn_model_space.fov_m_angle_fg),
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        uvws_m=np.asarray(vis_dict["uvws"], dtype=float),
        beam_type=beam_type,
        fwhm_deg=None if fwhm_deg is None else float(fwhm_deg),
        diam=None if diam is None else float(diam),
        n_u_eor=n_u_eor,
        n_v_eor=n_v_eor,
        du_eor=du_eor,
        dv_eor=dv_eor,
        n_u_fg=n_u_fg,
        n_v_fg=n_v_fg,
        du_fg=du_fg,
        dv_fg=dv_fg,
    )
    n_u_eor = int(resolved_grid["n_u_eor"])
    n_v_eor = int(resolved_grid["n_v_eor"])
    n_u_fg = int(resolved_grid["n_u_fg"])
    n_v_fg = int(resolved_grid["n_v_fg"])
    du_eor = float(resolved_grid["du_eor"])
    dv_eor = float(resolved_grid["dv_eor"])
    du_fg = float(resolved_grid["du_fg"])
    dv_fg = float(resolved_grid["dv_fg"])

    setup_args_dict = {
        "data_path": None if data_path is None else str(Path(data_path).expanduser().resolve()),
        "inst_model": None if inst_model is None else str(Path(inst_model).expanduser().resolve()),
        "noise_data_path": None if noise_data_path is None else str(Path(noise_data_path).expanduser().resolve()),
        "array_save_directory": str(array_save_directory),
        "output_dir": str(output_dir),
        "n_u_eor": int(n_u_eor),
        "n_v_eor": int(n_v_eor),
        "du_eor": float(du_eor),
        "dv_eor": float(dv_eor),
        "n_u_fg": int(n_u_fg),
        "n_v_fg": int(n_v_fg),
        "du_fg": float(du_fg),
        "dv_fg": float(dv_fg),
        "fit_for_monopole": bool(fit_for_monopole),
        "image_domain_trim": bool(image_domain_trim),
        "sky_mask_time_mode": str(sky_mask_time_mode),
        "fg_core_fov": None if fg_core_fov is None else float(fg_core_fov),
        "fg_outer_nside": None if fg_outer_nside is None else int(fg_outer_nside),
        "n_modes_fg": int(n_modes_fg),
        "n_powerlaws_fg": int(n_powerlaws_fg),
        "beta_fg": None if beta_fg is None else list(beta_fg),
        "fg_model_domain": str(fg_model_domain),
        "n_freq": int(n_freq),
        "f0_mhz": float(f0_mhz),
        "df_mhz": float(df_mhz),
        "n_eta": int(n_eta),
        "n_t": int(n_t),
        "d_t": None if d_t is None else float(d_t),
        "jd_min": None if jd_min is None else float(jd_min),
        "jd_center": None if jd_center is None else float(jd_center),
        "nside": int(nside),
        "fov_eor": float(fov_eor),
        "fov_fg": float(fov_fg),
        "fov_l_angle_eor": float(lmn_model_space.fov_l_angle_eor),
        "fov_m_angle_eor": float(lmn_model_space.fov_m_angle_eor),
        "fov_l_angle_fg": float(lmn_model_space.fov_l_angle_fg),
        "fov_m_angle_fg": float(lmn_model_space.fov_m_angle_fg),
        "beam_aperture_width_uv": float(resolved_grid["beam_aperture_width_uv"]),
        "u_max_inst_model": float(resolved_grid["u_max_inst_model"]),
        "v_max_inst_model": float(resolved_grid["v_max_inst_model"]),
        "u_extent_with_buffer": float(resolved_grid["u_extent_with_buffer"]),
        "v_extent_with_buffer": float(resolved_grid["v_extent_with_buffer"]),
        "n_u_eor_required": int(resolved_grid["n_u_eor_required"]),
        "n_v_eor_required": int(resolved_grid["n_v_eor_required"]),
        "n_u_fg_required": int(resolved_grid["n_u_fg_required"]),
        "n_v_fg_required": int(resolved_grid["n_v_fg_required"]),
        "telescope_latlonalt": [float(x) for x in telescope_latlonalt],
        "beam_type": beam_type,
        "peak_amp": float(peak_amp),
        "fwhm_deg": None if fwhm_deg is None else float(fwhm_deg),
        "diam": None if diam is None else float(diam),
        "cosfreq": None if cosfreq is None else float(cosfreq),
        "tanh_freq": None if tanh_freq is None else float(tanh_freq),
        "tanh_sl_red": None if tanh_sl_red is None else float(tanh_sl_red),
        "pol": pol,
        "freq_interp_kind": freq_interp_kind,
        "antenna_selection": antenna_selection,
        "bl_cutoff": None if bl_cutoff is None else float(bl_cutoff),
        "freq_idx_min": None if freq_idx_min is None else int(freq_idx_min),
        "freq_min": None if freq_min is None else float(freq_min),
        "freq_center": None if freq_center is None else float(freq_center),
        "n_freqs_select": None if n_freqs_select is None else int(n_freqs_select),
        "jd_idx_min": None if jd_idx_min is None else int(jd_idx_min),
        "n_times_select": None if n_times_select is None else int(n_times_select),
        "form_pi": bool(form_pi),
        "pi_norm": float(pi_norm),
        "redundant_avg": bool(redundant_avg),
        "uniform_redundancy": bool(uniform_redundancy),
        "calc_noise": bool(calc_noise),
        "sigma": None if sigma is None else float(sigma),
        "noise_seed": int(noise_seed),
        "save_vis": bool(save_vis),
        "save_model": bool(save_model),
        "save_dir": str(save_dir),
        "clobber": bool(clobber),
        "effective_noise_path": None if effective_noise_path is None else str(Path(effective_noise_path).expanduser().resolve()),
        "priors": None if priors is None else np.asarray(priors, dtype=float).tolist(),
        "uniform_priors": None if uniform_priors is None else np.asarray(uniform_priors, dtype=bool).astype(int).tolist(),
        "foreground_prior_std_mk": list(foreground_prior_std_mk),
        "use_intrinsic_noise_fitting": bool(use_intrinsic_noise_fitting),
    }

    # Cosmology-derived physical cube dimensions.
    # These map the instrumental coordinate extents to comoving Mpc scales and
    # are required for k-bin definitions and power normalization.
    cosmo = Cosmology()
    z = float(cosmo.f2z(jnp.mean(freqs_hz)))
    bandwidth_hz = float(vis_dict["df_hz"]) * float(n_freq)
    ps_box_size_para_mpc = float(cosmo.dL_df(z) * bandwidth_hz)
    ps_box_size_ra_mpc = float(cosmo.dL_dth(z) * float(lmn_model_space.fov_l_angle_eor))
    ps_box_size_dec_mpc = float(cosmo.dL_dth(z) * float(lmn_model_space.fov_m_angle_eor))

    # Build the EoR k-cube model:
    # 1) generate physical k coordinates
    # 2) mask LOS-DC and uv-monopole modes
    # 3) assign spherical bins
    # 4) drop empty bins and remap voxel->bin IDs to compact contiguous labels
    mod_k, *_ = generate_k_cube_in_physical_coordinates(
        int(n_u_eor),
        int(n_v_eor),
        int(n_eta),
        ps_box_size_ra_mpc,
        ps_box_size_dec_mpc,
        ps_box_size_para_mpc,
    )
    mod_k_vo = mask_k_cube(mod_k)
    bid_full, valid_full, _, _, counts = generate_k_cube_model_spherical_binning(
        mod_k_vo=mod_k_vo,
        ps_box_size_para_mpc=ps_box_size_para_mpc,
    )
    k_vals_full = calc_mean_binned_k_vals(
        mod_k_vo=mod_k_vo,
        bin_id_per_k_mode=bid_full,
        valid_mask=valid_full,
        num_k_bins=int(counts.size),
    )
    k_vals, bid_compact, valid_compact = compress_to_nonempty_k_bins(
        k_vals_full=k_vals_full,
        counts=counts,
        bin_id_per_k_mode=bid_full,
        valid_mask=valid_full,
    )
    if jax.process_index() == 0:
        write_analysis_metadata(
            output_dir=output_dir,
            args=setup_args_dict,
            k_vals=k_vals,
            mod_k_vo=mod_k_vo,
            bin_id_per_k_mode=bid_compact,
            valid_mask=valid_compact,
            clobber=clobber,
        )
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("k-vals-metadata-saved")

    # Optional per-visibility effective-noise model.
    # If provided, this overrides sigma/redundancy-based covariance construction.
    effective_noise = None
    if effective_noise_path is not None:
        effective_noise = jnp.asarray(
            load_numpy_dict(effective_noise_path, as_jax=True),
            dtype=jnp.complex128,
        ).reshape(-1)
        expected_n_vis = int(vis_dict["vis_noisy"].size)
        if int(effective_noise.size) != expected_n_vis:
            raise ValueError(
                "effective_noise_path length mismatch: "
                f"got {int(effective_noise.size)} values, expected {expected_n_vis} "
                "(must match vis_noisy size = n_t * n_freq * n_bls)."
            )

    # Build minimum sufficient matrix stack for likelihood compression:
    # - T   : forward operator from latent coefficients to data space
    # - Ninv: inverse noise covariance
    # - T_Ninv_T: dense normal matrix used in marginalized Gaussian objective
    bm = BuildMatrices(
        n_u_eor=int(n_u_eor),
        du_eor=float(du_eor),
        n_v_eor=int(n_v_eor),
        dv_eor=float(dv_eor),
        beta_fg=beta_fg,
        n_freq=int(n_freq),
        f0_mhz=float(f0_mhz),
        df_mhz=float(df_mhz),
        n_eta=int(n_eta),
        d_eta=float(1.0 / bandwidth_hz),
        image_domain_trim=bool(image_domain_trim),
        sky_mask_time_mode=str(sky_mask_time_mode),
        telescope_latlonalt=tuple(float(x) for x in telescope_latlonalt),
        jd_center=float(jd_center),
        nside=int(nside),
        fov_eor=float(fov_eor),
        fov_fg=float(fov_fg),
        fg_core_fov=None if fg_core_fov is None else float(fg_core_fov),
        fg_outer_nside=None if fg_outer_nside is None else int(fg_outer_nside),
        n_t=int(n_t),
        d_t=float(d_t),
        beam_type=beam_type,
        peak_amp=float(peak_amp),
        fwhm_deg=fwhm_deg,
        diam=diam,
        cosfreq=cosfreq,
        tanh_freq=tanh_freq,
        tanh_sl_red=tanh_sl_red,
        pol=pol,
        freq_interp_kind=freq_interp_kind,
        uvw_array_m=vis_dict["uvws"],
        bl_red_array=vis_dict["redundancy"],
        sigma=sigma,
        effective_noise=effective_noise,
        array_save_directory=str(array_save_directory),
        clobber=bool(clobber),
    )

    # The likelihood stack extends the heavy matrix products with the
    # data-dependent compressed terms needed by the marginalized Gaussian path.
    vis_noisy = jnp.asarray(vis_dict["vis_noisy"], dtype=jnp.complex128)
    n_params = bm.build_minimum_likelihood_stack(vis_noisy)
    setup_args_dict["n_fg_pixels_trimmed"] = int(bm.n_fg_pixels_trimmed)
    setup_args_dict["n_pix_fg_final"] = int(bm.lmn_model_space.n_pix_fg)
    setup_args_dict["use_multires_fg"] = bool(bm.lmn_model_space.use_multires_fg)
    setup_args_dict["n_fg_groups"] = int(bm.lmn_model_space.n_fg_groups)
    setup_args_dict["n_fg_group_highres"] = int(np.sum(bm.lmn_model_space.fg_group_is_highres))
    setup_args_dict["n_fg_group_lowres"] = int(
        bm.lmn_model_space.n_fg_groups - np.sum(bm.lmn_model_space.fg_group_is_highres)
    )

    # Assemble sampler-space prior and latent-space precision configs.
    # `Prior` now works directly in log10(power) sampler coordinates.
    # `PhiInverse` maps linearized power amplitudes to latent precision diag.
    if priors is None:
        raise ValueError("priors must be provided")

    bounds_log = jnp.asarray(priors, dtype=jnp.float64)
    if bounds_log.ndim != 2 or bounds_log.shape[1] != 2:
        raise ValueError("priors must have shape (n_power_params, 2)")
    n_power_params = int(k_vals.shape[0])
    if bounds_log.shape[0] != n_power_params:
        raise ValueError(
            f"priors has {bounds_log.shape[0]} bins, but k binning produced {n_power_params} non-empty bins"
        )

    if uniform_priors is None:
        uniform_priors_mask = jnp.zeros((n_power_params,), dtype=bool)
    else:
        uniform_priors_mask = jnp.asarray(uniform_priors, dtype=bool)
        if uniform_priors_mask.shape != (n_power_params,):
            raise ValueError(
                "uniform_priors must have shape "
                f"({n_power_params},), got {uniform_priors_mask.shape}"
            )

    prior_class = Prior(
        bounds_log=bounds_log,
        power_param_start=0,
        n_power_params=n_power_params,
    )

    phi_inverse_class = PhiInverse(
        bin_id_per_k_mode=bid_compact,
        valid_mask=valid_compact,
        k_vals=k_vals,
        ps_box_size_ra_mpc=ps_box_size_ra_mpc,
        ps_box_size_dec_mpc=ps_box_size_dec_mpc,
        ps_box_size_para_mpc=ps_box_size_para_mpc,
        inst_to_cosmo_vol=float(cosmo.inst_to_cosmo_vol(z)),
        n_par=int(n_params),
        foreground_prior_std_mk=foreground_prior_std_mk,
        n_fg_spatial_modes=int(bm.n_fg_spatial_modes),
    )
    
    likelihood_class = MarginalisedLikelihood(
        matrix_dir=str(array_save_directory),
        n_data=int(vis_noisy.size),
        phi_inv_class=phi_inverse_class,
        prior_class=prior_class,
        uniform_priors=uniform_priors_mask,
        intrinsic_noise_fitting=bool(use_intrinsic_noise_fitting),
        power_param_start=0,
        n_power_params=n_power_params,
    )

    # Return one object containing both configs and diagnostics needed by
    # run_nestedsampling.
    return SetupResult(
        likelihood_class=likelihood_class,
        prior_class=prior_class,
        phi_inverse_class=phi_inverse_class,
        bm=bm,
        vis_dict=vis_dict,
        k_vals=k_vals,
        bin_id_per_k_mode=bid_compact,
        valid_mask=valid_compact,
        ps_box_size_ra_mpc=ps_box_size_ra_mpc,
        ps_box_size_dec_mpc=ps_box_size_dec_mpc,
        ps_box_size_para_mpc=ps_box_size_para_mpc,
    )
