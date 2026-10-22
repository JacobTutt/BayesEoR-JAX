"""High-level setup orchestration for BayesEoR-JAX analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from .model import (
    BuildMatrices,
    build_fourier_geometry,
    build_image_geometry,
)
from .inference.likelihood_marg import MarginalisedLikelihood
from .inference.phi import PhiInverse
from .configuration import RunConfig
from .inference.prior import Prior
from .model.instrument.noise import build_noise_variance
from .model.construction.operator_construction import assemble_joint_operator
from .model.construction.likelihood_products import save_likelihood_products
from .model.construction.storage import ensure_cached, save_array
from .model.cosmology.transforms import Cosmology
from .model.cosmology.power_spectrum import (
    calc_mean_binned_k_vals,
    compress_to_nonempty_k_bins,
    generate_k_modes_in_physical_coordinates,
    generate_k_cube_model_spherical_binning,
)
from .model.cosmology.fourier_geometry import _resolve_uv_grid_from_extents
from .io import (
    load_numpy_dict,
    write_analysis_metadata,
    write_log_files,
)
from .data.vis import get_vis_data

jax.config.update("jax_enable_x64", True)


class SetupResult:
    """Container for setup outputs consumed by downstream samplers."""

    def __init__(
        self,
        config: RunConfig,
        likelihood_class: MarginalisedLikelihood,
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
        model: str = "eor+fg",
    ) -> None:
        self.config = config
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
        self.model = model


def run_setup(config: RunConfig) -> SetupResult:
    """Prepare the physical operators, priors and marginalised likelihood.

    Parameters
    ----------
    config : RunConfig
        Requested data/model/inference settings. Actual observation axes and
        model coordinates are resolved locally without modifying this object.

    Returns
    -------
    SetupResult
        Configured likelihood, coefficient and power-spectrum priors, matrix
        builder, selected visibility data and cosmological bin metadata.

    Notes
    -----
    This coordinates loading, geometry, component construction, joint assembly
    and likelihood products; it does not run nested sampling. A supplied KL
    operator is loaded with a standard-normal prior, not recomputed.
    """
    if not isinstance(config, RunConfig):
        raise TypeError("run_setup requires a RunConfig")
    # Resolve derived values locally; retain the caller's requested settings.
    array_save_directory = config.array_save_directory
    n_u_eor = config.n_u_eor
    n_v_eor = config.n_v_eor
    du_eor = config.du_eor
    dv_eor = config.dv_eor
    n_freq = config.n_freq
    f0_mhz = config.f0_mhz
    df_mhz = config.df_mhz
    n_eta = config.n_eta
    n_t = config.n_t
    d_t = config.d_t
    jd_center = config.jd_center
    fov_eor = config.fov_eor
    fov_fg = config.fov_fg
    save_dir = config.save_dir
    output_dir = config.output_dir
    foreground_kl_path = config.foreground_kl_path

    if config.model not in ("eor", "fg", "eor+fg"):
        raise ValueError("model must be 'eor', 'fg' or 'eor+fg'")
    include_eor, include_fg = config.model != "fg", config.model != "eor"
    if not include_fg and foreground_kl_path is not None:
        raise ValueError("foreground_kl_path cannot be used with model='eor'")
    if not include_eor and config.use_intrinsic_noise_fitting:
        raise ValueError("Foreground-only inference currently requires a fixed noise model")
    if fov_fg is None:
        fov_fg = fov_eor
    # Single-component models need only their own sky support for beam evaluation.
    if not include_fg:
        fov_fg = fov_eor
    if not include_eor:
        fov_eor = fov_fg
    if len(config.telescope_latlonalt) != 3:
        raise ValueError("telescope_latlonalt must contain exactly 3 values: [lat, lon, alt].")
    fov_eor = float(fov_eor)
    fov_fg = float(fov_fg)

    if include_fg and not config.beta_fg:
        raise ValueError("beta_fg must contain at least one spectral index")
    if include_fg and foreground_kl_path is None and len(config.foreground_prior_std_mk) != 1 + len(config.beta_fg):
        raise ValueError(
            "foreground_prior_std_mk must contain one width for the constant "
            "field and one for each beta_fg power law"
        )

    output_dir = Path(output_dir).expanduser().resolve()
    array_save_directory = Path(array_save_directory).expanduser().resolve()
    if foreground_kl_path is not None:
        foreground_kl_path = Path(foreground_kl_path).expanduser().resolve()
        # A supplied operator is an input, never a target of setup's output writers.
        output_names = ("T_21.npy", "T.npy", "Ninv.npy", "T_Ninv_T.npy", "dbar.npy", "d_ninv_d.npy")
        if foreground_kl_path in {(array_save_directory / name).resolve() for name in output_names}:
            raise ValueError("foreground_kl_path must not be a setup output path")
        reduced = np.load(foreground_kl_path, mmap_mode="r")
        if reduced.ndim != 2 or reduced.dtype != np.complex128:
            raise ValueError("The KL foreground operator must be a complex128 matrix")
        if not include_eor and reduced.shape[1] == 0:
            raise ValueError("Foreground-only inference requires at least one retained foreground mode")
    save_dir = Path(save_dir).expanduser().resolve()

    vis_dict = get_vis_data(
        data_path=config.data_path,
        inst_model=config.inst_model,
        noise_data_path=config.noise_data_path,
        sigma=config.sigma,
        noise_seed=config.noise_seed,
        calc_noise=config.calc_noise,
        antenna_selection=config.antenna_selection,
        bl_cutoff=config.bl_cutoff,
        freq_idx_min=config.freq_idx_min,
        freq_min=config.freq_min,
        freq_center=config.freq_center,
        n_freqs_select=config.n_freqs_select,
        jd_idx_min=config.jd_idx_min,
        jd_min=config.jd_min,
        jd_center=jd_center,
        n_times_select=config.n_times_select,
        form_pi=config.form_pi,
        pi_norm=config.pi_norm,
        pol=config.pol,
        redundant_avg=config.redundant_avg,
        uniform_redundancy=config.uniform_redundancy,
        n_freq=n_freq,
        f0_mhz=f0_mhz,
        df_mhz=df_mhz,
        n_t=n_t,
        d_t=d_t,
        save_vis=config.save_vis,
        save_model=config.save_model,
        save_dir=save_dir,
        clobber=config.clobber,
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
    image_geometry = build_image_geometry(
        fov_eor=float(fov_eor),
        fov_fg=float(fov_fg),
        nside=int(config.nside),
        telescope_latlonalt=tuple(float(x) for x in config.telescope_latlonalt),
        jd_center=float(jd_center),
        n_t=int(n_t),
        d_t=float(d_t),
    )
    bandwidth_hz = float(vis_dict["df_hz"]) * float(n_freq)
    d_eta = 1.0 / bandwidth_hz
    resolved_grid = {}
    fourier_geometry = None
    if include_eor:
        resolved_grid = _resolve_uv_grid_from_extents(
            extent_u_eor=float(image_geometry.fov_l_angle_eor),
            extent_v_eor=float(image_geometry.fov_m_angle_eor),
            freqs_hz=np.asarray(freqs_hz, dtype=float),
            uvws_m=np.asarray(vis_dict["uvws"], dtype=float),
            beam_type=config.beam_type,
            fwhm_deg=None if config.fwhm_deg is None else float(config.fwhm_deg),
            diam=None if config.diam is None else float(config.diam),
            n_u_eor=n_u_eor, n_v_eor=n_v_eor, du_eor=du_eor, dv_eor=dv_eor,
        )
        n_u_eor = int(resolved_grid["n_u_eor"])
        n_v_eor = int(resolved_grid["n_v_eor"])
        du_eor = float(resolved_grid["du_eor"])
        dv_eor = float(resolved_grid["dv_eor"])
        fourier_geometry = build_fourier_geometry(
            n_u=n_u_eor, n_v=n_v_eor, n_eta=int(n_eta),
            du=du_eor, dv=dv_eor, d_eta=d_eta,
        )

    setup_args_dict = {
        "model": config.model,
        "data_path": None if config.data_path is None else str(Path(config.data_path).expanduser().resolve()),
        "inst_model": None if config.inst_model is None else str(Path(config.inst_model).expanduser().resolve()),
        "noise_data_path": None if config.noise_data_path is None else str(Path(config.noise_data_path).expanduser().resolve()),
        "array_save_directory": str(array_save_directory),
        "output_dir": str(output_dir),
        **resolved_grid,
        "beta_fg": None if config.beta_fg is None else list(config.beta_fg),
        "n_freq": int(n_freq),
        "f0_mhz": float(f0_mhz),
        "df_mhz": float(df_mhz),
        "n_eta": int(n_eta),
        "n_t": int(n_t),
        "d_t": None if d_t is None else float(d_t),
        "jd_min": None if config.jd_min is None else float(config.jd_min),
        "jd_center": None if jd_center is None else float(jd_center),
        "nside": int(config.nside),
        "fov_eor": float(fov_eor),
        "fov_fg": float(fov_fg),
        "fov_l_angle_eor": float(image_geometry.fov_l_angle_eor),
        "fov_m_angle_eor": float(image_geometry.fov_m_angle_eor),
        "fov_l_angle_fg": float(image_geometry.fov_l_angle_fg),
        "fov_m_angle_fg": float(image_geometry.fov_m_angle_fg),
        "telescope_latlonalt": [float(x) for x in config.telescope_latlonalt],
        "beam_type": config.beam_type,
        "peak_amp": float(config.peak_amp),
        "fwhm_deg": None if config.fwhm_deg is None else float(config.fwhm_deg),
        "diam": None if config.diam is None else float(config.diam),
        "cosfreq": None if config.cosfreq is None else float(config.cosfreq),
        "tanh_freq": None if config.tanh_freq is None else float(config.tanh_freq),
        "tanh_sl_red": None if config.tanh_sl_red is None else float(config.tanh_sl_red),
        "pol": config.pol,
        "freq_interp_kind": config.freq_interp_kind,
        "antenna_selection": config.antenna_selection,
        "bl_cutoff": None if config.bl_cutoff is None else float(config.bl_cutoff),
        "freq_idx_min": None if config.freq_idx_min is None else int(config.freq_idx_min),
        "freq_min": None if config.freq_min is None else float(config.freq_min),
        "freq_center": None if config.freq_center is None else float(config.freq_center),
        "n_freqs_select": None if config.n_freqs_select is None else int(config.n_freqs_select),
        "jd_idx_min": None if config.jd_idx_min is None else int(config.jd_idx_min),
        "n_times_select": None if config.n_times_select is None else int(config.n_times_select),
        "form_pi": bool(config.form_pi),
        "pi_norm": float(config.pi_norm),
        "redundant_avg": bool(config.redundant_avg),
        "uniform_redundancy": bool(config.uniform_redundancy),
        "calc_noise": bool(config.calc_noise),
        "sigma": None if config.sigma is None else float(config.sigma),
        "noise_seed": int(config.noise_seed),
        "save_vis": bool(config.save_vis),
        "save_model": bool(config.save_model),
        "save_dir": str(save_dir),
        "clobber": bool(config.clobber),
        "effective_noise_path": None if config.effective_noise_path is None else str(Path(config.effective_noise_path).expanduser().resolve()),
        "priors": None if config.priors is None else np.asarray(config.priors, dtype=float).tolist(),
        "uniform_priors": None if config.uniform_priors is None else np.asarray(config.uniform_priors, dtype=bool).astype(int).tolist(),
        "foreground_prior_std_mk": list(config.foreground_prior_std_mk) if include_fg and foreground_kl_path is None else None,
        "foreground_kl_path": str(foreground_kl_path) if foreground_kl_path is not None else None,
        "use_intrinsic_noise_fitting": bool(config.use_intrinsic_noise_fitting),
    }

    # Cosmology-derived physical cube dimensions.
    # These map the instrumental coordinate extents to comoving Mpc scales and
    # are required for k-bin definitions and power normalization.
    cosmo = Cosmology()
    z = float(cosmo.f2z(jnp.mean(freqs_hz)))
    ps_box_size_para_mpc = float(cosmo.dL_df(z) * bandwidth_hz)
    ps_box_size_ra_mpc = float(cosmo.dL_dth(z) * float(image_geometry.fov_l_angle_eor))
    ps_box_size_dec_mpc = float(cosmo.dL_dth(z) * float(image_geometry.fov_m_angle_eor))

    # Convert the exact retained model modes to cosmological coordinates before
    # assigning spherical bins.
    k_vals = jnp.empty(0, dtype=jnp.float64)
    mod_k_vo = k_vals
    bid_compact = jnp.empty(0, dtype=jnp.int32)
    valid_compact = jnp.empty(0, dtype=bool)
    if include_eor:
        mod_k_vo = generate_k_modes_in_physical_coordinates(
            fourier_geometry.uv, fourier_geometry.eta,
            transverse_distance_mpc=float(cosmo.dL_dth(z)),
            line_of_sight_distance_mpc_per_hz=float(cosmo.dL_df(z)),
        )
        bid_full, valid_full, _, _, counts = generate_k_cube_model_spherical_binning(
            mod_k_vo=mod_k_vo, ps_box_size_para_mpc=ps_box_size_para_mpc,
        )
        k_vals_full = calc_mean_binned_k_vals(
            mod_k_vo=mod_k_vo, bin_id_per_k_mode=bid_full,
            valid_mask=valid_full, num_k_bins=int(counts.size),
        )
        k_vals, bid_compact, valid_compact = compress_to_nonempty_k_bins(
            k_vals_full=k_vals_full, counts=counts,
            bin_id_per_k_mode=bid_full, valid_mask=valid_full,
        )
    if jax.process_index() == 0:
        if include_eor:
            write_analysis_metadata(
                output_dir=output_dir, args=setup_args_dict, k_vals=k_vals,
                mod_k_vo=mod_k_vo, bin_id_per_k_mode=bid_compact,
                valid_mask=valid_compact, clobber=config.clobber,
            )
        else:
            write_log_files(parser=None, args=setup_args_dict, out_dir=output_dir)
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("k-vals-metadata-saved")

    # Optional per-visibility effective-noise model.
    # If provided, this overrides sigma/redundancy-based covariance construction.
    effective_noise = None
    if config.effective_noise_path is not None:
        effective_noise = jnp.asarray(
            load_numpy_dict(config.effective_noise_path, as_jax=True),
            dtype=jnp.complex128,
        ).reshape(-1)
        expected_n_vis = int(vis_dict["vis_noisy"].size)
        if int(effective_noise.size) != expected_n_vis:
            raise ValueError(
                "effective_noise_path length mismatch: "
                f"got {int(effective_noise.size)} values, expected {expected_n_vis} "
                "(must match vis_noisy size = n_t * n_freq * n_bls)."
            )

    # Prepare only physical-model inputs; likelihood preparation is a later stage.
    bm = BuildMatrices(
        config,
        # Keep the established regular-channel construction and normalization.
        frequencies_hz=(f0_mhz + df_mhz * jnp.arange(n_freq, dtype=jnp.float64)) * 1e6,
        image_geometry=image_geometry,
        fourier_geometry=fourier_geometry,
        uvw_array_m=vis_dict["uvws"],
    )

    # Select the foreground response and its matching prior together. Everything
    # downstream uses the same joint operator and likelihood-product functions.
    t21_path = None
    if include_eor:
        t21_path = ensure_cached(array_save_directory / "T_21.npy", bm.T_21_build_save, clobber=config.clobber)
    foreground_path = None
    foreground_std = np.empty(0)
    if include_fg and foreground_kl_path is None:
        foreground_std = np.repeat(config.foreground_prior_std_mk, bm.n_fg_spatial_modes)
        if (not np.isrealobj(foreground_std) or not np.all(np.isfinite(foreground_std))
                or np.any(foreground_std <= 0)):
            raise ValueError("foreground prior widths must be finite, real and positive")
        foreground_path = ensure_cached(array_save_directory / "T_fg.npy", bm.T_fg_build_save, clobber=config.clobber)
    elif include_fg:
        foreground_path = foreground_kl_path
        if reduced.shape[0] != np.asarray(vis_dict["vis_noisy"]).size:
            raise ValueError("The KL foreground operator must match the visibility rows")
        foreground_std = np.ones(reduced.shape[1])
    t_path = array_save_directory / "T.npy"
    ensure_cached(
        t_path,
        lambda: assemble_joint_operator(t21_path, foreground_path, output_path=t_path),
        dependencies=tuple(path for path in (t21_path, foreground_path) if path is not None),
        clobber=config.clobber,
    )

    # Noise belongs to the observation, not to either coefficient representation.
    variance = build_noise_variance(
        int(n_t), int(n_freq), bm.nbls, sigma=config.sigma,
        redundancy=vis_dict["redundancy"], effective_noise=effective_noise,
    )
    ninv_path = array_save_directory / "Ninv.npy"
    # This small vector can be compared exactly, so changed noise cannot reuse G.
    noise_changed = not ninv_path.exists() or not np.array_equal(np.load(ninv_path), np.asarray(1.0 / variance))
    ensure_cached(ninv_path, lambda: save_array(ninv_path, 1.0 / variance), clobber=config.clobber or noise_changed)

    # Only now prepare the likelihood products from the selected joint matrix.
    vis_noisy = jnp.asarray(vis_dict["vis_noisy"], dtype=jnp.complex128)
    save_likelihood_products(
        t_path, ninv_path, vis_noisy, output_directory=array_save_directory, clobber=config.clobber,
    )
    n_params = int(np.load(t_path, mmap_mode="r").shape[1])
    setup_args_dict["n_pix_fg_final"] = int(bm.image_geometry.n_pix_fg)

    # Assemble sampler-space prior and latent-space precision configs.
    # `Prior` now works directly in log10(power) sampler coordinates.
    # `PhiInverse` maps linearized power amplitudes to latent precision diag.
    if include_eor and config.priors is None:
        raise ValueError("priors must be provided")

    bounds_log = jnp.asarray(config.priors, dtype=jnp.float64) if include_eor else jnp.empty((0, 2))
    if bounds_log.ndim != 2 or bounds_log.shape[1] != 2:
        raise ValueError("priors must have shape (n_power_params, 2)")
    n_power_params = int(k_vals.shape[0])
    if bounds_log.shape[0] != n_power_params:
        raise ValueError(
            f"priors has {bounds_log.shape[0]} bins, but k binning produced {n_power_params} non-empty bins"
        )

    if config.uniform_priors is None or not include_eor:
        uniform_priors_mask = jnp.zeros((n_power_params,), dtype=bool)
    else:
        uniform_priors_mask = jnp.asarray(config.uniform_priors, dtype=bool)
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
        foreground_prior_std=foreground_std,
    )
    
    likelihood_class = MarginalisedLikelihood(
        matrix_dir=str(array_save_directory),
        n_data=int(vis_noisy.size),
        phi_inv_class=phi_inverse_class,
        prior_class=prior_class,
        uniform_priors=uniform_priors_mask,
        intrinsic_noise_fitting=bool(config.use_intrinsic_noise_fitting),
        power_param_start=0,
        n_power_params=n_power_params,
    )

    # Return one object containing both configs and diagnostics needed by
    # run_nestedsampling.
    return SetupResult(
        config=config,
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
        model=config.model,
    )
