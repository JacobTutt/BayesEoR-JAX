"""High-level setup orchestration for BayesEoR-JAX analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np
from astropy.time import Time

from .build_matricies import BuildMatrices
from .likelihood import LikelihoodConfig
from .model.cosmology_transforms import Cosmology
from .model.instrument import InstrumentModel
from .model.k_cube import (
    calc_mean_binned_k_vals,
    generate_k_cube_in_physical_coordinates,
    generate_k_cube_model_spherical_binning,
    mask_k_cube,
)
from .model.noise import generate_gaussian_noise
from .prior import PhiInverseConfig, PriorConfig
from .utils import load_numpy_dict, parse_uprior_inds
from .vis import preprocess_uvdata

jax.config.update("jax_enable_x64", True)


class SetupResult:
    """Container for setup outputs consumed by downstream samplers."""

    def __init__(
        self,
        likelihood_cfg: LikelihoodConfig,
        prior_cfg: PriorConfig,
        phi_cfg: PhiInverseConfig,
        bm: BuildMatrices,
        vis_dict: dict[str, Any],
        k_vals: jax.Array,
        bin_id_per_k_mode: jax.Array,
        valid_mask: jax.Array,
        ps_box_size_ra_mpc: float,
        ps_box_size_dec_mpc: float,
        ps_box_size_para_mpc: float,
    ) -> None:
        self.likelihood_cfg = likelihood_cfg
        self.prior_cfg = prior_cfg
        self.phi_cfg = phi_cfg
        self.bm = bm
        self.vis_dict = vis_dict
        self.k_vals = k_vals
        self.bin_id_per_k_mode = bin_id_per_k_mode
        self.valid_mask = valid_mask
        self.ps_box_size_ra_mpc = ps_box_size_ra_mpc
        self.ps_box_size_dec_mpc = ps_box_size_dec_mpc
        self.ps_box_size_para_mpc = ps_box_size_para_mpc


def _sparse_matvec(matrix: sparse.BCOO, vector: jax.Array) -> jax.Array:
    """Multiply sparse matrix by dense vector in JAX."""
    return sparse.bcoo_dot_general(
        matrix,
        vector,
        dimension_numbers=(((1,), (0,)), ((), ())),
    )


def _compress_to_nonempty_k_bins(
    k_vals_full: jax.Array,
    counts: jax.Array,
    bin_id_per_k_mode: jax.Array,
    valid_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Drop empty spherical bins and remap voxel bin IDs to a compact index range.
    """
    keep_bins = counts > 0
    k_vals = k_vals_full[keep_bins]

    # Old-id -> new-id map, with -1 for empty bins.
    n_bins = int(k_vals_full.shape[0])
    old_to_new = -jnp.ones((n_bins,), dtype=jnp.int32)
    kept_old_ids = jnp.where(keep_bins)[0]
    old_to_new = old_to_new.at[kept_old_ids].set(
        jnp.arange(kept_old_ids.shape[0], dtype=jnp.int32)
    )

    bid = jnp.asarray(bin_id_per_k_mode, dtype=jnp.int32)
    valid = jnp.asarray(valid_mask, dtype=bool)
    bid_clip = jnp.clip(bid, 0, n_bins - 1)
    mapped = old_to_new[bid_clip]
    compact_valid = valid & (mapped >= 0)
    compact_bid = jnp.where(compact_valid, mapped, -1).astype(jnp.int32)

    return k_vals, compact_bid, compact_valid


def _get_freqs_from_uvd(uvd: Any) -> np.ndarray:
    """Extract frequency array in Hz from pyuvdata object."""
    freqs = uvd.freq_array
    if len(freqs.shape) > 1:
        freqs = freqs[0]
    return np.asarray(freqs, dtype=float)


def get_vis_data(
    *,
    data_path: Path | str,
    inst_model: Path | str | None = None,
    noise_data_path: Path | str | None = None,
    sigma: float | None = None,
    noise_seed: int = 742123,
    calc_noise: bool = False,
    # Raw-uvdata preprocessing controls.
    antenna_selection: str = "cross",
    bl_cutoff: float | None = None,
    freq_idx_min: int | None = None,
    freq_min: float | None = None,
    freq_center: float | None = None,
    n_freqs_select: int | None = None,
    jd_idx_min: int | None = None,
    jd_min: float | None = None,
    jd_center: float | None = None,
    n_times_select: int | None = None,
    form_pi: bool = True,
    pi_norm: float = 1.0,
    pol: str = "xx",
    redundant_avg: bool = False,
    uniform_redundancy: bool = False,
    # For preprocessed vectors.
    n_freq: int | None = None,
    f0_mhz: float | None = None,
    df_mhz: float | None = None,
    n_t: int | None = None,
    d_t: float | None = None,
    # Save hooks for raw preprocessing.
    save_vis: bool = False,
    save_model: bool = False,
    save_dir: Path | str = "./",
    clobber: bool = False,
) -> dict[str, Any]:
    """
    Load visibilities and associated instrument/noise arrays.

    Supports two input modes:
    1. Raw pyuvdata file (`.uvh5`, `.uvfits`, `.ms`) via `preprocess_uvdata`.
    2. Preprocessed `.npy` visibility vector plus an instrument-model directory.
    """
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    vis: jax.Array
    noise_est: jax.Array | None
    antpairs: Any | None
    uvd = None

    if data_path.suffix in {".uvh5", ".uvfits", ".ms"}:
        out = preprocess_uvdata(
            filepath=data_path,
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
            calc_noise=calc_noise,
            return_uvd=True,
            save_vis=save_vis,
            save_model=save_model,
            save_dir=save_dir,
            clobber=clobber,
        )
        vis_np, antpairs, uvws_np, red_np, noise_np, uvd = out
        vis = jnp.asarray(vis_np, dtype=jnp.complex128)
        uvws = jnp.asarray(uvws_np, dtype=jnp.float64)
        redundancy = jnp.asarray(red_np)
        noise_est = None if noise_np is None else jnp.asarray(noise_np, dtype=jnp.complex128)

        freqs_hz = _get_freqs_from_uvd(uvd)
        jds = np.unique(uvd.time_array)
        if jds.size > 1:
            dt = float((Time(jds[1], format="jd") - Time(jds[0], format="jd")).to("s").value)
        else:
            dt = float(d_t) if d_t is not None else 0.0
    elif data_path.suffix == ".npy":
        if inst_model is None:
            raise ValueError("inst_model is required when loading a preprocessed .npy vector")
        if n_freq is None or f0_mhz is None or df_mhz is None or n_t is None or d_t is None:
            raise ValueError(
                "n_freq, f0_mhz, df_mhz, n_t, and d_t are required for preprocessed .npy inputs"
            )

        vis = jnp.asarray(load_numpy_dict(data_path, as_jax=True), dtype=jnp.complex128).reshape(-1)
        inst = InstrumentModel(inst_model)
        uvws = jnp.asarray(inst.uvw_array_m, dtype=jnp.float64)
        redundancy = jnp.asarray(inst.bl_red_array)
        antpairs = inst.antpairs
        noise_est = None

        freqs_hz = (float(f0_mhz) + float(df_mhz) * np.arange(int(n_freq), dtype=float)) * 1e6
        if jd_center is None:
            raise ValueError("jd_center is required for preprocessed .npy inputs")
        jds = Time(float(jd_center), format="jd") + np.arange(
            -(int(n_t) // 2), int(n_t) // 2 + int(n_t) % 2
        ) * (float(d_t) / 86400.0)
        jds = jds.jd
        dt = float(d_t)
    else:
        raise ValueError(f"Unsupported data_path suffix '{data_path.suffix}'")

    # Optional external noise vector.
    if noise_data_path is not None:
        noise = jnp.asarray(load_numpy_dict(noise_data_path, as_jax=True), dtype=jnp.complex128).reshape(-1)
        vis_noisy = vis
        bl_conj_pairs_map: dict[int, int] = {}
    elif noise_est is not None:
        # Raw-data empirical noise estimate from even/odd differencing path.
        noise = noise_est
        vis_noisy = vis
        bl_conj_pairs_map = {}
    else:
        if sigma is None:
            raise ValueError("sigma is required when no noise vector is supplied/estimated")
        vis_noisy, noise, bl_conj_pairs_map = generate_gaussian_noise(
            sigma=float(sigma),
            s=vis,
            n_freq=int(freqs_hz.size),
            n_t=int(np.unique(jds).size),
            uvw_array_m=uvws,
            bl_red_array=redundancy,
            random_seed=noise_seed,
        )

    df_hz = float(freqs_hz[1] - freqs_hz[0]) if freqs_hz.size > 1 else float(df_mhz) * 1e6

    return {
        "vis": vis,
        "vis_noisy": vis_noisy,
        "noise": noise,
        "bl_conj_pairs_map": bl_conj_pairs_map,
        "uvws": uvws,
        "redundancy": redundancy,
        "antpairs": antpairs,
        "freqs_hz": jnp.asarray(freqs_hz, dtype=jnp.float64),
        "df_hz": float(df_hz),
        "jds": jnp.asarray(jds, dtype=jnp.float64),
        "dt": float(dt),
        "uvd": uvd,
    }


def run_setup(
    *,
    # IO
    data_path: Path | str,
    inst_model: Path | str | None = None,
    noise_data_path: Path | str | None = None,
    array_save_directory: Path | str = "./matrices/",
    # UV model
    n_u_eor: int,
    n_v_eor: int | None = None,
    du_eor: float = 1.0,
    dv_eor: float = 1.0,
    n_u_fg: int | None = None,
    n_v_fg: int | None = None,
    du_fg: float = 1.0,
    dv_fg: float = 1.0,
    fit_for_monopole: bool = False,
    n_modes_fg: int = 0,
    n_powerlaws_fg: int | None = None,
    beta_fg: list[float] | None = None,
    # Frequency / time
    n_freq: int | None = None,
    f0_mhz: float | None = None,
    df_mhz: float | None = None,
    n_eta: int | None = None,
    n_t: int = 1,
    d_t: float | None = None,
    jd_center: float = 2461040.5,
    # Sky / beam
    nside: int = 256,
    fov_eor: float = 10.0,
    fov_fg: float | None = None,
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
    clobber: bool = False,
    # Prior / likelihood
    priors: Any = None,
    log_priors: bool = True,
    uprior_bins: str = "",
    inverse_lw_power: float = 1e-16,
    use_intrinsic_noise_fitting: bool = False,
    **_: Any,
) -> SetupResult:
    """
    Build all objects required for posterior sampling.

    The returned object contains:
    - compressed data products (`vis_noisy`, `dbar`, `T_Ninv_T`, ...)
    - k-binning metadata
    - instantiated `PriorConfig`, `PhiInverseConfig`, `LikelihoodConfig`.
    """
    # Resolve defaults mirroring parser behavior.
    if n_v_eor is None:
        n_v_eor = n_u_eor
    if n_u_fg is None:
        n_u_fg = n_u_eor
    if n_v_fg is None:
        n_v_fg = n_u_fg
    if fov_fg is None:
        fov_fg = fov_eor
    if n_eta is None:
        if n_freq is None:
            raise ValueError("n_eta is None and n_freq is None; one must be provided")
        n_eta = n_freq
    if n_powerlaws_fg is None:
        n_powerlaws_fg = len(beta_fg) if (beta_fg is not None and n_modes_fg > 0) else 0
    if n_modes_fg > n_powerlaws_fg and n_powerlaws_fg > 0:
        n_modes_fg = n_powerlaws_fg

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

    # Data-driven values overwrite nominal inputs when reading raw uvdata.
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

    # Cosmology-derived physical cube dimensions.
    cosmo = Cosmology()
    z = float(cosmo.f2z(jnp.mean(freqs_hz)))
    bandwidth_hz = float(vis_dict["df_hz"]) * float(n_freq)
    ps_box_size_para_mpc = float(cosmo.dL_df(z) * bandwidth_hz)
    ps_box_size_ra_mpc = float(cosmo.dL_dth(z) * np.deg2rad(float(fov_eor)))
    ps_box_size_dec_mpc = float(cosmo.dL_dth(z) * np.deg2rad(float(fov_eor)))

    # Build k-cube and spherical bin assignments.
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
    k_vals, bid_compact, valid_compact = _compress_to_nonempty_k_bins(
        k_vals_full=k_vals_full,
        counts=counts,
        bin_id_per_k_mode=bid_full,
        valid_mask=valid_full,
    )

    # Matrix construction and compression terms.
    bm = BuildMatrices(
        n_u_eor=int(n_u_eor),
        du_eor=float(du_eor),
        n_v_eor=int(n_v_eor),
        dv_eor=float(dv_eor),
        n_u_fg=int(n_u_fg),
        du_fg=float(du_fg),
        n_v_fg=int(n_v_fg),
        dv_fg=float(dv_fg),
        n_modes_fg=int(n_modes_fg),
        n_powerlaws_fg=int(n_powerlaws_fg),
        beta_fg=beta_fg,
        n_freq=int(n_freq),
        f0_mhz=float(f0_mhz),
        df_mhz=float(df_mhz),
        n_eta=int(n_eta),
        d_eta=float(1.0 / bandwidth_hz),
        fit_for_monopole=fit_for_monopole,
        telescope_latlonalt=tuple(float(x) for x in telescope_latlonalt),
        jd_center=float(jd_center),
        nside=int(nside),
        fov_eor=float(fov_eor),
        fov_fg=float(fov_fg),
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
        effective_noise=None,
        array_save_directory=str(array_save_directory),
    )
    bm.build_minimum_matrix_stack()

    T = bm.load_matrix("T")
    Ninv = bm.load_matrix("Ninv")
    T_Ninv_T = bm.load_matrix("T_Ninv_T")

    vis_noisy = jnp.asarray(vis_dict["vis_noisy"], dtype=jnp.complex128)
    Ninv_d = _sparse_matvec(Ninv, vis_noisy)
    dbar = _sparse_matvec(T.conjugate().T, Ninv_d)
    d_ninv_d = jnp.real(jnp.vdot(vis_noisy, Ninv_d))

    # Prior and likelihood configs.
    if priors is None:
        raise ValueError("priors must be provided")
    bounds = jnp.asarray(priors, dtype=jnp.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("priors must have shape (n_power_params, 2)")
    if bounds.shape[0] != k_vals.shape[0]:
        raise ValueError(
            f"priors has {bounds.shape[0]} bins, but k binning produced {k_vals.shape[0]} non-empty bins"
        )

    uprior_inds = parse_uprior_inds(uprior_bins, int(k_vals.shape[0]))
    prior_cfg = PriorConfig(
        bounds=bounds,
        log_priors=bool(log_priors),
        uprior_inds=uprior_inds,
        power_param_start=0,
        n_power_params=int(k_vals.shape[0]),
    )
    phi_cfg = PhiInverseConfig(
        bin_id_per_k_mode=bid_compact,
        valid_mask=valid_compact,
        k_vals=k_vals,
        ps_box_size_ra_mpc=ps_box_size_ra_mpc,
        ps_box_size_dec_mpc=ps_box_size_dec_mpc,
        ps_box_size_para_mpc=ps_box_size_para_mpc,
        inst_to_cosmo_vol=float(cosmo.inst_to_cosmo_vol(z)),
        n_par=int(T_Ninv_T.shape[0]),
        inverse_lw_power=float(inverse_lw_power),
    )
    likelihood_cfg = LikelihoodConfig(
        t_ninv_t=jnp.asarray(T_Ninv_T),
        dbar=jnp.asarray(dbar),
        d_ninv_d=jnp.asarray(d_ninv_d),
        n_data=int(vis_noisy.size),
        phi_cfg=phi_cfg,
        prior_cfg=prior_cfg,
        log_priors=bool(log_priors),
        intrinsic_noise_fitting=bool(use_intrinsic_noise_fitting),
        power_param_start=0,
        n_power_params=int(k_vals.shape[0]),
    )

    return SetupResult(
        likelihood_cfg=likelihood_cfg,
        prior_cfg=prior_cfg,
        phi_cfg=phi_cfg,
        bm=bm,
        vis_dict=vis_dict,
        k_vals=k_vals,
        bin_id_per_k_mode=bid_compact,
        valid_mask=valid_compact,
        ps_box_size_ra_mpc=ps_box_size_ra_mpc,
        ps_box_size_dec_mpc=ps_box_size_dec_mpc,
        ps_box_size_para_mpc=ps_box_size_para_mpc,
    )

