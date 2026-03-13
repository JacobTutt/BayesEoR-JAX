"""High-level setup orchestration for BayesEoR-JAX analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import jax
import jax.numpy as jnp
# from jax.experimental import sparse
import numpy as np
from astropy.time import Time

from .build_matricies import BuildMatrices
from .inference.likelihood import Likelihood
from .inference.prior import PhiInverse, Prior
from .model.cosmology_transforms import Cosmology
from .model.instrument import InstrumentModel
from .model.k_cube import (
    calc_mean_binned_k_vals,
    compress_to_nonempty_k_bins,
    generate_k_cube_in_physical_coordinates,
    generate_k_cube_model_spherical_binning,
    mask_k_cube,
)
from .model.noise import generate_gaussian_noise
from .utils import load_numpy_dict
from .vis import preprocess_uvdata

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


def _get_freqs_from_uvd(uvd: Any) -> np.ndarray:
    """Extract frequency array in Hz from pyuvdata object."""
    freqs = uvd.freq_array
    if len(freqs.shape) > 1:
        freqs = freqs[0]
    return np.asarray(freqs, dtype=float)


def _resolve_uv_spacing(
    *,
    fov_deg: float,
    du: float | None,
    dv: float | None,
    label: str,
    rtol: float = 5e-2,
) -> tuple[float, float]:
    """Resolve uv spacings, deriving from FoV when omitted and warning on mismatch."""
    fov_deg = float(fov_deg)
    if fov_deg <= 0:
        raise ValueError(f"{label}: fov must be > 0 deg, got {fov_deg}")

    derived = float(1.0 / np.deg2rad(fov_deg))
    du_val = derived if du is None else float(du)
    dv_val = derived if dv is None else float(dv)

    if du is not None and not np.isclose(du_val, derived, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: du={du_val:.6g} is inconsistent with FoV-derived value "
            f"{derived:.6g} (=1/deg2rad({fov_deg:.6g})).",
            RuntimeWarning,
            stacklevel=2,
        )
    if dv is not None and not np.isclose(dv_val, derived, rtol=rtol, atol=0.0):
        warnings.warn(
            f"{label}: dv={dv_val:.6g} is inconsistent with FoV-derived value "
            f"{derived:.6g} (=1/deg2rad({fov_deg:.6g})).",
            RuntimeWarning,
            stacklevel=2,
        )

    return du_val, dv_val


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
    # Normalize and validate the input data path once at the boundary.
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    vis: jax.Array
    noise_est: jax.Array | None
    antpairs: Any | None
    uvd = None

    if data_path.suffix in {".uvh5", ".uvfits", ".ms"}:
        # Raw-file mode:
        # - select frequencies/times/baselines in vis.preprocess_uvdata
        # - build Hermitian-expanded data vector + instrument arrays
        # - optionally estimate empirical noise from even/odd differencing
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

        # Extract frequency/time metadata directly from the selected UVData.
        freqs_hz = _get_freqs_from_uvd(uvd)
        jds = np.unique(uvd.time_array)
        if jds.size > 1:
            dt = float((Time(jds[1], format="jd") - Time(jds[0], format="jd")).to("s").value)
        else:
            dt = float(d_t) if d_t is not None else 0.0
    elif data_path.suffix == ".npy":
        # Preprocessed-vector mode:
        # - load visibilities from disk
        # - load instrument model from an instrument directory
        # - construct frequency/time axes from parser inputs
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

        # In this mode, `n_freq/f0/df` define the analysis band explicitly.
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
        # If a noise vector is supplied, treat input visibilities as already noisy.
        noise = jnp.asarray(load_numpy_dict(noise_data_path, as_jax=True), dtype=jnp.complex128).reshape(-1)
        vis_noisy = vis
        bl_conj_pairs_map: dict[int, int] = {}
    elif noise_est is not None:
        # Reuse empirical noise estimated during raw UVData preprocessing.
        noise = noise_est
        vis_noisy = vis
        bl_conj_pairs_map = {}
    else:
        # Otherwise inject synthetic Gaussian noise using the instrument redundancy model.
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
    du_eor: float | None = None,
    dv_eor: float | None = None,
    n_u_fg: int | None = None,
    n_v_fg: int | None = None,
    du_fg: float | None = None,
    dv_fg: float | None = None,
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
    effective_noise_path: Path | str | None = None,
    # Prior / likelihood
    priors: Any = None,
    uniform_priors: Any = None,
    inverse_lw_power: float = 1e-16,
    use_intrinsic_noise_fitting: bool = False,
    **_: Any,
) -> SetupResult:
    """
    Build all objects required to run posterior sampling in BayesEoR-JAX.

    This is the main setup entry point used by :mod:`bayeseor_jax.run`. It:

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

    n_u_eor, n_v_eor, du_eor, dv_eor : int/float
        EoR model UV grid shape and spacing.
    n_u_fg, n_v_fg, du_fg, dv_fg : int/float, optional
        FG model UV grid shape and spacing.
    fit_for_monopole : bool, optional
        Whether to include a dedicated FG monopole block.
    n_modes_fg, n_powerlaws_fg, beta_fg : optional
        FG large spectral scale model controls.

    n_freq, f0_mhz, df_mhz : optional
        Frequency-axis setup (overridden by data-driven values in raw mode).
    n_eta : int, optional
        Number of line-of-sight Fourier modes. Defaults to ``n_freq``.
    n_t, d_t, jd_center : optional
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
        Boolean mask of length ``n_power_params`` where True bins are sampled
        directly in linear power space and False bins are sampled in log10
        space. If None, all bins are sampled in log10 space.
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
    # Resolve parser-like defaults once so downstream uses a fully-specified config.
    if n_v_eor is None:
        n_v_eor = n_u_eor
    if n_u_fg is None:
        n_u_fg = n_u_eor
    if n_v_fg is None:
        n_v_fg = n_u_fg
    if fov_fg is None:
        fov_fg = fov_eor
    du_eor, dv_eor = _resolve_uv_spacing(
        fov_deg=float(fov_eor),
        du=du_eor,
        dv=dv_eor,
        label="EoR uv grid spacing",
    )
    du_fg, dv_fg = _resolve_uv_spacing(
        fov_deg=float(fov_fg),
        du=du_fg,
        dv=dv_fg,
        label="FG uv grid spacing",
    )
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

    # Cosmology-derived physical cube dimensions.
    # These map the instrumental coordinate extents to comoving Mpc scales and
    # are required for k-bin definitions and power normalization.
    cosmo = Cosmology()
    z = float(cosmo.f2z(jnp.mean(freqs_hz)))
    bandwidth_hz = float(vis_dict["df_hz"]) * float(n_freq)
    ps_box_size_para_mpc = float(cosmo.dL_df(z) * bandwidth_hz)
    ps_box_size_ra_mpc = float(cosmo.dL_dth(z) * np.deg2rad(float(fov_eor)))
    ps_box_size_dec_mpc = float(cosmo.dL_dth(z) * np.deg2rad(float(fov_eor)))

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
        effective_noise=effective_noise,
        array_save_directory=str(array_save_directory),
    )
    bm.build_minimum_matrix_stack()

    # Load compressed matrix terms and form:
    #   dbar = T^H N^-1 d
    #   d_Ninv_d = d^H N^-1 d
    # using sparse matvecs upstream to avoid unnecessary dense materialization.
    T = bm.load_matrix("T")
    Ninv = bm.load_matrix("Ninv")
    T_Ninv_T = bm.load_matrix("T_Ninv_T")
    if hasattr(T, "todense"):
        T = T.todense()
    T = jnp.asarray(T)

    vis_noisy = jnp.asarray(vis_dict["vis_noisy"], dtype=jnp.complex128)
    Ninv_d = Ninv @ vis_noisy
    dbar = jnp.conj(T).T @ Ninv_d
    d_ninv_d = jnp.real(jnp.vdot(vis_noisy, Ninv_d))

    # Assemble sampler-space prior and latent-space precision configs.
    # `Prior` now receives log-space bounds and constructs sampler-space bounds.
    # `PhiInverse` maps sampled power amplitudes to latent precision diag.
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
        uniform_priors=uniform_priors_mask,
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
        n_par=int(T_Ninv_T.shape[0]),
        inverse_lw_power=float(inverse_lw_power),
    )
    
    likelihood_class = Likelihood(
        t_ninv_t=jnp.asarray(T_Ninv_T),
        dbar=jnp.asarray(dbar),
        d_ninv_d=jnp.asarray(d_ninv_d),
        n_data=int(vis_noisy.size),
        phi_inv_class=phi_inverse_class,
        prior_class=prior_class,
        uniform_priors=uniform_priors_mask,
        intrinsic_noise_fitting=bool(use_intrinsic_noise_fitting),
        power_param_start=0,
        n_power_params=n_power_params,
    )

    # Return one object containing both configs and diagnostics needed by run.py.
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
