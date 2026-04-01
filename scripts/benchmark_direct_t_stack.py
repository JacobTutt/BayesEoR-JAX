"""Benchmark direct dense T and T_Ninv_T construction."""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

import jax


def _maybe_initialize_distributed() -> None:
    if os.environ.get("SLURM_PROCID") is None:
        return
    if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
        return
    os.environ.setdefault("NCCL_PXN_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_PXN_LEVEL", "0")
    jax.distributed.initialize()


_maybe_initialize_distributed()

import jax.numpy as jnp  # noqa: E402
from bayeseor_jax import BayesEoRParser, load_numpy_dict  # noqa: E402
from bayeseor_jax.build_matricies import BuildMatrices  # noqa: E402
from bayeseor_jax.build_matricies.build_utils import _dense_matrix_path, _log_build, save_dense_matrix  # noqa: E402
from bayeseor_jax.setup import get_vis_data  # noqa: E402
from bayeseor_jax.utils import _resolve_uv_spacing  # noqa: E402


def _parse_args():
    parser = BayesEoRParser()
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--t-matrix-name", type=str, default="T")
    parser.add_argument("--ninv-matrix-name", type=str, default="Ninv")
    parser.add_argument("--t-ninv-t-matrix-name", type=str, default="T_Ninv_T")
    parser.add_argument("--t-ninv-t-block-cols", type=int, default=16384)
    parser.add_argument("--t-ninv-t-right-block-batch", type=int, default=4)
    return parser.parse_args()


def _build_matrices_only(args) -> BuildMatrices:
    n_u_eor = int(args.n_u_eor)
    n_v_eor = int(args.n_v_eor) if args.n_v_eor is not None else n_u_eor
    n_u_fg = int(args.n_u_fg) if args.n_u_fg is not None else n_u_eor
    n_v_fg = int(args.n_v_fg) if args.n_v_fg is not None else n_u_fg
    fov_fg = float(args.fov_fg) if args.fov_fg is not None else float(args.fov_eor)
    du_eor, dv_eor = _resolve_uv_spacing(
        fov_deg=float(args.fov_eor),
        du=args.du_eor,
        dv=args.dv_eor,
        label="EoR uv grid spacing",
    )
    du_fg, dv_fg = _resolve_uv_spacing(
        fov_deg=float(fov_fg),
        du=args.du_fg,
        dv=args.dv_fg,
        label="FG uv grid spacing",
    )
    n_eta = int(args.n_eta) if args.n_eta is not None else int(args.n_freq)
    n_powerlaws_fg = (
        int(args.n_powerlaws_fg)
        if args.n_powerlaws_fg is not None
        else (len(args.beta_fg) if args.beta_fg is not None and int(args.n_modes_fg) > 0 else 0)
    )
    n_modes_fg = int(args.n_modes_fg)
    if n_modes_fg > n_powerlaws_fg and n_powerlaws_fg > 0:
        n_modes_fg = n_powerlaws_fg

    vis_dict = get_vis_data(
        data_path=args.data_path,
        inst_model=args.inst_model,
        noise_data_path=args.noise_data_path,
        sigma=args.sigma,
        noise_seed=args.noise_seed,
        calc_noise=args.calc_noise,
        antenna_selection=args.antenna_selection,
        bl_cutoff=args.bl_cutoff,
        freq_idx_min=args.freq_idx_min,
        freq_min=args.freq_min,
        freq_center=args.freq_center,
        n_freqs_select=args.n_freqs_select,
        jd_idx_min=args.jd_idx_min,
        jd_min=args.jd_min,
        jd_center=args.jd_center,
        n_times_select=args.n_times_select,
        form_pi=args.form_pi,
        pi_norm=args.pi_norm,
        pol=args.pol,
        redundant_avg=args.redundant_avg,
        uniform_redundancy=args.uniform_redundancy,
        n_freq=args.n_freq,
        f0_mhz=args.f0_mhz,
        df_mhz=args.df_mhz,
        n_t=args.n_t,
        d_t=args.d_t,
        save_vis=args.save_vis,
        save_model=args.save_model,
        save_dir=args.save_dir,
        clobber=args.clobber,
    )

    freqs_hz = vis_dict["freqs_hz"]
    n_freq = int(freqs_hz.size)
    f0_mhz = float(freqs_hz[0] / 1e6)
    df_mhz = float(vis_dict["df_hz"] / 1e6)
    n_t = int(vis_dict["uvws"].shape[0])
    d_t = float(vis_dict["dt"]) if args.d_t is None else float(args.d_t)
    jd_center = float(vis_dict["jds"][n_t // 2]) if vis_dict["jds"].size > 0 else float(args.jd_center)

    effective_noise = None
    if args.effective_noise_path is not None:
        effective_noise = jnp.asarray(
            load_numpy_dict(args.effective_noise_path, as_jax=True),
            dtype=jnp.complex128,
        ).reshape(-1)

    bandwidth_hz = float(vis_dict["df_hz"]) * float(n_freq)

    bm_kwargs = dict(
        n_u_eor=n_u_eor,
        du_eor=float(du_eor),
        n_v_eor=n_v_eor,
        dv_eor=float(dv_eor),
        n_u_fg=n_u_fg,
        du_fg=float(du_fg),
        n_v_fg=n_v_fg,
        dv_fg=float(dv_fg),
        n_modes_fg=n_modes_fg,
        n_powerlaws_fg=n_powerlaws_fg,
        beta_fg=args.beta_fg,
        n_freq=n_freq,
        f0_mhz=f0_mhz,
        df_mhz=df_mhz,
        n_eta=n_eta,
        d_eta=float(1.0 / bandwidth_hz),
        fit_for_monopole=bool(args.fit_for_monopole),
        fg_model_domain=str(args.fg_model_domain),
        telescope_latlonalt=tuple(float(x) for x in args.telescope_latlonalt),
        jd_center=jd_center,
        nside=int(args.nside),
        fov_eor=float(args.fov_eor),
        fov_fg=float(fov_fg),
        n_t=n_t,
        d_t=d_t,
        beam_type=args.beam_type,
        peak_amp=float(args.peak_amp),
        fwhm_deg=args.fwhm_deg,
        diam=args.diam,
        cosfreq=args.cosfreq,
        tanh_freq=args.tanh_freq,
        tanh_sl_red=args.tanh_sl_red,
        pol=args.pol,
        freq_interp_kind=args.freq_interp_kind,
        uvw_array_m=vis_dict["uvws"],
        bl_red_array=vis_dict["redundancy"],
        sigma=args.sigma,
        effective_noise=effective_noise,
        array_save_directory=str(args.array_save_directory),
        t_ninv_t_block_cols=int(args.t_ninv_t_block_cols),
        t_ninv_t_right_block_batch=int(args.t_ninv_t_right_block_batch),
    )
    supported = set(inspect.signature(BuildMatrices.__init__).parameters)
    return BuildMatrices(**{k: v for k, v in bm_kwargs.items() if k in supported})


def _path_info(path: Path) -> dict[str, object]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": int(path.stat().st_size) if exists else None,
    }


def main() -> None:
    args = _parse_args()
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.array_save_directory).mkdir(parents=True, exist_ok=True)

    bm = _build_matrices_only(args)
    _log_build(
        bm,
        (
            "direct dense benchmark "
            f"process={jax.process_index() + 1}/{jax.process_count()} "
            f"devices={jax.local_device_count()}/{jax.device_count()}"
        ),
    )

    summary: dict[str, object] = {
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "local_device_count": int(jax.local_device_count()),
        "device_count": int(jax.device_count()),
        "array_save_directory": str(args.array_save_directory),
        "n_u_eor": int(args.n_u_eor),
        "n_v_eor": int(args.n_v_eor if args.n_v_eor is not None else args.n_u_eor),
        "n_u_fg": int(args.n_u_fg if args.n_u_fg is not None else args.n_u_eor),
        "n_v_fg": int(args.n_v_fg if args.n_v_fg is not None else (args.n_u_fg if args.n_u_fg is not None else args.n_u_eor)),
        "fov_eor": float(args.fov_eor),
        "fov_fg": float(args.fov_fg if args.fov_fg is not None else args.fov_eor),
        "build_timings_seconds": {},
    }

    t0 = time.perf_counter()
    _log_build(bm, f"building direct T as {args.t_matrix_name}")
    T = bm.build_T_directly()
    T.block_until_ready()
    summary["build_timings_seconds"]["T_direct_build"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    _log_build(bm, f"saving direct T as {args.t_matrix_name}")
    save_dense_matrix(bm, args.t_matrix_name, T)
    summary["build_timings_seconds"]["T_direct_save"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    _log_build(bm, f"building direct Ninv as {args.ninv_matrix_name}")
    Ninv = bm.build_Ninv()
    Ninv.block_until_ready()
    summary["build_timings_seconds"]["Ninv_direct_build"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    _log_build(bm, f"saving direct Ninv as {args.ninv_matrix_name}")
    save_dense_matrix(bm, args.ninv_matrix_name, Ninv)
    summary["build_timings_seconds"]["Ninv_direct_save"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    _log_build(bm, f"building direct T_Ninv_T as {args.t_ninv_t_matrix_name}")
    T_Ninv_T = bm.build_T_Ninv_T(T, Ninv)
    T_Ninv_T.block_until_ready()
    summary["build_timings_seconds"]["T_Ninv_T_direct_build"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    _log_build(bm, f"saving direct T_Ninv_T as {args.t_ninv_t_matrix_name}")
    save_dense_matrix(bm, args.t_ninv_t_matrix_name, T_Ninv_T)
    summary["build_timings_seconds"]["T_Ninv_T_direct_save"] = time.perf_counter() - t5

    if getattr(jax, "distributed", None) is not None and jax.distributed.is_initialized():
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("benchmark-direct-t-stack-done")

    if jax.process_index() == 0:
        summary["t_matrix"] = _path_info(_dense_matrix_path(bm, args.t_matrix_name))
        summary["ninv_matrix"] = _path_info(_dense_matrix_path(bm, args.ninv_matrix_name))
        summary["t_ninv_t_matrix"] = _path_info(_dense_matrix_path(bm, args.t_ninv_t_matrix_name))
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
