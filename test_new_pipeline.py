#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from pyuvdata import UVData

SECS_PER_DAY = 24 * 60 * 60


def _load_uvdata(path: Path) -> UVData:
    uvd = UVData()
    try:
        uvd.read(path, read_data=False)
    except TypeError:
        uvd.read(path)
    return uvd


def _select_times(uvd: UVData, n_t: int) -> np.ndarray:
    times = np.unique(uvd.time_array)
    if times.size < n_t:
        raise ValueError(f"File has only {times.size} unique times")
    return times[:n_t]


def _uvw_by_time(uvd: UVData, times: np.ndarray) -> np.ndarray:
    uvw_rows = []
    for t in times:
        mask = uvd.time_array == t
        if not np.any(mask):
            raise ValueError(f"No BLTs found for time {t}")
        order = np.argsort(uvd.baseline_array[mask])
        uvw_rows.append(uvd.uvw_array[mask][order])
    nbls = {row.shape[0] for row in uvw_rows}
    if len(nbls) != 1:
        raise ValueError(f"Inconsistent Nbls across times: {sorted(nbls)}")
    return np.stack(uvw_rows, axis=0)


def _freqs_mhz(uvd: UVData, n_freq: int) -> np.ndarray:
    freqs = uvd.freq_array
    if freqs.ndim > 1:
        freqs = freqs.squeeze()
    if freqs.size < n_freq:
        raise ValueError(f"File has only {freqs.size} freqs")
    return freqs[:n_freq] / 1e6


def _summarize_matrix(path: Path) -> None:
    if not path.exists():
        print(f"{path.name}: <missing>")
        return
    if path.suffix == ".npz":
        with np.load(path) as npz:
            shape = tuple(npz["shape"].tolist())
            nnz = npz["data"].shape[0]
        print(f"{path.name}: shape={shape} nnz={nnz}")
        return
    if path.suffix == ".npy":
        arr = np.load(path)
        print(f"{path.name}: shape={arr.shape} dense")
        return
    print(f"{path.name}: <unknown format>")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "BayesEoR-JAX"))
    from build_matricies.overall import BuildMatrices

    parser = argparse.ArgumentParser(
        description="Build matrices with the new JAX pipeline."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/Users/jacobtutt/Desktop/PhD/BayesEoR/test_data/eor/eor.uvh5"),
        help="Path to pyuvdata file (default: /Users/jacobtutt/Desktop/PhD/BayesEoR/test_data/eor/eor.uvh5)",
    )
    parser.add_argument("--n-t", type=int, default=2, help="Number of times")
    parser.add_argument("--n-freq", type=int, default=3, help="Number of freqs")
    parser.add_argument("--n-eta", type=int, default=5, help="Number of eta modes")
    parser.add_argument("--n-u-eor", type=int, default=4, help="EoR nu")
    parser.add_argument("--n-v-eor", type=int, default=4, help="EoR nv")
    parser.add_argument("--n-u-fg", type=int, default=4, help="FG nu")
    parser.add_argument("--n-v-fg", type=int, default=4, help="FG nv")
    parser.add_argument("--nside", type=int, default=32, help="HEALPix nside")
    parser.add_argument("--fov-eor", type=float, default=10.0, help="EoR FoV (deg)")
    parser.add_argument("--fov-fg", type=float, default=10.0, help="FG FoV (deg)")
    parser.add_argument("--d-eta", type=float, default=1.0, help="Eta spacing")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise sigma")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/bayeseor_matrices_new"),
        help="Directory to write matrices",
    )
    parser.add_argument(
        "--fit-for-monopole",
        action="store_true",
        help="Include (u,v)=(0,0) in FG model",
    )
    args = parser.parse_args()

    uvd = _load_uvdata(args.path)
    times = _select_times(uvd, args.n_t)
    uvw_array_m = _uvw_by_time(uvd, times)

    freqs_mhz = _freqs_mhz(uvd, args.n_freq)
    f0_mhz = float(freqs_mhz[0])
    df_mhz = float(freqs_mhz[1] - freqs_mhz[0]) if args.n_freq > 1 else 1.0

    d_t = (times[1] - times[0]) * SECS_PER_DAY if args.n_t > 1 else None

    nbls = uvw_array_m.shape[1]
    bl_red_array = np.ones((args.n_t, nbls, 1))

    print("Configuration summary:")
    print(f"  path={args.path}")
    print(f"  n_t={args.n_t} n_freq={args.n_freq} n_eta={args.n_eta}")
    print(
        f"  n_u_eor/n_v_eor={args.n_u_eor}/{args.n_v_eor} "
        f"n_u_fg/n_v_fg={args.n_u_fg}/{args.n_v_fg}"
    )
    print(f"  nside={args.nside} fov_eor={args.fov_eor} fov_fg={args.fov_fg}")
    print(f"  f0_mhz={f0_mhz:.6f} df_mhz={df_mhz:.6f}")
    print(f"  d_eta={args.d_eta} sigma={args.sigma}")
    print(f"  fit_for_monopole={args.fit_for_monopole}")
    print(f"  uvw_array_m shape={uvw_array_m.shape} bl_red_array shape={bl_red_array.shape}")
    print(f"  out_dir={args.out_dir}")

    bm = BuildMatrices(
        n_u_eor=args.n_u_eor,
        n_v_eor=args.n_v_eor,
        n_u_fg=args.n_u_fg,
        n_v_fg=args.n_v_fg,
        du_eor=1.0,
        dv_eor=1.0,
        du_fg=1.0,
        dv_fg=1.0,
        n_freq=args.n_freq,
        f0_mhz=f0_mhz,
        df_mhz=df_mhz,
        n_eta=args.n_eta,
        d_eta=args.d_eta,
        nside=args.nside,
        fov_eor=args.fov_eor,
        fov_fg=args.fov_fg,
        n_t=args.n_t,
        d_t=d_t,
        jd_start=float(times[0]),
        beam_type="uniform",
        uvw_array_m=uvw_array_m,
        bl_red_array=bl_red_array,
        sigma=args.sigma,
        fit_for_monopole=args.fit_for_monopole,
        array_save_directory=str(args.out_dir),
    )

    # Build and save selected matrices.
    Fz = bm.build_Fz()
    bm.save_matrix("Fz", Fz)

    Finv_Fprime = bm.build_Finv_Fprime()
    bm.save_matrix("Finv_Fprime", Finv_Fprime)

    T = bm.build_T()
    bm.save_matrix("T", T)

    Ninv = bm.build_Ninv()
    bm.save_matrix("Ninv", Ninv)

    print(f"Wrote matrices to {args.out_dir}")
    _summarize_matrix(args.out_dir / "Fz.npz")
    _summarize_matrix(args.out_dir / "Finv_Fprime.npz")
    _summarize_matrix(args.out_dir / "T.npz")
    _summarize_matrix(args.out_dir / "Ninv.npz")


if __name__ == "__main__":
    main()
