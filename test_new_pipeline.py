#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from pyuvdata import UVData

SECS_PER_DAY = 24 * 60 * 60


DEFAULT_DATA_PATH = Path("/Users/jacobtutt/Desktop/PhD/BayesEoR/test_data/eor/eor.uvh5")
DEFAULT_OLD_REPO = Path("/Users/jacobtutt/Desktop/PhD/BayesEoR")


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


def _load_new_npz(path: Path) -> np.ndarray:
    z = np.load(path)
    rows = z["indices"][:, 0]
    cols = z["indices"][:, 1]
    mat = sp.coo_matrix((z["data"], (rows, cols)), shape=tuple(z["shape"]))
    return mat.tocsr().toarray()


def _load_old_matrix(base_path: Path, name: str) -> np.ndarray:
    npz_path = base_path / f"{name}.npz"
    h5_path = base_path / f"{name}.h5"
    if npz_path.exists():
        return sp.load_npz(npz_path).tocsr().toarray()
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            if name in f:
                return np.array(f[name][...])
            first_key = next(iter(f.keys()))
            return np.array(f[first_key][...])
    raise FileNotFoundError(f"Missing old matrix {name} in {base_path}")


def _compare_matrix_pair(old_arr: np.ndarray, new_arr: np.ndarray) -> tuple[bool, float]:
    if old_arr.shape != new_arr.shape:
        return False, np.inf
    max_abs = float(np.nanmax(np.abs(old_arr - new_arr))) if old_arr.size else 0.0
    ok = bool(np.allclose(old_arr, new_arr, rtol=1e-8, atol=1e-10, equal_nan=True))
    return ok, max_abs


def _build_new(
    repo_root: Path,
    cfg: dict,
    times: np.ndarray,
    uvw_array_m: np.ndarray,
    f0_mhz: float,
    df_mhz: float,
    d_t: float | None,
    out_dir: Path,
) -> None:
    sys.path.insert(0, str(repo_root / "BayesEoR-JAX"))
    from build_matricies.overall import BuildMatrices

    out_dir.mkdir(parents=True, exist_ok=True)
    nbls = uvw_array_m.shape[1]
    bl_red_array = np.ones((cfg["n_t"], nbls, 1))

    bm = BuildMatrices(
        n_u_eor=cfg["n_u_eor"],
        n_v_eor=cfg["n_v_eor"],
        n_u_fg=cfg["n_u_fg"],
        n_v_fg=cfg["n_v_fg"],
        du_eor=1.0,
        dv_eor=1.0,
        du_fg=1.0,
        dv_fg=1.0,
        n_freq=cfg["n_freq"],
        f0_mhz=f0_mhz,
        df_mhz=df_mhz,
        n_eta=cfg["n_eta"],
        d_eta=1.0,
        n_modes_fg=cfg["n_modes_fg"],
        n_powerlaws_fg=cfg["n_powerlaws_fg"],
        nside=cfg["nside"],
        fov_eor=cfg["fov_eor"],
        fov_fg=cfg["fov_fg"],
        n_t=cfg["n_t"],
        d_t=d_t,
        jd_start=float(times[0]),
        beam_type="uniform",
        uvw_array_m=uvw_array_m,
        bl_red_array=bl_red_array,
        sigma=1.0,
        fit_for_monopole=cfg["fit_for_monopole"],
        array_save_directory=str(out_dir),
    )

    bm.save_matrix("Fz", bm.build_Fz())
    bm.save_matrix("Finv_Fprime", bm.build_Finv_Fprime())
    bm.save_matrix("T", bm.build_T())
    bm.save_matrix("Ninv", bm.build_Ninv())


def _build_old(
    old_repo: Path,
    cfg: dict,
    times: np.ndarray,
    uvw_array_m: np.ndarray,
    f0_mhz: float,
    df_mhz: float,
    d_t: float | None,
    out_dir: Path,
) -> None:
    sys.path.insert(0, str(old_repo))
    from bayeseor.matrices.build import BuildMatrices as OldBuildMatrices

    out_dir.mkdir(parents=True, exist_ok=True)
    nbls = uvw_array_m.shape[1]
    bl_red_array = np.ones((cfg["n_t"], nbls, 1))

    old = OldBuildMatrices(
        nu=cfg["n_u_eor"],
        du_eor=1.0,
        nv=cfg["n_v_eor"],
        dv_eor=1.0,
        nu_fg=cfg["n_u_fg"],
        du_fg=1.0,
        nv_fg=cfg["n_v_fg"],
        dv_fg=1.0,
        nf=cfg["n_freq"],
        neta=cfg["n_eta"],
        deta=1.0,
        fit_for_monopole=cfg["fit_for_monopole"],
        f_min=f0_mhz,
        df=df_mhz,
        nq=cfg["n_modes_fg"],
        npl=cfg["n_powerlaws_fg"],
        beta=[2.63, 2.82],
        sigma=1.0,
        nside=cfg["nside"],
        fov_ra_eor=cfg["fov_eor"],
        fov_dec_eor=cfg["fov_eor"],
        fov_ra_fg=cfg["fov_fg"],
        fov_dec_fg=cfg["fov_fg"],
        simple_za_filter=True,
        include_instrumental_effects=True,
        telescope_latlonalt=(0.0, 0.0, 0.0),
        nt=cfg["n_t"],
        jd_center=float(times[0]),
        dt=d_t,
        beam_type="uniform",
        drift_scan=True,
        uvw_array_m=uvw_array_m,
        bl_red_array=bl_red_array,
        array_save_directory=str(out_dir) + "/",
        use_sparse_matrices=True,
        Finv_Fprime=True,
        verbose=False,
    )

    old.build_Fz()
    old.build_Finv_Fprime()
    old.build_T()
    old.build_Ninv()


def _configs() -> list[dict]:
    # Keep u/v odd in all cases.
    return [
        {
            "name": "cfg01_baseline_5x5",
            "n_u_eor": 5,
            "n_v_eor": 5,
            "n_u_fg": 5,
            "n_v_fg": 5,
            "n_freq": 3,
            "n_eta": 5,
            "n_modes_fg": 2,
            "n_powerlaws_fg": 2,
            "nside": 32,
            "fov_eor": 10.0,
            "fov_fg": 10.0,
            "n_t": 2,
            "fit_for_monopole": True,
        },
        {
            "name": "cfg02_small_3x3",
            "n_u_eor": 3,
            "n_v_eor": 3,
            "n_u_fg": 3,
            "n_v_fg": 3,
            "n_freq": 3,
            "n_eta": 5,
            "n_modes_fg": 2,
            "n_powerlaws_fg": 2,
            "nside": 32,
            "fov_eor": 10.0,
            "fov_fg": 10.0,
            "n_t": 2,
            "fit_for_monopole": True,
        },
        {
            "name": "cfg03_fg_wider",
            "n_u_eor": 5,
            "n_v_eor": 5,
            "n_u_fg": 7,
            "n_v_fg": 7,
            "n_freq": 3,
            "n_eta": 5,
            "n_modes_fg": 2,
            "n_powerlaws_fg": 2,
            "nside": 32,
            "fov_eor": 8.0,
            "fov_fg": 12.0,
            "n_t": 2,
            "fit_for_monopole": True,
        },
        {
            "name": "cfg04_single_time",
            "n_u_eor": 5,
            "n_v_eor": 5,
            "n_u_fg": 5,
            "n_v_fg": 5,
            "n_freq": 2,
            "n_eta": 5,
            "n_modes_fg": 2,
            "n_powerlaws_fg": 2,
            "nside": 32,
            "fov_eor": 10.0,
            "fov_fg": 10.0,
            "n_t": 1,
            "fit_for_monopole": True,
        },
        {
            "name": "cfg05_low_nside",
            "n_u_eor": 5,
            "n_v_eor": 5,
            "n_u_fg": 5,
            "n_v_fg": 5,
            "n_freq": 3,
            "n_eta": 5,
            "n_modes_fg": 2,
            "n_powerlaws_fg": 2,
            "nside": 16,
            "fov_eor": 10.0,
            "fov_fg": 10.0,
            "n_t": 2,
            "fit_for_monopole": True,
        },
        {
            "name": "cfg06_more_channels",
            "n_u_eor": 5,
            "n_v_eor": 5,
            "n_u_fg": 5,
            "n_v_fg": 5,
            "n_freq": 4,
            "n_eta": 7,
            "n_modes_fg": 2,
            "n_powerlaws_fg": 2,
            "nside": 32,
            "fov_eor": 10.0,
            "fov_fg": 10.0,
            "n_t": 2,
            "fit_for_monopole": True,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 6 small old-vs-new pipeline regression cases.")
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA_PATH, help=f"UVData path (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--old-repo", type=Path, default=DEFAULT_OLD_REPO, help=f"Old BayesEoR repo path (default: {DEFAULT_OLD_REPO})")
    parser.add_argument("--out-root", type=Path, default=Path("pipeline_regression_runs"), help="Output root for per-case matrices")
    parser.add_argument("--keep", action="store_true", help="Keep existing output root instead of deleting first")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    if not args.keep and args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    uvd = _load_uvdata(args.path)
    configs = _configs()

    print("Running regression suite with 6 odd u/v configurations")
    print(f"data_path={args.path}")
    print(f"old_repo={args.old_repo}")
    print(f"out_root={args.out_root}")

    failures = []

    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== {name} ===")
        times = _select_times(uvd, cfg["n_t"])
        uvw_array_m = _uvw_by_time(uvd, times)
        freqs_mhz = _freqs_mhz(uvd, cfg["n_freq"])
        f0_mhz = float(freqs_mhz[0])
        df_mhz = float(freqs_mhz[1] - freqs_mhz[0]) if cfg["n_freq"] > 1 else 1.0
        d_t = (times[1] - times[0]) * SECS_PER_DAY if cfg["n_t"] > 1 else 1.0

        new_dir = args.out_root / "new" / name
        old_dir = args.out_root / "old" / name

        _build_new(repo_root, cfg, times, uvw_array_m, f0_mhz, df_mhz, d_t, new_dir)
        _build_old(args.old_repo, cfg, times, uvw_array_m, f0_mhz, df_mhz, d_t, old_dir)

        case_ok = True
        for matrix_name in ["Fz", "Finv_Fprime", "T", "Ninv"]:
            old_arr = _load_old_matrix(old_dir, matrix_name)
            new_arr = _load_new_npz(new_dir / f"{matrix_name}.npz")
            ok, max_abs = _compare_matrix_pair(old_arr, new_arr)
            status = "PASS" if ok else "FAIL"
            print(f"  {matrix_name:12s} {status}  max_abs_diff={max_abs:.3e}")
            if not ok:
                case_ok = False

        if not case_ok:
            failures.append(name)

    print("\n=== Summary ===")
    if failures:
        print(f"FAILED cases ({len(failures)}):")
        for name in failures:
            print(f"  - {name}")
        raise SystemExit(1)
    print("All 6 cases passed.")


if __name__ == "__main__":
    main()
