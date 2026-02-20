#!/usr/bin/env python3
"""Parity checks between legacy BayesEoR k-cube utilities and JAX-native JOKER outputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

OLD_FILE = Path("/Users/jacobtutt/Desktop/PhD/BayesEoR/bayeseor/model/k_cube.py")
NEW_FILE = Path("/Users/jacobtutt/Desktop/PhD/JOKER/BayesEoR-JAX/k_cube.py")


def _load_module(file_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_legacy_kcube_without_mpi(file_path: Path):
    source = file_path.read_text()
    source = source.replace(
        "from ..utils import mpiprint",
        "def mpiprint(*args, **kwargs):\n    return None",
    )
    module_dict: dict = {"__name__": "legacy_kcube_standalone"}
    exec(compile(source, str(file_path), "exec"), module_dict)

    class ModuleProxy:
        pass

    proxy = ModuleProxy()
    for key, value in module_dict.items():
        if not key.startswith("__"):
            setattr(proxy, key, value)
    return proxy


def _old_membership_to_bin_id(n_k_modes: int, k_mode_inds_in_bin):
    bid = np.full((n_k_modes,), -1, dtype=np.int32)
    for i_bin, where_tuple in enumerate(k_mode_inds_in_bin):
        inds = where_tuple[0]
        bid[inds] = i_bin
    return bid


def _run_case(old_kcube, new_kcube, cfg: dict):
    n_u = cfg["n_u"]
    n_v = cfg["n_v"]
    n_eta = cfg["n_eta"]
    l_ra = cfg["l_ra"]
    l_dec = cfg["l_dec"]
    l_para = cfg["l_para"]
    n_k_perp_bins = cfg["n_k_perp_bins"]

    old_mod_k, old_kx, old_ky, old_kz, *_ = old_kcube.generate_k_cube_in_physical_coordinates(
        n_u, n_v, n_eta, l_ra, l_dec, l_para
    )
    new_mod_k, new_kx, new_ky, new_kz, *_ = new_kcube.generate_k_cube_in_physical_coordinates(
        n_u, n_v, n_eta, l_ra, l_dec, l_para
    )

    assert np.allclose(old_mod_k, np.asarray(new_mod_k), rtol=1e-12, atol=1e-12)
    assert np.allclose(old_kx, np.asarray(new_kx), rtol=1e-12, atol=1e-12)
    assert np.allclose(old_ky, np.asarray(new_ky), rtol=1e-12, atol=1e-12)
    assert np.allclose(old_kz, np.asarray(new_kz), rtol=1e-12, atol=1e-12)

    old_mod_k_vo = old_kcube.mask_k_cube(old_mod_k)
    new_mod_k_vo = np.asarray(new_kcube.mask_k_cube(new_mod_k))
    assert np.allclose(old_mod_k_vo, new_mod_k_vo, rtol=1e-12, atol=1e-12)

    # Spherical binning parity.
    old_k_modes, old_meta = old_kcube.generate_k_cube_model_spherical_binning(old_mod_k_vo, l_para)
    old_bid = _old_membership_to_bin_id(old_mod_k_vo.size, old_k_modes)
    old_valid = old_bid >= 0
    old_counts = np.array([count for _, count in old_meta], dtype=np.int32)
    old_k_vals = old_kcube.calc_mean_binned_k_vals(old_mod_k_vo, old_k_modes)

    new_bid, new_valid, _new_low, _new_high, new_counts_full = new_kcube.generate_k_cube_model_spherical_binning(
        new_mod_k_vo, l_para
    )
    new_bid = np.asarray(new_bid)
    new_valid = np.asarray(new_valid)
    new_counts_full = np.asarray(new_counts_full)

    new_nonempty_bins = np.where(new_counts_full > 0)[0]
    remap = np.full_like(new_counts_full, -1, dtype=np.int32)
    remap[new_nonempty_bins] = np.arange(new_nonempty_bins.size, dtype=np.int32)
    new_bid_compact = np.where(new_valid, remap[new_bid], -1)
    new_counts = new_counts_full[new_nonempty_bins]
    new_k_vals_full = np.asarray(
        new_kcube.calc_mean_binned_k_vals(new_mod_k_vo, new_bid, new_valid, new_counts_full.size)
    )
    new_k_vals = new_k_vals_full[new_nonempty_bins]

    assert np.array_equal(old_valid, new_valid)
    assert np.array_equal(old_bid, new_bid_compact)
    assert np.array_equal(old_counts, new_counts)
    assert np.allclose(old_k_vals, new_k_vals, rtol=1e-12, atol=1e-12)

    # Cylindrical binning parity.
    old_kz_masked = old_kcube.mask_k_cube(old_kz)
    old_ky_masked = old_kcube.mask_k_cube(old_ky)
    old_kx_masked = old_kcube.mask_k_cube(old_kx)
    old_c_vox, old_c_meta, old_kperp = old_kcube.generate_k_cube_model_cylindrical_binning(
        old_mod_k_vo, old_kz_masked, old_ky_masked, old_kx_masked, n_k_perp_bins, l_para
    )
    old_c_bid = _old_membership_to_bin_id(old_mod_k_vo.size, old_c_vox)
    old_c_valid = old_c_bid >= 0
    old_kpar_nonempty = len(old_c_meta)
    old_kperp_n = len(old_kperp[0]) if old_kpar_nonempty else 0
    old_counts_2d = np.zeros((old_kpar_nonempty, old_kperp_n), dtype=np.int32)
    for flat_id in old_c_bid[old_c_valid]:
        i = flat_id // old_kperp_n
        j = flat_id % old_kperp_n
        old_counts_2d[i, j] += 1

    new_kz_masked = np.asarray(new_kcube.mask_k_cube(new_kz))
    new_ky_masked = np.asarray(new_kcube.mask_k_cube(new_ky))
    new_kx_masked = np.asarray(new_kcube.mask_k_cube(new_kx))
    new_c_bid, new_c_valid, kp_lo, kp_hi, _kt_lo, _kt_hi, new_counts_2d_full = (
        new_kcube.generate_k_cube_model_cylindrical_binning(
            new_mod_k_vo, new_kz_masked, new_ky_masked, new_kx_masked, n_k_perp_bins, l_para
        )
    )
    new_c_bid = np.asarray(new_c_bid)
    new_c_valid = np.asarray(new_c_valid)
    new_counts_2d_full = np.asarray(new_counts_2d_full)

    # Match legacy row-selection rule:
    # rows are kept if mod_k bin has k-modes with k_z > 0 (before |k_z| split).
    row_keep = []
    for i in range(len(kp_lo)):
        n_elements = np.sum(
            (new_mod_k_vo > kp_lo[i]) & (new_mod_k_vo <= kp_hi[i]) & (new_kz_masked > 0.0)
        )
        if n_elements > 0:
            row_keep.append(i)
    row_keep = np.asarray(row_keep, dtype=np.int32)
    col_keep = np.arange(old_kperp_n, dtype=np.int32)
    new_counts_2d = new_counts_2d_full[row_keep][:, col_keep]

    # Remap new flat ids into compact legacy-style flat ids.
    n_rows, n_cols = new_counts_2d_full.shape
    row_remap = np.full((n_rows,), -1, dtype=np.int32)
    col_remap = np.full((n_cols,), -1, dtype=np.int32)
    row_remap[row_keep] = np.arange(row_keep.size, dtype=np.int32)
    col_remap[col_keep] = np.arange(col_keep.size, dtype=np.int32)

    new_c_bid_compact = np.full_like(new_c_bid, -1)
    valid_inds = np.where(new_c_valid)[0]
    if valid_inds.size > 0:
        r = new_c_bid[valid_inds] // n_cols
        c = new_c_bid[valid_inds] % n_cols
        rr = row_remap[r]
        cc = col_remap[c]
        keep = (rr >= 0) & (cc >= 0)
        new_c_bid_compact[valid_inds[keep]] = rr[keep] * col_keep.size + cc[keep]

    assert np.array_equal(old_c_valid, new_c_valid)
    assert np.array_equal(old_c_bid, new_c_bid_compact)
    assert np.array_equal(old_counts_2d, new_counts_2d)


def main():
    old_kcube = _load_legacy_kcube_without_mpi(OLD_FILE)
    new_kcube = _load_module(NEW_FILE, "new_kcube")

    cases = [
        {"n_u": 9, "n_v": 7, "n_eta": 8, "l_ra": 450.0, "l_dec": 390.0, "l_para": 520.0, "n_k_perp_bins": 8},
        {"n_u": 11, "n_v": 9, "n_eta": 9, "l_ra": 430.0, "l_dec": 410.0, "l_para": 505.0, "n_k_perp_bins": 9},
    ]
    for case in cases:
        _run_case(old_kcube, new_kcube, case)
        print(f"PASS parity case: {case}")

    # Even-grid support smoke check (legacy grid constructor is odd-only).
    even_case = {"n_u": 8, "n_v": 6, "n_eta": 9, "l_ra": 420.0, "l_dec": 395.0, "l_para": 500.0}
    mod_k, *_ = new_kcube.generate_k_cube_in_physical_coordinates(
        even_case["n_u"], even_case["n_v"], even_case["n_eta"], even_case["l_ra"], even_case["l_dec"], even_case["l_para"]
    )
    mod_k_vo = np.asarray(new_kcube.mask_k_cube(mod_k))
    expected_len = (even_case["n_eta"] - 1) * (even_case["n_u"] * even_case["n_v"] - 1)
    if mod_k_vo.size != expected_len:
        raise AssertionError(f"even-grid masked length mismatch {mod_k_vo.size} != {expected_len}")
    print(f"PASS even-grid smoke case: {even_case}")

    print("All k_cube parity tests passed.")


if __name__ == "__main__":
    main()
