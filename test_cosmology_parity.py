#!/usr/bin/env python3
"""Parity checks for JOKER JAX cosmology transforms against legacy BayesEoR cosmology."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import jax.numpy as jnp

OLD_FILE = Path("/Users/jacobtutt/Desktop/PhD/BayesEoR/bayeseor/cosmology.py")
NEW_FILE = Path("/Users/jacobtutt/Desktop/PhD/JOKER/BayesEoR-JAX/model/cosmology_transforms.py")


def _load_module(file_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _err_stats(a: np.ndarray, b: np.ndarray):
    abs_err = np.max(np.abs(a - b)) if a.size else 0.0
    denom = np.maximum(np.abs(a), 1e-15)
    rel_err = np.max(np.abs(a - b) / denom) if a.size else 0.0
    return float(abs_err), float(rel_err)


def main() -> int:
    old_mod = _load_module(OLD_FILE, "old_cosmo")
    new_mod = _load_module(NEW_FILE, "new_cosmo")

    old = old_mod.Cosmology()
    new = new_mod.Cosmology()

    # Frequency and redshift ranges representative of EoR analyses.
    freqs_hz = np.linspace(100e6, 220e6, 257, dtype=np.float64)
    z_grid = np.linspace(5.0, 20.0, 301, dtype=np.float64)

    # f2z parity
    old_f2z = np.array([old.f2z(float(f)) for f in freqs_hz], dtype=np.float64)
    new_f2z = np.asarray(new.f2z(jnp.asarray(freqs_hz, dtype=jnp.float64)), dtype=np.float64)
    f2z_abs, f2z_rel = _err_stats(old_f2z, new_f2z)

    # z2f parity
    old_z2f = np.array([old.z2f(float(z)) for z in z_grid], dtype=np.float64)
    new_z2f = np.asarray(new.z2f(jnp.asarray(z_grid, dtype=jnp.float64)), dtype=np.float64)
    z2f_abs, z2f_rel = _err_stats(old_z2f, new_z2f)

    # dL_df parity
    old_dldf = np.array([old.dL_df(float(z)) for z in z_grid], dtype=np.float64)
    new_dldf = np.asarray(new.dL_df(jnp.asarray(z_grid, dtype=jnp.float64)), dtype=np.float64)
    dldf_abs, dldf_rel = _err_stats(old_dldf, new_dldf)

    # dL_dth parity (new implementation uses numerical integration; expect small but non-zero differences)
    old_dldth = np.array([old.dL_dth(float(z)) for z in z_grid], dtype=np.float64)
    new_dldth = np.asarray(new.dL_dth(jnp.asarray(z_grid, dtype=jnp.float64), n_z_samples=4096), dtype=np.float64)
    dldth_abs, dldth_rel = _err_stats(old_dldth, new_dldth)

    # inst_to_cosmo_vol parity
    old_i2v = np.array([old.inst_to_cosmo_vol(float(z)) for z in z_grid], dtype=np.float64)
    new_i2v = np.asarray(new.inst_to_cosmo_vol(jnp.asarray(z_grid, dtype=jnp.float64), n_z_samples=4096), dtype=np.float64)
    i2v_abs, i2v_rel = _err_stats(old_i2v, new_i2v)

    print("Cosmology parity summary:")
    print(f"  f2z               max_abs={f2z_abs:.3e} max_rel={f2z_rel:.3e}")
    print(f"  z2f               max_abs={z2f_abs:.3e} max_rel={z2f_rel:.3e}")
    print(f"  dL_df             max_abs={dldf_abs:.3e} max_rel={dldf_rel:.3e}")
    print(f"  dL_dth            max_abs={dldth_abs:.3e} max_rel={dldth_rel:.3e}")
    print(f"  inst_to_cosmo_vol max_abs={i2v_abs:.3e} max_rel={i2v_rel:.3e}")

    # Thresholds: exact or near-exact for algebraic transforms, modest tolerance for integrated distance.
    failures = []
    if not np.allclose(old_f2z, new_f2z, rtol=1e-12, atol=1e-12):
        failures.append("f2z")
    if not np.allclose(old_z2f, new_z2f, rtol=1e-12, atol=1e-12):
        failures.append("z2f")
    if not np.allclose(old_dldf, new_dldf, rtol=1e-11, atol=1e-13):
        failures.append("dL_df")
    if not np.allclose(old_dldth, new_dldth, rtol=5e-5, atol=5e-3):
        failures.append("dL_dth")
    if not np.allclose(old_i2v, new_i2v, rtol=1e-4, atol=5e-2):
        failures.append("inst_to_cosmo_vol")

    if failures:
        print("FAIL parity for:", ", ".join(failures))
        return 1

    print("PASS all cosmology parity checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
