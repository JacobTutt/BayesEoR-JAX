"""Check CPU numerics and optional GPU dispatch without a CUDA installation."""

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from bayeseor_jax.inference.linear_solver import build_gaussian_solver


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_single_device_solve_and_logdet_without_jaxmg(monkeypatch, platform):
    monkeypatch.setitem(sys.modules, "jaxmg", None)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("x",))
    if platform == "gpu":
        # Exercise GPU dispatch on CPU-only CI without importing JAXMG.
        mesh = SimpleNamespace(devices=np.array([SimpleNamespace(platform="gpu")]), size=1)
    solve = jax.jit(build_gaussian_solver(mesh))
    matrix = jnp.array([[3., 1j], [-1j, 2.]], dtype=jnp.complex128)
    rhs = jnp.array([1. + 2j, 3. - 1j])
    solution, logdet = solve(matrix, rhs)
    np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), atol=1e-12)
    np.testing.assert_allclose(logdet, np.linalg.slogdet(matrix)[1], atol=1e-12)


def test_multi_gpu_dispatch_preserves_jaxmg_arguments(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "jaxmg", SimpleNamespace(
        potrs=lambda *args, **kwargs: calls.append((args, kwargs)) or ("solution", "logdet"),
    ))
    mesh = SimpleNamespace(devices=np.array([SimpleNamespace(platform="gpu")] * 2), size=2)
    assert build_gaussian_solver(mesh, 128)("matrix", "rhs") == ("solution", "logdet")
    assert calls == [(("matrix", "rhs"), dict(
        T_A=128, mesh=mesh, in_specs=(P("x", None),), return_logdet=True,
    ))]


def test_missing_multi_gpu_dependency_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "jaxmg", None)
    mesh = SimpleNamespace(devices=np.array([SimpleNamespace(platform="gpu")] * 2), size=2)
    with pytest.raises(ImportError, match="HPC installation"):
        build_gaussian_solver(mesh)


def test_multiple_cpu_devices_are_not_silently_gathered():
    mesh = SimpleNamespace(devices=np.array([SimpleNamespace(platform="cpu")]), size=2)
    with pytest.raises(ValueError, match="single-device"):
        build_gaussian_solver(mesh)


@pytest.mark.parametrize("platforms", [("tpu",), ("cpu", "gpu")])
def test_unsupported_solver_mesh_is_rejected(platforms):
    mesh = SimpleNamespace(
        devices=np.array([SimpleNamespace(platform=p) for p in platforms]),
        size=len(platforms),
    )
    with pytest.raises(ValueError, match="CPU device or a GPU solver mesh"):
        build_gaussian_solver(mesh)
