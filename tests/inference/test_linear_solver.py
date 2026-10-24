"""Tests for ``bayeseor_jax.inference.linear_solver``."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from bayeseor_jax.inference.linear_solver import build_gaussian_solver


def _hermitian_pd(rng, n, complex_valued):
    """Random well-conditioned Hermitian positive-definite matrix."""
    a = rng.normal(size=(n, n))
    if complex_valued:
        a = a + 1j * rng.normal(size=(n, n))
    return a @ a.conj().T + n * np.eye(n)


def _fake_mesh(platforms, size):
    """Mesh-like object with the given device platforms and size."""
    devices = np.array([SimpleNamespace(platform=p) for p in platforms], dtype=object)
    return SimpleNamespace(devices=devices, size=size)


@pytest.fixture
def cpu_mesh() -> Mesh:
    """Single-device CPU mesh with the solver axis name."""
    return Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))


class TestBuildGaussianSolver:
    @pytest.mark.parametrize("complex_valued", [False, True])
    @pytest.mark.parametrize("n", [1, 5, 12])
    def test_solution_and_logdet_match_numpy(self, cpu_mesh, rng, n, complex_valued):
        """The single-device solve and log-determinant agree with NumPy."""
        matrix = _hermitian_pd(rng, n, complex_valued)
        rhs = rng.normal(size=n) + (1j * rng.normal(size=n) if complex_valued else 0.0)
        solution, logdet = build_gaussian_solver(cpu_mesh)(jnp.asarray(matrix), jnp.asarray(rhs))
        np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(logdet, np.linalg.slogdet(matrix)[1], rtol=1e-12)

    def test_logdet_is_real(self, cpu_mesh, rng):
        """The log-determinant of a complex Hermitian matrix is returned as a real number."""
        _, logdet = build_gaussian_solver(cpu_mesh)(jnp.asarray(_hermitian_pd(rng, 4, True)), jnp.ones(4, complex))
        assert not jnp.iscomplexobj(logdet)

    def test_usable_inside_jit(self, cpu_mesh, rng):
        """The returned solver can be traced inside ``jax.jit``."""
        solve = jax.jit(build_gaussian_solver(cpu_mesh))
        matrix = _hermitian_pd(rng, 3, False)
        np.testing.assert_allclose(solve(jnp.asarray(matrix), jnp.ones(3))[0], np.linalg.solve(matrix, np.ones(3)))

    def test_rejects_unsupported_platform(self):
        """Only CPU or GPU meshes are accepted."""
        with pytest.raises(ValueError, match="CPU device or a GPU"):
            build_gaussian_solver(_fake_mesh(["tpu"], 1))

    def test_rejects_mixed_platforms(self):
        """A mesh mixing CPU and GPU devices is rejected."""
        with pytest.raises(ValueError, match="CPU device or a GPU"):
            build_gaussian_solver(_fake_mesh(["cpu", "gpu"], 2))

    def test_rejects_multi_device_cpu_mesh(self):
        """CPU inference requires exactly one device."""
        with pytest.raises(ValueError, match="single-device"):
            build_gaussian_solver(_fake_mesh(["cpu", "cpu"], 2))

    def test_multi_gpu_requires_jaxmg(self, monkeypatch):
        """A multi-GPU mesh without JAXMg installed raises an informative ImportError."""
        import builtins

        real_import = builtins.__import__

        def no_jaxmg(name, *args, **kwargs):
            if name == "jaxmg":
                raise ImportError("no jaxmg")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_jaxmg)
        with pytest.raises(ImportError, match="JAXMG"):
            build_gaussian_solver(_fake_mesh(["gpu", "gpu"], 2))
