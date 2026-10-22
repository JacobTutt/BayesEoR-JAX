"""Check KL truncation and shard publication independently of the GPU solver."""

import json
import math
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bayeseor_jax.model.construction import kl_reduction as svd
from bayeseor_jax.model.operators import weight_foreground_operator


@pytest.fixture
def reference_solver(monkeypatch):
    """Substitute only the native SVD and GPU check; exercise real shard I/O."""
    devices = np.asarray(jax.devices())
    rows = math.isqrt(devices.size)
    while devices.size % rows:
        rows -= 1
    mesh = Mesh(devices.reshape(rows, -1), ("svd_rows", "svd_cols"))
    calls = []

    def gesvd(a, **kwargs):
        assert kwargs["compute_u"] and not kwargs["compute_vh"]
        assert kwargs["return_status"] and kwargs["donate"]
        assert not kwargs["full_matrices"]
        assert kwargs["matrix_specs"] == P("svd_rows", "svd_cols")
        assert a.sharding == NamedSharding(mesh, kwargs["matrix_specs"])
        host = np.asarray(a)
        calls.append(host.copy())
        u, values, _ = np.linalg.svd(host, full_matrices=False)
        return (jax.device_put(u, a.sharding),
                jax.device_put(values, NamedSharding(mesh, P())),
                jnp.zeros(devices.size, dtype=jnp.int32))

    monkeypatch.setattr(svd, "_svd_mesh", lambda: mesh)
    monkeypatch.setitem(sys.modules, "jaxmg", SimpleNamespace(gesvd=gesvd))
    return calls


@pytest.mark.parametrize("shape", [(7, 5), (5, 9), (3, 2)])
@pytest.mark.parametrize("retention", ["partial", "full", "none"])
def test_reduce_weighted_operator_and_restore_physical_noise(tmp_path, reference_solver, shape, retention):
    rng = np.random.default_rng(5)
    physical = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    prior_std = np.linspace(.3, 2., shape[1])
    inverse_noise = np.linspace(.5, 3., shape[0])
    weighted = np.asarray(weight_foreground_operator(physical, prior_std, np.sqrt(inverse_noise)))
    u, values, _ = np.linalg.svd(weighted, full_matrices=False)
    _, tails = svd.select_foreground_rank(values, shape[0])
    if retention == "partial":
        expected_rank = 1
        f = (tails[0]+tails[1])/2
    elif retention == "none":
        expected_rank, f = 0, tails[0]*1.01
    else:
        expected_rank, f = min(shape), 0.
    a_path, n_path, out_path = tmp_path / "A.npy", tmp_path / "Ninv.npy", tmp_path / "reduced.npy"
    np.save(a_path, weighted)
    np.save(n_path, inverse_noise)
    result = svd.T_fg_svd_build_save(a_path, n_path, output_path=out_path, f=f, tile_size=2)
    reduced = np.load(result)
    assert reduced.shape == (shape[0], expected_rank)
    expected = u[:, :expected_rank] * values[:expected_rank] / np.sqrt(inverse_noise[:, None])
    # Phases of singular vectors are arbitrary; their induced covariance is not.
    np.testing.assert_allclose(reduced @ reduced.conj().T, expected @ expected.conj().T, atol=1e-11)
    np.testing.assert_array_equal(np.load(a_path), weighted)
    np.testing.assert_array_equal(reference_solver[0][:shape[0], :shape[1]], weighted)
    assert np.count_nonzero(reference_solver[0][shape[0]:]) == 0
    assert np.count_nonzero(reference_solver[0][:, shape[1]:]) == 0
    metadata = json.loads((tmp_path / "reduced.svd.json").read_text())
    assert metadata["retained_rank"] == expected_rank
    assert metadata["original_shape"] == list(shape)
    assert metadata["coefficient_prior"] == "standard-normal"
    assert metadata["discarded_rms_over_noise_rms"] <= f
    saved_values = np.load(tmp_path / "reduced.singular_values.npy")
    np.testing.assert_allclose(saved_values, values, rtol=1e-12, atol=1e-12)
    assert not list(tmp_path.glob("*.partial.*"))
    assert not list(tmp_path.glob("*.ready"))


def test_rank_uses_original_visibility_count_and_small_tail():
    values = [1e12, 3., 1., .1]
    rank, tails = svd.select_foreground_rank(values, 10, f=.04)
    assert rank == 3
    np.testing.assert_allclose(tails[1:], [np.sqrt(10.01/10), np.sqrt(1.01/10), np.sqrt(.01/10), 0])
    assert svd.select_foreground_rank([0., 0.], 10, f=0)[0] == 0
    assert svd.select_foreground_rank([3., 1.], 4, f=.5)[0] == 1
    assert svd.select_foreground_rank([1e200], 1, f=1)[0] == 1


@pytest.mark.parametrize("values,n_vis,f", [([1.], 0, .1), ([1.], 2, -.1),
                                           ([1.], 2, np.nan), ([1.], 2, np.inf),
                                           ([np.nan], 2, .1), ([-1.], 2, .1),
                                           ([1., 2.], 2, .1), ([1j], 2, .1)])
def test_invalid_rank_inputs(values, n_vis, f):
    with pytest.raises(ValueError):
        svd.select_foreground_rank(values, n_vis, f)


@pytest.mark.parametrize("shape,grid", [((7, 19), (2, 3)), ((19, 7), (2, 3)), ((5, 2), (1, 4))])
def test_padding_also_makes_thin_u_legal(shape, grid):
    padded = svd._svd_padded_shape(shape, grid, 2)
    assert all(size >= original for size, original in zip(padded, shape))
    assert padded[0] % (grid[0]*2) == padded[1] % (grid[1]*2) == 0
    assert min(padded) % (grid[1]*2) == 0


@pytest.mark.parametrize("status", [-1, 1])
def test_solver_failure_preserves_completed_operator(tmp_path, reference_solver, monkeypatch, status):
    np.save(tmp_path / "A.npy", np.eye(3, dtype=np.complex128))
    np.save(tmp_path / "Ninv.npy", np.ones(3))
    out_path = tmp_path / "reduced.npy"
    np.save(out_path, np.array([[42.]]))

    def failed(*args, **kwargs):
        # A negative status must not be hidden by a zero status on another rank.
        return None, None, jnp.array([status, 0])

    monkeypatch.setitem(sys.modules, "jaxmg", SimpleNamespace(gesvd=failed))
    with pytest.raises(RuntimeError, match="GESVD failed"):
        svd.T_fg_svd_build_save(tmp_path / "A.npy", tmp_path / "Ninv.npy", output_path=out_path)
    np.testing.assert_array_equal(np.load(out_path), [[42.]])


def test_invalid_inputs_are_rejected_before_solver(tmp_path, reference_solver):
    a_path, n_path = tmp_path / "A.npy", tmp_path / "Ninv.npy"
    np.save(a_path, np.eye(3, dtype=np.complex128))
    np.save(n_path, [1., 0., 1.])
    with pytest.raises(ValueError, match="Inverse noise"):
        svd.T_fg_svd_build_save(a_path, n_path, output_path=tmp_path / "out.npy")
    np.save(n_path, np.ones(3))
    for out_path in (a_path, n_path):
        with pytest.raises(ValueError, match="overwrite"):
            svd.T_fg_svd_build_save(a_path, n_path, output_path=out_path)
    # Sidecar names must not collide with the weighted input, either.
    sidecar = tmp_path / "out.singular_values.npy"
    np.save(sidecar, np.eye(3, dtype=np.complex128))
    with pytest.raises(ValueError, match="overwrite"):
        svd.T_fg_svd_build_save(sidecar, n_path, output_path=tmp_path / "out.npy")
    np.save(a_path, np.full((3, 2), np.nan, dtype=np.complex128))
    with pytest.raises(RuntimeError, match="nonfinite"):
        svd.T_fg_svd_build_save(a_path, n_path, output_path=tmp_path / "out.npy", tile_size=2)
    assert not reference_solver


def test_no_implicit_cpu_fallback():
    if jax.default_backend() == "cpu":
        with pytest.raises(RuntimeError, match="GPU backend"):
            svd._svd_mesh()
