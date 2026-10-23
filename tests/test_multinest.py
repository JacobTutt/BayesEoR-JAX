"""Check the MultiNest-only workflow without invoking a native sampler."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from bayeseor_jax import DataContainer
from bayeseor_jax.inference.phi import PhiInverse
from bayeseor_jax.inference.prior import Prior


def test_multinest_runner_needs_no_removed_dependencies(tmp_path, monkeypatch):
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        theta = kwargs["Prior"](np.array([.25, .75]))
        np.testing.assert_allclose(theta, [-1., 1.])
        assert kwargs["LogLikelihood"](theta) == -2.
        return {"logZ": -3., "logZerr": .1, "samples": theta[None, :]}

    for dependency in ("blackjax", "anesthetic", "tqdm"):
        monkeypatch.setitem(sys.modules, dependency, None)
    monkeypatch.setitem(sys.modules, "pymultinest.solve", SimpleNamespace(solve=solve))
    monkeypatch.setitem(sys.modules, "bayeseor_jax.setup", SimpleNamespace(SetupResult=object))
    spec = importlib.util.spec_from_file_location(
        "bayeseor_jax.inference._multinest_under_test",
        Path(__file__).parents[1] / "bayeseor_jax/inference/run_ns.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.run_nestedsampling is runner.run_multinest
    setup = SimpleNamespace(
        model="eor+fg",
        prior_class=SimpleNamespace(
            bounds_log=np.array([[-2., 2.], [-2., 2.]]),
            lower=np.array([-2., -2.]), upper=np.array([2., 2.]),
            transform_unit_cube=lambda cube: -2. + 4. * cube,
        ),
        likelihood_class=SimpleNamespace(
            intrinsic_noise_fitting=False,
            log_likelihood=lambda theta: -np.sum(theta**2),
        ),
    )
    summary = runner.run_nestedsampling(setup, tmp_path, random_key=7, print_rate=0)
    assert len(calls) == 1
    assert calls[0]["n_dims"] == 2
    assert calls[0]["n_live_points"] == 50
    assert calls[0]["seed"] == 7
    assert summary["sampler"] == "multinest"
    assert summary["loglikelihood_calls"] == 1
    assert summary["logZ"] == -3.
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_fg_only_runner_saves_analytic_evidence_and_map_without_multinest(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "jaxmg", None)
    monkeypatch.setitem(sys.modules, "pymultinest", None)
    monkeypatch.setitem(sys.modules, "pymultinest.solve", None)
    root = Path(__file__).parents[1] / "bayeseor_jax/inference"
    modules = {}
    for name in ("maxaposteriori", "run_ns"):
        spec = importlib.util.spec_from_file_location(f"bayeseor_jax.inference._test_{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    monkeypatch.setitem(sys.modules, "bayeseor_jax.inference.maxaposteriori", modules["maxaposteriori"])
    gram = jnp.array([[2., .3], [.3, 1.]], dtype=jnp.complex128)
    dbar = jnp.array([1., 2.], dtype=jnp.complex128)
    phi = PhiInverse(
        bin_id_per_k_mode=jnp.empty(0, dtype=int), valid_mask=jnp.empty(0, dtype=bool),
        k_vals=jnp.empty(0), ps_box_size_ra_mpc=1., ps_box_size_dec_mpc=1.,
        ps_box_size_para_mpc=1., inst_to_cosmo_vol=1., n_par=2, foreground_prior_std=[1., 2.],
    )
    setup = SimpleNamespace(
        model="fg", prior_class=Prior(jnp.empty((0, 2))), phi_inverse_class=phi,
        likelihood_class=SimpleNamespace(
            intrinsic_noise_fitting=False, t_ninv_t=gram, dbar=dbar, d_ninv_d=4., n_data=4,
            mesh=Mesh(np.asarray(jax.devices()[:1]), ('x',)), jaxmg_tile_size=256,
        ),
    )
    summary = modules["run_ns"].run_nestedsampling(setup, tmp_path)
    sigma = np.asarray(gram) + np.diag([1., .25])
    expected_map = np.linalg.solve(sigma, np.asarray(dbar))
    expected_logz = .5 * (np.vdot(dbar, expected_map).real - np.linalg.slogdet(sigma)[1] - np.log(4.))
    np.testing.assert_allclose(summary["logZ"], expected_logz)
    np.testing.assert_allclose(np.load(tmp_path / "map_coefficients.npy"), expected_map)
    assert summary["sampler"] == "analytic"
    assert summary["n_dim"] == 0
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with pytest.raises(ValueError, match="no sampler checkpoint"):
        modules["run_ns"].run_nestedsampling(setup, tmp_path, resume=True)


@pytest.mark.parametrize("file_root", ["data-", "short-"])
def test_multinest_analysis_preserves_weights_and_log_power(tmp_path, file_root):
    samples = np.array([[.2, 8., 0., 1.], [.3, 6., .5, 1.5], [.5, 4., 1., 2.]])
    np.savetxt(tmp_path / f"{file_root}.txt", samples)
    np.savetxt(tmp_path / "k-vals.txt", [.1, .2])
    result = DataContainer([tmp_path])
    np.testing.assert_allclose(result.k_vals[0], [.1, .2])
    np.testing.assert_allclose(result.samples[0], samples[:, 2:])
    expected = np.average(10**samples[:, 2:], axis=0, weights=samples[:, 0])
    np.testing.assert_allclose(result.avgs[0], expected)
    np.testing.assert_allclose(result.weights[0], samples[:, 0])
    np.testing.assert_allclose(result.log_likelihoods[0], -.5 * samples[:, 1])


@pytest.mark.parametrize("offset", [0., 7e11])
def test_centred_sampler_restores_evidence_and_records_chain_offset(tmp_path, monkeypatch, offset):
    from bayeseor_jax.inference.run_ns import run_multinest
    import inspect

    assert 'center_likelihood' not in inspect.signature(run_multinest).parameters
    def solve(**kwargs):
        assert kwargs['LogLikelihood'](np.array([1.])) == -.5
        assert 'center_likelihood' not in kwargs
        return {'logZ':-2., 'logZerr':.1, 'samples':np.array([[1.]])}
    monkeypatch.setitem(sys.modules, 'pymultinest.solve', SimpleNamespace(solve=solve))
    setup = SimpleNamespace(model='eor+fg', prior_class=Prior(np.array([[-2.,2.]])),
        likelihood_class=SimpleNamespace(intrinsic_noise_fitting=False,
                                         log_likelihood=lambda theta:offset-.5*np.sum(theta**2)))
    summary = run_multinest(setup, tmp_path, print_rate=0)
    assert summary['logZ'] == offset-2.
    assert summary['logZ_centered'] == -2.
    assert summary['log_likelihood_offset'] == offset
    assert summary['logZerr'] == .1
    np.savetxt(tmp_path/'data-.txt', [[1.,1.,1.]])
    np.savetxt(tmp_path/'k-vals.txt', [.1])
    data = DataContainer([tmp_path])
    np.testing.assert_array_equal(data.log_likelihoods[0], [offset-.5])
    (tmp_path/'data-resume.dat').touch()
    resumed = run_multinest(setup, tmp_path, resume=True, print_rate=0)
    assert resumed['log_likelihood_offset'] == offset
    def resumed_likelihood(theta):
        assert not np.all(theta == 0), 'A resumed run must not recompute its reference offset'
        return offset-.5*np.sum(theta**2)
    setup.likelihood_class.log_likelihood = resumed_likelihood
    default_resume = run_multinest(setup, tmp_path, print_rate=0)
    assert default_resume['log_likelihood_offset'] == offset
    (tmp_path/'likelihood_offset.json').unlink()
    with pytest.raises(FileNotFoundError, match='likelihood_offset.json'):
        run_multinest(setup, tmp_path, resume=True, print_rate=0)


def test_centred_sampler_rejects_nonfinite_reference(tmp_path, monkeypatch):
    from bayeseor_jax.inference.run_ns import run_multinest
    monkeypatch.setitem(sys.modules, 'pymultinest.solve', SimpleNamespace(solve=lambda **kwargs:pytest.fail('must not sample')))
    setup = SimpleNamespace(model='eor', prior_class=Prior(np.array([[-2.,2.]])),
        likelihood_class=SimpleNamespace(intrinsic_noise_fitting=False, log_likelihood=lambda theta:np.nan))
    with pytest.raises(ValueError, match='offset must be finite'):
        run_multinest(setup, tmp_path)
