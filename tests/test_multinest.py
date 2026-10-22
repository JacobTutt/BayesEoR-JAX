"""Check the MultiNest-only workflow without invoking a native sampler."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bayeseor_jax import DataContainer


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
        "bayeseor_jax._multinest_under_test",
        Path(__file__).parents[1] / "bayeseor_jax/run_ns.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.run_nestedsampling is runner.run_multinest
    setup = SimpleNamespace(
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


@pytest.mark.parametrize("file_root", ["data-", "short-"])
@pytest.mark.parametrize("weighted", [True, False])
def test_multinest_analysis_preserves_weights_and_log_power(tmp_path, file_root, weighted):
    samples = np.array([[.2, 8., 0., 1.], [.3, 6., .5, 1.5], [.5, 4., 1., 2.]])
    np.savetxt(tmp_path / f"{file_root}.txt", samples)
    result = DataContainer([tmp_path], store_samples=True, posterior_weighted=weighted)
    np.testing.assert_allclose(result.k_vals[0], [1., 2.])
    np.testing.assert_allclose(result.samples[0], samples[:, 2:])
    expected = np.average(10**samples[:, 2:], axis=0, weights=samples[:, 0] if weighted else None)
    np.testing.assert_allclose(result.avgs[0], expected)
    assert result.posteriors[0].shape == (2, 31)
