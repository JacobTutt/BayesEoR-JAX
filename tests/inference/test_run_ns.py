"""Tests for ``bayeseor_jax.inference.run_ns``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from bayeseor_jax.inference.likelihood_marg import MarginalisedLikelihood
from bayeseor_jax.inference.phi import PhiInverse
from bayeseor_jax.inference.prior import Prior
from bayeseor_jax.inference.run_ns import (
    _multinest_resume_checkpoints,
    estimate_average_eval_time,
    run_multinest,
    run_nestedsampling,
)

BOX = dict(ps_box_size_ra_mpc=1.0, ps_box_size_dec_mpc=1.0, ps_box_size_para_mpc=1.0)
FG_STD = np.array([0.8, 1.5])
BOUNDS = np.array([[-1.0, 3.0]])


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path, monkeypatch):
    """Run from ``tmp_path`` so any MultiNest short links are created there."""
    monkeypatch.chdir(tmp_path)


def _setup(rng, n_eor, intrinsic_noise_fitting=False):
    """SetupResult-like object for ``n_eor`` EoR voxels in one bin plus two foreground columns."""
    m, p = 10, n_eor + len(FG_STD)
    a = rng.normal(size=(m, p)) + 1j * rng.normal(size=(m, p))
    noise = np.tile(rng.uniform(0.5, 1.5, size=m), 2)
    v = a @ np.concatenate([rng.normal(scale=3.0, size=n_eor), rng.normal(size=len(FG_STD))])
    v = v + rng.normal(size=m) + 1j * rng.normal(size=m)
    t, d = np.vstack([a, a.conj()]), np.concatenate([v, v.conj()])
    phi = PhiInverse(
        bin_id_per_k_mode=np.zeros(n_eor, int), valid_mask=np.ones(n_eor, bool),
        k_vals=np.array([0.5]) if n_eor else np.empty(0), inst_to_cosmo_vol=30.0,
        n_par=p, foreground_prior_std=FG_STD, **BOX,
    )
    bounds = BOUNDS if n_eor else np.empty((0, 2))
    prior = Prior(bounds)
    likelihood = MarginalisedLikelihood(
        t_ninv_t=t.conj().T @ (t / noise[:, None]), dbar=t.conj().T @ (d / noise),
        d_ninv_d=float(np.real(np.vdot(d, d / noise))), n_data=2 * m,
        phi_inv_class=phi, prior_class=prior, n_power_params=len(bounds),
        intrinsic_noise_fitting=intrinsic_noise_fitting,
    )
    return SimpleNamespace(
        prior_class=prior, likelihood_class=likelihood, phi_inverse_class=phi,
        model="fg" if n_eor == 0 else "eor+fg",
    )


class TestEstimateAverageEvalTime:
    def test_returns_mean_time_and_calls_function(self):
        """The average is positive and the function is called ``avg_iters`` times."""
        calls = []
        seconds = estimate_average_eval_time(lambda x: calls.append(1) or 1.0, np.zeros(1), 4)
        assert len(calls) == 4 and seconds >= 0.0

    def test_rejects_non_finite_values(self):
        """A non-finite likelihood value raises during timing."""
        with pytest.raises(ValueError, match="Non-finite"):
            estimate_average_eval_time(lambda x: np.nan, np.zeros(1), 2)


class TestMultinestResumeCheckpoints:
    def test_candidate_names(self):
        """The three MultiNest checkpoint files are derived from the basename."""
        names = [p.as_posix() for p in _multinest_resume_checkpoints("out/data-")]
        assert names == ["out/data-resume.dat", "out/data-phys_live.points", "out/data-.resume"]


class TestRunMultinestAnalytic:
    def test_alias(self):
        """``run_nestedsampling`` is the same function as ``run_multinest``."""
        assert run_nestedsampling is run_multinest

    def test_writes_evidence_and_map(self, rng, tmp_path):
        """A parameter-free model saves its MAP coefficients and analytic evidence."""
        setup = _setup(rng, n_eor=0)
        summary = run_multinest(setup, tmp_path / "out")
        assert summary["sampler"] == "analytic" and summary["n_dim"] == 0 and summary["logZerr"] == 0.0
        assert json.loads((tmp_path / "out" / "summary.json").read_text()) == summary
        coeffs = np.load(tmp_path / "out" / "map_coefficients.npy")
        like = setup.likelihood_class
        expected = np.linalg.solve(np.asarray(like.t_ninv_t) + np.diag(1 / FG_STD**2), np.asarray(like.dbar))
        np.testing.assert_allclose(coeffs, expected, rtol=1e-10)

    def test_evidence_equals_marginal_likelihood(self, rng, tmp_path):
        """The analytic ``logZ`` equals the marginalised likelihood of the fixed-prior model."""
        setup = _setup(rng, n_eor=0)
        summary = run_multinest(setup, tmp_path / "out")
        np.testing.assert_allclose(summary["logZ"], float(setup.likelihood_class.log_likelihood(jnp.empty(0))))

    def test_resume_is_rejected(self, rng, tmp_path):
        """There is no checkpoint to resume for an analytic evaluation."""
        with pytest.raises(ValueError, match="no sampler checkpoint"):
            run_multinest(_setup(rng, n_eor=0), tmp_path / "out", resume=True)

    def test_intrinsic_noise_not_supported(self, rng, tmp_path):
        """Noise-amplitude fitting is not supported by the sampler wrapper."""
        with pytest.raises(NotImplementedError, match="intrinsic-noise"):
            run_multinest(_setup(rng, n_eor=0, intrinsic_noise_fitting=True), tmp_path / "out")


@pytest.mark.multinest
@pytest.mark.slow
class TestRunMultinestSampling:
    @pytest.fixture
    def result(self, rng, tmp_path):
        """Run MultiNest on a one-parameter problem, returning the setup, summary and output directory."""
        setup = _setup(rng, n_eor=4)
        out = tmp_path / "ns"
        summary = run_multinest(setup, out, random_key=7, n_live_points=150, print_rate=0, calc_avg_eval=True, avg_iters=2)
        return setup, summary, out

    def test_summary_contents(self, result):
        """The summary records the sampler settings, evidence and call count."""
        _, summary, out = result
        assert summary["sampler"] == "multinest" and summary["n_dim"] == 1
        assert summary["n_live_points"] == 150 and summary["random_key"] == 7
        assert summary["loglikelihood_calls"] > 0 and summary["num_posterior_samples"] > 0
        assert summary["avg_eval_iters"] == 2 and summary["avg_eval_seconds"] >= 0.0
        assert json.loads((out / "summary.json").read_text()) == summary

    def test_offset_is_likelihood_at_prior_midpoint(self, result):
        """The centring offset is the likelihood at the prior midpoint, saved for resuming."""
        setup, summary, out = result
        expected = float(setup.likelihood_class.log_likelihood(BOUNDS.mean(axis=1)))
        np.testing.assert_allclose(summary["log_likelihood_offset"], expected)
        assert json.loads((out / "likelihood_offset.json").read_text())["log_likelihood_offset"] == summary["log_likelihood_offset"]

    def test_offset_restored_in_logz(self, result):
        """The reported ``logZ`` is the centred MultiNest value plus the offset."""
        _, summary, _ = result
        np.testing.assert_allclose(summary["logZ"], summary["logZ_centered"] + summary["log_likelihood_offset"])

    def test_evidence_matches_quadrature(self, result):
        """The evidence agrees with direct one-dimensional quadrature over the prior."""
        setup, summary, _ = result
        grid = np.linspace(*BOUNDS[0], 2001)
        logl = np.array([float(setup.likelihood_class.log_likelihood(np.array([x]))) for x in grid])
        log_z = logsumexp(logl) + np.log(grid[1] - grid[0]) - np.log(BOUNDS[0, 1] - BOUNDS[0, 0])
        assert abs(summary["logZ"] - log_z) < 4 * summary["logZerr"] + 0.05

    def test_writes_chains(self, result):
        """MultiNest writes its standard output files to the run directory."""
        _, summary, out = result
        assert Path(f"{summary['logical_outputfiles_basename']}.txt").exists()
        assert (out / "data-post_equal_weights.dat").exists()


@pytest.mark.multinest
class TestRunMultinestResume:
    def test_resume_without_checkpoint_raises(self, rng, tmp_path):
        """Requesting a resume with no checkpoint files raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="no existing checkpoint"):
            run_multinest(_setup(rng, n_eor=4), tmp_path / "ns", resume=True)

    def test_resume_without_offset_raises(self, rng, tmp_path):
        """Resuming requires the saved likelihood offset."""
        out = tmp_path / "ns"
        out.mkdir()
        (out / "data-resume.dat").write_text("")
        with pytest.raises(FileNotFoundError, match="likelihood_offset.json"):
            run_multinest(_setup(rng, n_eor=4), out, resume=True)


@pytest.fixture
def stub_solve(monkeypatch):
    """Stand in for ``pymultinest.solve.solve`` with a recorder that calls both callbacks once.

    Fake modules are installed so these tests need neither PyMultiNest nor MultiNest.
    """
    import sys
    import types

    calls = {}

    def fake_solve(**kwargs):
        calls.update(kwargs)
        n = kwargs["n_dims"]
        theta = kwargs["Prior"](np.full(n, 0.5))
        calls["theta"] = np.asarray(theta)
        calls["centred_loglike"] = kwargs["LogLikelihood"](theta)
        return {"logZ": -3.0, "logZerr": 0.1, "samples": np.zeros((7, n))}

    package = types.ModuleType("pymultinest")
    module = types.ModuleType("pymultinest.solve")
    module.solve = fake_solve
    package.solve = module
    monkeypatch.setitem(sys.modules, "pymultinest", package)
    monkeypatch.setitem(sys.modules, "pymultinest.solve", module)
    return calls


class TestRunMultinestWithStubSampler:
    @pytest.mark.parametrize(
        ("option", "value", "solve_key"),
        [
            ("importance_nested_sampling", True, "importance_nested_sampling"),
            ("multimodal", False, "multimodal"),
            ("const_efficiency_mode", True, "const_efficiency_mode"),
            ("evidence_tolerance", 0.2, "evidence_tolerance"),
            ("sampling_efficiency", 0.3, "sampling_efficiency"),
            ("max_iter", 50, "max_iter"),
            ("random_key", 7, "seed"),
        ],
    )
    def test_optional_settings_are_forwarded_and_recorded(self, rng, stub_solve, option, value, solve_key):
        """Each optional MultiNest setting reaches ``solve`` and is recorded in the summary."""
        summary = run_multinest(_setup(rng, 2), "out", **{option: value})
        assert stub_solve[solve_key] == value
        assert summary[option] == value
        assert solve_key in summary["multinest_kwargs_passed"]

    def test_unset_settings_use_multinest_defaults(self, rng, stub_solve):
        """Settings left as ``None`` are not passed, so PyMultiNest's defaults apply."""
        run_multinest(_setup(rng, 2), "out")
        assert not {"importance_nested_sampling", "multimodal", "max_iter", "seed"} & stub_solve.keys()

    def test_callbacks_map_unit_cube_and_centre_likelihood(self, rng, stub_solve):
        """The prior maps 0.5 to the box midpoint and the likelihood is offset-centred."""
        setup = _setup(rng, 2)
        summary = run_multinest(setup, "out")
        np.testing.assert_allclose(stub_solve["theta"], BOUNDS.mean(axis=1))
        assert stub_solve["centred_loglike"] == pytest.approx(0.0, abs=1e-9)
        assert summary["loglikelihood_calls"] == 1

    def test_reported_evidence_restores_offset(self, rng, stub_solve):
        """The summary ``logZ`` adds back the centring offset recorded on disk."""
        summary = run_multinest(_setup(rng, 2), "out")
        offset = json.loads(Path("out/likelihood_offset.json").read_text())["log_likelihood_offset"]
        assert summary["logZ_centered"] == -3.0
        assert summary["logZ"] == pytest.approx(-3.0 + offset)
        assert summary["num_posterior_samples"] == 7

    def test_print_rate_reports_progress(self, rng, stub_solve, capsys):
        """A print rate of one prints the call count after every likelihood call."""
        run_multinest(_setup(rng, 2), "out", print_rate=1)
        assert "count 1" in capsys.readouterr().out

    def test_average_evaluation_time_is_recorded(self, rng, stub_solve):
        """``calc_avg_eval`` times the likelihood and records the result."""
        summary = run_multinest(_setup(rng, 2), "out", calc_avg_eval=True, avg_iters=2)
        assert summary["avg_eval_seconds"] >= 0.0 and summary["avg_eval_iters"] == 2

    def test_resume_reuses_stored_offset(self, rng, stub_solve):
        """Resuming reads the original offset rather than recomputing it."""
        Path("out").mkdir()
        Path("out/data-resume.dat").write_text("")
        Path("out/likelihood_offset.json").write_text(json.dumps({"log_likelihood_offset": 5.0}))
        summary = run_multinest(_setup(rng, 2), "out", resume=True)
        assert summary["log_likelihood_offset"] == 5.0
        assert summary["resume_checkpoint_files_found"] == ["data-resume.dat"]

    def test_resume_without_offset_record_raises(self, rng, stub_solve):
        """Resuming without ``likelihood_offset.json`` is refused."""
        Path("out").mkdir()
        Path("out/data-resume.dat").write_text("")
        with pytest.raises(FileNotFoundError, match="likelihood_offset.json"):
            run_multinest(_setup(rng, 2), "out", resume=True)

    def test_non_finite_offset_raises(self, rng, stub_solve, monkeypatch):
        """A non-finite likelihood at the prior midpoint is rejected before sampling."""
        setup = _setup(rng, 2)
        monkeypatch.setattr(setup.likelihood_class, "log_likelihood", lambda theta: np.nan)
        with pytest.raises(ValueError, match="finite"):
            run_multinest(setup, "out")
