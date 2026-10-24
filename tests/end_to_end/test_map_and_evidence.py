"""End-to-end checks of MAP reconstruction and the foreground-only evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bayeseor_jax import MaximumAPosteriori, run_nestedsampling


def _map_solver(setup) -> MaximumAPosteriori:
    """Build the MAP solver from a setup's cached likelihood products."""
    likelihood = setup.likelihood_class
    return MaximumAPosteriori(
        t_ninv_t=likelihood.t_ninv_t,
        dbar=likelihood.dbar,
        d_ninv_d=likelihood.d_ninv_d,
        n_data=likelihood.n_data,
        phi_inv_class=setup.phi_inverse_class,
        k_vals=setup.k_vals,
        n_power_params=likelihood.n_power_params,
        mesh=likelihood.mesh,
    )


def _visibility_tolerance(problem) -> float:
    """Absolute tolerance on model visibilities: ``1e-6`` of the largest data amplitude.

    The bright foreground prior makes the coefficient system ill-conditioned, so
    reconstructions agree to about ``1e-8`` relative rather than to round-off.
    """
    return 1e-6 * float(np.max(np.abs(problem.d)))


@pytest.mark.parametrize("model", ("eor", "fg", "eor+fg"))
class TestConditionalMap:
    """``MaximumAPosteriori`` reproduces the dense posterior mean."""

    def test_model_visibilities(self, problem, dense, model):
        """``T b̂`` equals the dense reconstruction ``T Ψ^½ A^H (I + A A^H)⁻¹ N^-½ d``."""
        p = problem(model)
        solver = _map_solver(p.setup)
        for theta in p.thetas():
            coefficients, _ = solver.map_estimate(theta)
            np.testing.assert_allclose(p.T @ coefficients, p.T @ dense.map(p, theta), rtol=0, atol=_visibility_tolerance(p))

    def test_objective_matches_likelihood(self, problem, model):
        """The MAP objective equals the marginalised log likelihood at the same parameters."""
        p = problem(model)
        solver = _map_solver(p.setup)
        for theta in p.thetas():
            _, objective = solver.map_estimate(theta)
            np.testing.assert_allclose(objective, float(p.setup.likelihood_class.log_likelihood(theta)), rtol=1e-12)


class TestEorCoefficients:
    """With a well-conditioned EoR-only system, the coefficients themselves agree."""

    def test_coefficients(self, problem, dense):
        """The EoR MAP coefficients equal the dense posterior mean."""
        p = problem("eor")
        solver = _map_solver(p.setup)
        for theta in p.thetas():
            coefficients, _ = solver.map_estimate(theta)
            expected = dense.map(p, theta)
            np.testing.assert_allclose(coefficients, expected, rtol=0, atol=1e-8 * np.max(np.abs(expected)))

    def test_dmps_input(self, problem):
        """Passing ``dmps`` in mK² is equivalent to passing its ``log10`` as ``theta``."""
        p = problem("eor")
        solver = _map_solver(p.setup)
        theta = p.thetas()[1]
        from_theta, objective_theta = solver.map_estimate(theta)
        from_dmps, objective_dmps = solver.map_estimate(dmps=10.0**theta)
        np.testing.assert_allclose(from_dmps, from_theta, rtol=1e-12)
        np.testing.assert_allclose(objective_dmps, objective_theta, rtol=1e-12)


class TestForegroundOnlyEvidence:
    """``run_nestedsampling`` evaluates the parameter-free foreground model analytically."""

    @pytest.fixture
    def summary(self, problem, tmp_path):
        """Run the analytic foreground-only evaluation into ``tmp_path``."""
        return run_nestedsampling(problem("fg").setup, out_dir=tmp_path)

    def test_sampler_is_analytic(self, summary):
        """No sampler is used, and the evidence has no sampling error."""
        assert summary["sampler"] == "analytic"
        assert summary["n_dim"] == 0
        assert summary["logZerr"] == 0.0

    def test_evidence_matches_dense(self, summary, problem, dense):
        """``logZ`` equals the dense marginal likelihood of the foreground-only model."""
        p = problem("fg")
        np.testing.assert_allclose(summary["logZ"], dense.log_likelihood(p, np.empty(0)), rtol=0, atol=1e-8 * dense.scale(p))

    def test_summary_written(self, summary, tmp_path):
        """``summary.json`` on disk matches the returned summary."""
        assert json.loads((tmp_path / "summary.json").read_text()) == summary

    def test_map_coefficients_written(self, summary, problem, dense):
        """The saved MAP coefficients reproduce the dense foreground reconstruction."""
        p = problem("fg")
        coefficients = np.load(summary["map_coefficients_path"])
        np.testing.assert_allclose(p.T @ coefficients, p.T @ dense.map(p, np.empty(0)), rtol=0, atol=_visibility_tolerance(p))

    def test_resume_rejected(self, problem, tmp_path):
        """Resuming is meaningless without a sampler and is rejected."""
        with pytest.raises(ValueError, match="resume"):
            run_nestedsampling(problem("fg").setup, out_dir=tmp_path, resume=True)
