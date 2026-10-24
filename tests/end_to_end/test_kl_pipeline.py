"""End-to-end checks of KL construction inside ``run_setup``."""

from __future__ import annotations

import json

import numpy as np
import pytest

#: Discarded-RMS thresholds, from keeping every nonzero mode to the default.
THRESHOLDS = (0.0, 0.1)


def _svd_metadata(problem) -> dict:
    """Return the ``.svd.json`` diagnostics written beside the reduced operator."""
    kl_path = problem.setup.config.foreground_kl_path
    return json.loads(kl_path.with_name(f"{kl_path.stem}.svd.json").read_text())


@pytest.mark.parametrize("kl_f", THRESHOLDS)
class TestReducedBasis:
    """The reduced foreground basis has the expected size and diagnostics."""

    def test_rank_bounded(self, kl_problem, problem, kl_f):
        """The retained rank is at most ``min(N_d, N_FG)`` of the physical model."""
        reduced, physical = kl_problem("eor+fg", kl_f), problem("fg")
        rank = _svd_metadata(reduced)["retained_rank"]
        assert 0 < rank <= min(physical.T.shape)

    def test_operator_columns_match_rank(self, kl_problem, problem, kl_f):
        """The joint operator holds the EoR columns followed by the retained KL modes."""
        reduced = kl_problem("eor+fg", kl_f)
        n_eor = problem("eor").T.shape[1]
        assert reduced.T.shape[1] == n_eor + _svd_metadata(reduced)["retained_rank"]

    def test_discarded_rms_within_threshold(self, kl_problem, kl_f):
        """The achieved discarded RMS does not exceed the requested fraction of the noise."""
        metadata = _svd_metadata(kl_problem("eor+fg", kl_f))
        assert metadata["discarded_rms_over_noise_rms"] <= kl_f + 1e-15

    def test_unit_prior(self, kl_problem, problem, kl_f):
        """Reduced foreground coefficients have a standard-normal prior."""
        reduced = kl_problem("eor+fg", kl_f)
        n_eor = problem("eor").T.shape[1]
        np.testing.assert_allclose(reduced.psi(reduced.thetas()[0])[n_eor:], 1.0)

    def test_same_data(self, kl_problem, problem, kl_f):
        """The KL run sees exactly the same noisy data as the physical run."""
        np.testing.assert_array_equal(kl_problem("eor+fg", kl_f).d, problem("eor+fg").d)


def test_rank_decreases_with_threshold(kl_problem):
    """A larger discarded-RMS threshold never retains more modes."""
    ranks = [_svd_metadata(kl_problem("eor+fg", f))["retained_rank"] for f in THRESHOLDS]
    assert ranks == sorted(ranks, reverse=True)


class TestLikelihoodAgreement:
    """The KL likelihood reproduces the physical likelihood to the truncation accuracy."""

    def test_no_truncation(self, kl_problem, problem, dense):
        """Keeping every nonzero mode (``f = 0``), the KL target equals the dense physical target."""
        reduced, physical = kl_problem("eor+fg", 0.0), problem("eor+fg")
        for theta in physical.thetas():
            value = float(reduced.setup.likelihood_class.log_likelihood(theta))
            np.testing.assert_allclose(value, dense.log_likelihood(physical, theta), rtol=0, atol=1e-8 * dense.scale(physical))

    def test_default_truncation(self, kl_problem, problem):
        """At ``f = 0.1`` the KL target differs by no more than ``½ N_d f²``."""
        reduced, physical = kl_problem("eor+fg", 0.1), problem("eor+fg")
        bound = 0.5 * physical.d.size * 0.1**2
        for theta in physical.thetas():
            difference = float(reduced.setup.likelihood_class.log_likelihood(theta)) - float(
                physical.setup.likelihood_class.log_likelihood(theta)
            )
            assert abs(difference) <= bound

    def test_reduced_covariance_matches_dense(self, kl_problem, dense):
        """The KL target equals the dense reference built from its own reduced operator."""
        reduced = kl_problem("eor+fg", 0.1)
        for theta in reduced.thetas():
            value = float(reduced.setup.likelihood_class.log_likelihood(theta))
            np.testing.assert_allclose(value, dense.log_likelihood(reduced, theta), rtol=0, atol=1e-8 * dense.scale(reduced))
