"""End-to-end MultiNest run of the small EoR configuration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bayeseor_jax import DataContainer, run_nestedsampling

PRIOR = [[4.0, 10.0], [4.0, 10.0]]

pytestmark = [pytest.mark.multinest, pytest.mark.slow]


@pytest.fixture(scope="module")
def sampled(tmp_path_factory, build_setup):
    """Set up the EoR model and sample it with a small, seeded MultiNest run."""
    directory = tmp_path_factory.mktemp("sampling")
    setup = build_setup(directory, model="eor", priors=PRIOR)
    out_dir = directory / "output" / "nested_sampling"
    summary = run_nestedsampling(setup, out_dir=out_dir, n_live_points=25, random_key=42, print_rate=0)
    return setup, summary, out_dir, directory / "output"


class TestMultiNestRun:
    """A short MultiNest run completes and records its results."""

    def test_sampler(self, sampled):
        """The run used MultiNest over the two power-spectrum amplitudes."""
        _, summary, _, _ = sampled
        assert summary["sampler"] == "multinest"
        assert summary["n_dim"] == 2
        assert summary["n_live_points"] == 25

    def test_evidence_finite(self, sampled):
        """The evidence and its uncertainty are finite."""
        _, summary, _, _ = sampled
        assert np.isfinite(summary["logZ"])
        assert np.isfinite(summary["logZerr"]) and summary["logZerr"] >= 0.0

    def test_summary_written(self, sampled):
        """``summary.json`` records the same evidence as the returned summary."""
        _, summary, out_dir, _ = sampled
        assert json.loads((out_dir / "summary.json").read_text())["logZ"] == pytest.approx(summary["logZ"])

    def test_offset_recorded(self, sampled):
        """The likelihood-centring offset is saved for resuming."""
        _, _, out_dir, _ = sampled
        offset = json.loads((out_dir / "likelihood_offset.json").read_text())["log_likelihood_offset"]
        assert np.isfinite(offset)

    def test_evidence_bounded_by_likelihood(self, sampled):
        """``logZ`` does not exceed the largest likelihood found on a grid over the prior."""
        setup, summary, _, _ = sampled
        grid = np.linspace(PRIOR[0][0], PRIOR[0][1], 13)
        best = max(float(setup.likelihood_class.log_likelihood(np.array([a, b]))) for a in grid for b in grid)
        assert summary["logZ"] <= best + 1.0


class TestReadingChains:
    """``DataContainer`` reads the run's chain together with the setup metadata."""

    @pytest.fixture(scope="class")
    @staticmethod
    def container(sampled):
        """Load the chain, taking k-bin metadata from the setup output directory."""
        _, _, out_dir, metadata_dir = sampled
        return DataContainer([out_dir], metadata_dirs=[metadata_dir])

    def test_samples_shape(self, container):
        """Each posterior sample has one value per power-spectrum bin."""
        assert container.samples[0].ndim == 2 and container.samples[0].shape[1] == 2

    def test_weights_normalised(self, container):
        """Posterior weights are non-negative and sum to one."""
        weights = container.weights[0]
        assert np.all(weights >= 0.0)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-10)

    def test_samples_within_prior(self, container):
        """Every sample lies inside the configured ``log10`` prior box."""
        samples = container.samples[0]
        bounds = np.asarray(PRIOR)
        assert np.all(samples >= bounds[:, 0]) and np.all(samples <= bounds[:, 1])

    def test_medians_in_linear_units(self, container):
        """Reported medians are linear powers in mK², inside the prior range."""
        medians = np.asarray(container.medians[0])
        assert np.all(medians >= 10.0 ** PRIOR[0][0]) and np.all(medians <= 10.0 ** PRIOR[0][1])
