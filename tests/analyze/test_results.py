"""Tests for ``bayeseor_jax.analyze.results``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayeseor_jax.analyze.results import (
    DataContainer,
    _chain_path,
    _weighted_quantiles,
    dmps_to_ps,
    ps_to_dmps,
)

K_VALS = np.array([0.2, 0.4, 0.8])
K_EDGES = np.array([0.1, 0.3, 0.6, 1.0])


def write_run(path: Path, chain: np.ndarray, *, k=K_VALS, edges=K_EDGES, args=None,
              summary=None, chain_name="data-.txt") -> Path:
    """Write a MultiNest-style run directory with k-bin metadata and return it."""
    path.mkdir(parents=True, exist_ok=True)
    np.savetxt(path / "k-vals.txt", k)
    if edges is not None:
        np.savetxt(path / "k-vals-bins.txt", edges)
    if args is not None:
        (path / "args.json").write_text(json.dumps(args))
    if summary is not None:
        (path / "summary.json").write_text(json.dumps(summary))
    np.savetxt(path / chain_name, chain)
    return path


def make_chain(rng, n=400, n_bins=3, centre=2.0) -> np.ndarray:
    """Return a weighted chain with columns (weight, -2 log L, log10 powers...)."""
    weights = rng.uniform(0.1, 1.0, n)
    log_powers = centre + 0.1 * rng.normal(size=(n, n_bins))
    minus_two_log_l = rng.uniform(0.0, 10.0, n)
    return np.column_stack([weights, minus_two_log_l, log_powers])


class TestPowerConversions:
    """Converting between P(k) and Δ²(k)."""

    def test_ps_to_dmps_formula(self):
        """Δ²(k) = k³ P(k) / 2π²."""
        assert ps_to_dmps(2.0, 3.0) == pytest.approx(2.0 * 27.0 / (2 * np.pi**2))

    def test_dmps_to_ps_formula(self):
        """P(k) = 2π² Δ²(k) / k³."""
        assert dmps_to_ps(5.0, 0.5) == pytest.approx(5.0 * 2 * np.pi**2 / 0.125)

    def test_round_trip(self, rng):
        """The two conversions are inverse to each other, element-wise."""
        power, k = rng.uniform(1, 100, 10), rng.uniform(0.05, 2.0, 10)
        np.testing.assert_allclose(dmps_to_ps(ps_to_dmps(power, k), k), power)

    def test_broadcasts(self):
        """A scalar power broadcasts across an array of wavenumbers."""
        assert ps_to_dmps(1.0, K_VALS).shape == K_VALS.shape


class TestWeightedQuantiles:
    """Interpolating weighted empirical CDFs."""

    @staticmethod
    def reference(column, quantile, weights):
        """Brute-force weighted-CDF interpolation for one column."""
        pairs = sorted((v, w) for v, w in zip(column, weights) if w > 0)
        values = [v for v, _ in pairs]
        cdf, running = [], 0.0
        total = sum(w for _, w in pairs)
        for _, w in pairs:
            running += w
            cdf.append(running / total)
        return np.interp(quantile, cdf, values)

    def test_matches_brute_force(self, rng):
        """Each column agrees with an explicit weighted-CDF interpolation."""
        samples, weights = rng.normal(size=(50, 3)), rng.uniform(0, 1, 50)
        quantiles = [0.05, 0.5, 0.95]
        result = _weighted_quantiles(samples, quantiles, weights)
        assert result.shape == (3, 3)
        for j in range(3):
            for i, q in enumerate(quantiles):
                assert result[i, j] == pytest.approx(self.reference(samples[:, j], q, weights))

    def test_equal_weights_hit_samples(self):
        """With equal weights, quantile (i+1)/n returns the (i+1)-th smallest sample."""
        samples = np.array([[3.0], [1.0], [4.0], [2.0]])
        result = _weighted_quantiles(samples, [0.25, 0.5, 0.75, 1.0], np.ones(4))
        np.testing.assert_allclose(result[:, 0], [1.0, 2.0, 3.0, 4.0])

    def test_zero_weights_ignored(self, rng):
        """Samples with zero weight do not change the result."""
        samples = rng.normal(size=(20, 2))
        weights = rng.uniform(0.1, 1, 20)
        padded = np.vstack([samples, [[1e6, -1e6]]])
        result = _weighted_quantiles(padded, [0.1, 0.9], np.r_[weights, 0.0])
        np.testing.assert_allclose(result, _weighted_quantiles(samples, [0.1, 0.9], weights))


class TestChainPath:
    """Locating the weighted MultiNest chain in a run directory."""

    def test_recorded_basename(self, tmp_path):
        """The chain named by ``outputfiles_basename`` in the summary is used."""
        (tmp_path / "short-.txt").write_text("")
        assert _chain_path(tmp_path, {"outputfiles_basename": "/elsewhere/short-"}) == tmp_path / "short-.txt"

    def test_recorded_basename_missing(self, tmp_path):
        """A recorded chain that does not exist raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="missing"):
            _chain_path(tmp_path, {"outputfiles_basename": "gone-"})

    def test_default_name(self, tmp_path):
        """Without a summary, ``data-.txt`` is preferred."""
        (tmp_path / "data-.txt").write_text("")
        (tmp_path / "other-.txt").write_text("")
        assert _chain_path(tmp_path, {}) == tmp_path / "data-.txt"

    def test_single_candidate(self, tmp_path):
        """A single ``*-.txt`` file is used when ``data-.txt`` is absent."""
        (tmp_path / "run-.txt").write_text("")
        assert _chain_path(tmp_path, {}) == tmp_path / "run-.txt"

    @pytest.mark.parametrize("names", [[], ["a-.txt", "b-.txt"]])
    def test_ambiguous_or_missing(self, tmp_path, names):
        """Zero or several candidate chains raise ``FileNotFoundError``."""
        for name in names:
            (tmp_path / name).write_text("")
        with pytest.raises(FileNotFoundError, match="Expected one"):
            _chain_path(tmp_path, {})


class TestDataContainer:
    """Reading runs and summarising weighted Δ²(k) posteriors."""

    def test_summaries_match_weighted_statistics(self, tmp_path, rng):
        """Means, medians, intervals and limits equal weighted statistics of 10**samples."""
        chain = make_chain(rng)
        run = write_run(tmp_path / "run", chain)
        dc = DataContainer([run], cred_intervals=(68,), uplim_quantile=0.9)
        weights = chain[:, 0] / chain[:, 0].sum()
        powers = 10.0 ** chain[:, 2:]
        np.testing.assert_allclose(dc.weights[0], weights)
        np.testing.assert_allclose(dc.avgs[0], np.average(powers, axis=0, weights=weights))
        expected = _weighted_quantiles(powers, [0.5, 0.9, 0.16, 0.84], weights)
        np.testing.assert_allclose(dc.medians[0], expected[0])
        np.testing.assert_allclose(dc.uplims[0], expected[1])
        np.testing.assert_allclose(dc.cred_intervals[0][68.0]["lo"], expected[2])
        np.testing.assert_allclose(dc.cred_intervals[0][68.0]["hi"], expected[3])

    def test_samples_and_log_likelihoods(self, tmp_path, rng):
        """Samples are the log powers and log likelihoods are -0.5 × column 1 plus the offset."""
        chain = make_chain(rng)
        run = write_run(tmp_path / "run", chain, summary={"log_likelihood_offset": 7.5})
        dc = DataContainer([run])
        np.testing.assert_array_equal(dc.samples[0], chain[:, 2:])
        np.testing.assert_allclose(dc.log_likelihoods[0], -0.5 * chain[:, 1] + 7.5)
        assert dc.summaries[0] == {"log_likelihood_offset": 7.5}

    def test_metadata(self, tmp_path, rng):
        """k values, edges, args and default labels are read from each run."""
        run = write_run(tmp_path / "myrun", make_chain(rng), args={"model": "eor"})
        dc = DataContainer([run])
        np.testing.assert_allclose(dc.k_vals[0], K_VALS)
        np.testing.assert_allclose(dc.k_vals_bins[0], K_EDGES)
        assert dc.args[0] == {"model": "eor"}
        assert dc.labels == ["myrun"] and dc.Ndirs == 1

    def test_optional_files_absent(self, tmp_path, rng):
        """Missing edges, args and summary give ``None`` and empty dictionaries."""
        dc = DataContainer([write_run(tmp_path / "r", make_chain(rng), edges=None)])
        assert dc.k_vals_bins[0] is None and dc.args[0] == {} and dc.summaries[0] == {}

    def test_shared_metadata_dir(self, tmp_path, rng):
        """A single ``metadata_dirs`` path supplies k-bins for every run."""
        meta = tmp_path / "setup"
        meta.mkdir()
        np.savetxt(meta / "k-vals.txt", K_VALS)
        runs = []
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
            np.savetxt(tmp_path / name / "data-.txt", make_chain(rng))
            runs.append(tmp_path / name)
        dc = DataContainer(runs, metadata_dirs=meta, labels=["A", "B"])
        assert dc.Ndirs == 2 and dc.labels == ["A", "B"]
        assert dc.k_vals_identical and dc.bins_correspond

    def test_expected_ps_converted_per_run(self, tmp_path, rng):
        """A reference P(k) is converted to Δ²(k) on each run's own grid."""
        run = write_run(tmp_path / "r", make_chain(rng))
        dc = DataContainer([run], expected_ps=100.0)
        np.testing.assert_allclose(dc.expected_spectra[0], ps_to_dmps(100.0, K_VALS))

    def test_expected_dmps_broadcast(self, tmp_path, rng):
        """A scalar reference Δ²(k) is broadcast to every bin."""
        dc = DataContainer([write_run(tmp_path / "r", make_chain(rng))], expected_dmps=3.0)
        np.testing.assert_allclose(dc.expected_spectra[0], [3.0, 3.0, 3.0])

    def test_no_reference(self, tmp_path, rng):
        """Without a reference, each run's expected spectrum is ``None``."""
        assert DataContainer([write_run(tmp_path / "r", make_chain(rng))]).expected_spectra == [None]

    def test_k_vals_identical_tolerates_rounding(self, tmp_path, rng):
        """Grids differing only by rounding count as identical."""
        a = write_run(tmp_path / "a", make_chain(rng))
        b = write_run(tmp_path / "b", make_chain(rng), k=K_VALS * (1 + 1e-13))
        assert DataContainer([a, b]).k_vals_identical

    def test_bins_correspond_flag(self, tmp_path, rng):
        """Different grids correspond only when explicitly confirmed."""
        a = write_run(tmp_path / "a", make_chain(rng))
        b = write_run(tmp_path / "b", make_chain(rng), k=K_VALS * 1.1, edges=None)
        assert not DataContainer([a, b]).bins_correspond
        assert DataContainer([a, b], bins_correspond=True).bins_correspond

    def test_bins_correspond_requires_same_count(self, tmp_path, rng):
        """Confirming correspondence between different bin counts raises ``ValueError``."""
        a = write_run(tmp_path / "a", make_chain(rng))
        b = write_run(tmp_path / "b", make_chain(rng, n_bins=2), k=K_VALS[:2], edges=None)
        with pytest.raises(ValueError, match="same count"):
            DataContainer([a, b], bins_correspond=True)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"labels": ["a", "b"]}, "labels"),
            ({"metadata_dirs": ["x", "y"]}, "metadata_dirs"),
            ({"cred_intervals": (0,)}, "Credible intervals"),
            ({"cred_intervals": (100,)}, "Credible intervals"),
            ({"uplim_quantile": 1.0}, "uplim_quantile"),
            ({"expected_ps": 1.0, "expected_dmps": 1.0}, "not both"),
            ({"expected_dmps": -1.0}, "Reference power"),
        ],
    )
    def test_invalid_options(self, tmp_path, rng, kwargs, match):
        """Inconsistent or out-of-range options raise ``ValueError``."""
        run = write_run(tmp_path / "r", make_chain(rng))
        with pytest.raises(ValueError, match=match):
            DataContainer([run], **kwargs)

    def test_requires_a_run(self):
        """An empty list of runs raises ``ValueError``."""
        with pytest.raises(ValueError, match="At least one"):
            DataContainer([])

    def test_rejects_analytic_run(self, tmp_path, rng):
        """An analytic foreground-only run has no chain and is rejected."""
        run = write_run(tmp_path / "r", make_chain(rng), summary={"sampler": "analytic"})
        with pytest.raises(ValueError, match="Analytic"):
            DataContainer([run])

    @pytest.mark.parametrize("k", [np.array([0.4, 0.2, 0.8]), np.array([0.0, 0.2, 0.4])])
    def test_rejects_invalid_k(self, tmp_path, rng, k):
        """Non-increasing or non-positive wavenumbers raise ``ValueError``."""
        run = write_run(tmp_path / "r", make_chain(rng), k=k, edges=None)
        with pytest.raises(ValueError, match="k-vals.txt"):
            DataContainer([run])

    @pytest.mark.parametrize("edges", [np.array([0.1, 0.3, 1.0]), np.array([0.3, 0.35, 0.6, 1.0])])
    def test_rejects_invalid_edges(self, tmp_path, rng, edges):
        """Edges of the wrong length or not enclosing each k raise ``ValueError``."""
        run = write_run(tmp_path / "r", make_chain(rng), edges=edges)
        with pytest.raises(ValueError, match="k-vals-bins.txt"):
            DataContainer([run])

    def test_rejects_wrong_chain_width(self, tmp_path, rng):
        """A chain without one column per bin raises ``ValueError``."""
        run = write_run(tmp_path / "r", make_chain(rng, n_bins=2))
        with pytest.raises(ValueError, match="Weighted chain"):
            DataContainer([run])

    def test_rejects_negative_weights(self, tmp_path, rng):
        """Negative posterior weights raise ``ValueError``."""
        chain = make_chain(rng)
        chain[0, 0] = -1.0
        with pytest.raises(ValueError, match="weights"):
            DataContainer([write_run(tmp_path / "r", chain)])

    def test_rejects_overflowing_powers(self, tmp_path, rng):
        """Log powers that overflow on conversion raise ``ValueError``."""
        chain = make_chain(rng)
        chain[0, 2] = 400.0
        with pytest.raises(ValueError, match="finite positive powers"):
            DataContainer([write_run(tmp_path / "r", chain)])

    def test_rejects_non_finite_offset(self, tmp_path, rng):
        """A non-finite saved likelihood offset raises ``ValueError``."""
        run = write_run(tmp_path / "r", make_chain(rng))
        (run / "summary.json").write_text('{"log_likelihood_offset": Infinity}')
        with pytest.raises(ValueError, match="offset"):
            DataContainer([run])
