"""Tests for ``bayeseor_jax.analyze.plotting``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bayeseor_jax.analyze.plotting import (
    _default_colors,
    _prior_bounds,
    plot_posteriors,
    plot_power_spectra,
    plot_power_spectra_and_posteriors,
    save_analysis,
)
from bayeseor_jax.analyze.results import DataContainer

K_VALS = np.array([0.2, 0.4, 0.8])
K_EDGES = np.array([0.1, 0.3, 0.6, 1.0])
PRIORS = [[0.0, 4.0], [0.5, 4.5], [1.0, 5.0]]


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure created by a test."""
    yield
    plt.close("all")


def write_run(path: Path, rng, *, model="eor", k=K_VALS, edges=K_EDGES, priors=PRIORS) -> Path:
    """Write a small MultiNest-style run with k-bin metadata and args.json."""
    path.mkdir(parents=True)
    n = 300
    chain = np.column_stack([
        rng.uniform(0.1, 1.0, n),
        rng.uniform(0.0, 5.0, n),
        2.0 + 0.2 * rng.normal(size=(n, k.size)),
    ])
    np.savetxt(path / "data-.txt", chain)
    np.savetxt(path / "k-vals.txt", k)
    if edges is not None:
        np.savetxt(path / "k-vals-bins.txt", edges)
    (path / "args.json").write_text(json.dumps({"model": model, "priors": priors}))
    return path


@pytest.fixture
def one_run(tmp_path, rng):
    """A ``DataContainer`` with one 21-cm-only run and a reference spectrum."""
    return DataContainer([write_run(tmp_path / "eor", rng)], expected_dmps=100.0)


@pytest.fixture
def two_runs(tmp_path, rng):
    """A ``DataContainer`` comparing a 21-cm-only and a joint run on the same grid."""
    return DataContainer(
        [write_run(tmp_path / "eor", rng), write_run(tmp_path / "joint", rng, model="eor+fg")],
        labels=["EoR", "Joint"],
    )


class TestPriorBounds:
    """Reading prior bounds for shading."""

    def test_converts_to_linear(self, one_run):
        """Log10 bounds are returned as linear powers."""
        np.testing.assert_allclose(_prior_bounds(one_run, 0), 10.0 ** np.asarray(PRIORS))

    @pytest.mark.parametrize(
        "priors", [None, [[0, 4], [0, 4]], [[4, 0], [0, 4], [0, 4]], [[0, 400], [0, 4], [0, 4]]],
    )
    def test_rejects_invalid(self, tmp_path, rng, priors):
        """Missing, mis-sized, unordered or overflowing bounds raise ``ValueError``."""
        data = DataContainer([write_run(tmp_path / "r", rng, priors=priors)])
        with pytest.raises(ValueError, match="[Pp]rior"):
            _prior_bounds(data, 0)


class TestDefaultColors:
    """Choosing colours from each run's model."""

    def test_model_colors(self, two_runs):
        """21-cm-only runs take the first style colour and joint runs the second."""
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        assert _default_colors(two_runs) == [palette[0], palette[1]]

    def test_unknown_model_uses_index(self, tmp_path, rng):
        """Runs without a recognised model fall back to their position in the cycle."""
        data = DataContainer([write_run(tmp_path / "a", rng, model="fg"),
                              write_run(tmp_path / "b", rng, model=None)])
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        assert _default_colors(data) == [palette[0], palette[1 % len(palette)]]


class TestPlotPowerSpectra:
    """Plotting Δ²(k) constraints."""

    def test_returns_figure_and_axes(self, one_run):
        """A figure and logarithmic axes are returned."""
        fig, ax = plot_power_spectra(one_run)
        assert ax.figure is fig
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"

    def test_reference_and_priors(self, one_run):
        """The reference curve and one prior patch per bin are drawn."""
        _, ax = plot_power_spectra(one_run, plot_priors=True)
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Reference" in labels
        assert len(ax.patches) == K_VALS.size

    def test_uses_existing_axes(self, one_run):
        """An existing axis is drawn into and returned."""
        fig, ax = plt.subplots()
        assert plot_power_spectra(one_run, ax=ax) == (fig, ax)

    @pytest.mark.parametrize("mask", [[True, False, True], [[True, False, False]]])
    def test_upper_limit_masks(self, one_run, mask):
        """A shared mask, or one mask per run, selects bins drawn as upper limits."""
        plot_power_spectra(one_run, uplim_inds=mask)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"colors": ["r", "b"]}, "colors"),
            ({"uplim_inds": [True, False]}, "Upper-limit masks"),
            ({"uplim_inds": [[True, False, True], [True, True, True]]}, "one upper-limit mask per run"),
        ],
    )
    def test_invalid_options(self, one_run, kwargs, match):
        """Mis-sized colours or upper-limit masks raise ``ValueError``."""
        with pytest.raises(ValueError, match=match):
            plot_power_spectra(one_run, **kwargs)

    def test_priors_require_edges(self, tmp_path, rng):
        """Prior shading without bin edges raises ``ValueError``."""
        data = DataContainer([write_run(tmp_path / "r", rng, edges=None)])
        with pytest.raises(ValueError, match="k-vals-bins.txt"):
            plot_power_spectra(data, plot_priors=True)

    def test_shared_priors_shaded_once(self, two_runs):
        """Identical priors on the same edges are shaded only once."""
        _, ax = plot_power_spectra(two_runs, plot_priors=True)
        assert len(ax.patches) == K_VALS.size


class TestPlotPosteriors:
    """Plotting stacked marginal posteriors."""

    def test_one_axis_per_bin(self, two_runs):
        """One axis is returned per k bin, sharing a logarithmic x axis."""
        fig, axes = plot_posteriors(two_runs, plot_priors=True)
        assert axes.shape == (K_VALS.size,)
        assert all(ax.figure is fig and ax.get_xscale() == "log" for ax in axes)

    def test_peak_normalised(self, one_run):
        """Peak normalisation scales every histogram to a maximum of one."""
        _, axes = plot_posteriors(one_run)
        for ax in axes:
            heights = ax.patches[0].get_data().values
            assert heights.max() == pytest.approx(1.0)

    def test_density_normalised(self, one_run):
        """Density normalisation integrates to one over log10 power."""
        _, axes = plot_posteriors(one_run, normalisation="density", bins=20)
        for ax in axes:
            data = ax.patches[0].get_data()
            values, edges = data.values, data.edges
            assert np.sum(values * np.diff(np.log10(edges))) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"normalisation": "area"}, "normalisation"),
            ({"bins": 0}, "bins"),
            ({"bins": 2.5}, "bins"),
            ({"colors": ["r"]}, "colors"),
        ],
    )
    def test_invalid_options(self, two_runs, kwargs, match):
        """Invalid normalisation, bins or colours raise ``ValueError``."""
        with pytest.raises(ValueError, match=match):
            plot_posteriors(two_runs, **kwargs)

    def test_requires_corresponding_bins(self, tmp_path, rng):
        """Runs on different grids need ``bins_correspond=True``."""
        data = DataContainer([write_run(tmp_path / "a", rng),
                              write_run(tmp_path / "b", rng, k=K_VALS * 1.05, edges=None)])
        with pytest.raises(ValueError, match="bins_correspond"):
            plot_posteriors(data)

    def test_rejects_wrong_axes(self, one_run):
        """Supplying the wrong number of axes raises ``ValueError``."""
        _, axes = plt.subplots(2, 1)
        with pytest.raises(ValueError, match="one axis per k bin"):
            plot_posteriors(one_run, axes=axes)


class TestPlotPowerSpectraAndPosteriors:
    """The combined power-spectrum and posterior figure."""

    def test_layout(self, two_runs):
        """One spectrum axis and one posterior axis per bin share a single figure."""
        fig, (power_ax, posterior_axes) = plot_power_spectra_and_posteriors(two_runs, plot_priors=True)
        assert power_ax.figure is fig
        assert len(posterior_axes) == K_VALS.size
        assert posterior_axes[0].get_legend() is None

    def test_requires_corresponding_bins(self, tmp_path, rng):
        """Runs on different grids need ``bins_correspond=True``."""
        data = DataContainer([write_run(tmp_path / "a", rng),
                              write_run(tmp_path / "b", rng, k=K_VALS * 1.05, edges=None)])
        with pytest.raises(ValueError, match="bins_correspond"):
            plot_power_spectra_and_posteriors(data)


class TestSaveAnalysis:
    """Writing the posterior summary and combined figure."""

    def test_writes_outputs(self, tmp_path, rng):
        """``posterior_summary.json`` and the PNG are written, matching the returned summary."""
        run = write_run(tmp_path / "run", rng)
        out = tmp_path / "plots"
        summary = save_analysis(run, output_dir=out, label="Test", expected_ps=1e4)
        assert (out / "power_spectrum_and_posteriors.png").stat().st_size > 0
        assert json.loads((out / "posterior_summary.json").read_text()) == summary
        dc = DataContainer([run])
        np.testing.assert_allclose(summary["medians"], dc.medians[0])
        np.testing.assert_allclose(summary["k_vals"], K_VALS)
        assert set(summary["credible_intervals"]) == {"68.0", "95.0"}
        assert summary["sampler_dir"] == str(run)

    def test_separate_metadata_dir(self, tmp_path, rng):
        """k-bin metadata and priors may come from a separate setup directory."""
        meta = write_run(tmp_path / "setup", rng)
        run = tmp_path / "chains"
        run.mkdir()
        (run / "data-.txt").write_text((meta / "data-.txt").read_text())
        summary = save_analysis(run, output_dir=tmp_path / "plots", metadata_dir=meta,
                                uplim_inds=[True, False, False])
        np.testing.assert_allclose(summary["k_vals"], K_VALS)
        assert not plt.get_fignums()
