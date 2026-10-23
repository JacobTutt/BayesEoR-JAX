"""Check weighted constraints and diagnostic figures without running inference."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bayeseor_jax.analyze import (
    DataContainer, dmps_to_ps, ps_to_dmps, plot_power_spectra,
    plot_posteriors, plot_power_spectra_and_posteriors,
)


def write_run(path, k=(.1, .2), chain=None):
    path.mkdir(exist_ok=True)
    k = np.asarray(k)
    np.savetxt(path / "k-vals.txt", k)
    edges = np.r_[k[0] / 2, (k[:-1] + k[1:]) / 2, k[-1] * 1.5]
    np.savetxt(path / "k-vals-bins.txt", edges)
    if chain is None:
        chain = np.column_stack(([.2, .3, .5], [8., 6., 4.],
                                 np.tile([0., .5, 1.], (k.size, 1)).T))
    np.savetxt(path / "data-.txt", chain)
    (path / "args.json").write_text(json.dumps({"priors": [[-1., 2.]] * k.size}))
    return path


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_statistics_and_units(tmp_path):
    data = DataContainer([write_run(tmp_path)], expected_ps=10.)
    values = np.array([1., np.sqrt(10.), 10.])
    assert data.samples[0].shape == (3, 2)
    np.testing.assert_allclose(data.medians[0], np.sqrt(10.))
    np.testing.assert_allclose(data.uplims[0], np.interp(.95, [.2, .5, 1.], values))
    np.testing.assert_allclose(data.cred_intervals[0][68]["lo"], 1.)
    np.testing.assert_allclose(data.cred_intervals[0][68]["hi"], np.interp(.84, [.2, .5, 1.], values))
    np.testing.assert_allclose(data.expected_spectra[0], ps_to_dmps(10., data.k_vals[0]))
    np.testing.assert_allclose(dmps_to_ps(data.expected_spectra[0], data.k_vals[0]), 10.)


def test_zero_weight_points_do_not_change_quantiles(tmp_path):
    path = write_run(tmp_path, k=(.1,), chain=[[0., 1., -5.], [.25, 1., 0.], [.75, 1., 1.]])
    data = DataContainer([path])
    np.testing.assert_allclose(data.medians[0], 4.)


def test_single_bin_single_sample_and_all_plots(tmp_path):
    data = DataContainer([write_run(tmp_path, k=(.1,), chain=[[1., 4., .5]])],
                         expected_dmps=2., labels=["one bin"])
    np.testing.assert_allclose(data.medians[0], np.sqrt(10.))
    for plot in (plot_power_spectra, plot_posteriors, plot_power_spectra_and_posteriors):
        fig, axes = plot(data, plot_priors=True)
        fig.canvas.draw()
    assert axes[1].shape == (1,)
    fig, ax = plot_power_spectra(data, uplim_inds=[True])
    fig.canvas.draw()
    assert len(ax.containers) == 2


def test_multiple_runs_references_and_shared_histogram_edges(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", chain=[[.1, 1., 1., 2.], [.9, 2., 2., 3.]])
    data = DataContainer([a, b], expected_dmps=[2., 3.])
    fig, axes = plot_posteriors(data, bins=3, normalisation="density")
    for ax in axes:
        one, two = [patch.get_data() for patch in ax.patches]
        np.testing.assert_array_equal(one.edges, two.edges)
        np.testing.assert_allclose(np.sum(one.values * np.diff(np.log10(one.edges))), 1.)
        np.testing.assert_allclose(np.sum(two.values * np.diff(np.log10(two.edges))), 1.)
    fig.canvas.draw()
    fig, _ = plot_power_spectra_and_posteriors(data, plot_priors=True,
                                             uplim_inds=[[True, False], [False, True]])
    fig.canvas.draw()


def test_different_grids_supported_only_for_power_spectra(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", k=(.15,))
    data = DataContainer([a, b], expected_ps=10.)
    assert not data.k_vals_identical
    np.testing.assert_allclose(data.expected_spectra[1], ps_to_dmps(10., [.15]))
    fig, _ = plot_power_spectra(data, uplim_inds=[[True, False], [False]], plot_priors=True)
    fig.canvas.draw()
    for plot in (plot_posteriors, plot_power_spectra_and_posteriors):
        with pytest.raises(ValueError, match="bins_correspond"):
            plot(data)


def test_recorded_chain_takes_precedence(tmp_path):
    write_run(tmp_path)
    np.savetxt(tmp_path / "short-.txt", [[1., 6., 1., 2.]])
    (tmp_path / "summary.json").write_text(json.dumps({"outputfiles_basename": "/old/location/short-"}))
    data = DataContainer([tmp_path])
    np.testing.assert_allclose(data.medians[0], [10., 100.])
    np.testing.assert_allclose(data.log_likelihoods[0], [-3.])


def test_missing_recorded_chain_does_not_load_stale_default(tmp_path):
    write_run(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"outputfiles_basename": "missing-"}))
    with pytest.raises(FileNotFoundError, match="Recorded MultiNest chain"):
        DataContainer([tmp_path])


@pytest.mark.parametrize("k", [[.2, .1], [0., .2], [.1, np.nan]])
def test_invalid_k_metadata_rejected(tmp_path, k):
    write_run(tmp_path)
    np.savetxt(tmp_path / "k-vals.txt", k)
    with pytest.raises(ValueError, match="physical wavenumbers"):
        DataContainer([tmp_path])


def test_degenerate_last_bin_edges_from_setup_are_readable(tmp_path):
    write_run(tmp_path, k=(.1,))
    np.savetxt(tmp_path / "k-vals-bins.txt", [.1, .1])
    data = DataContainer([tmp_path])
    fig, _ = plot_power_spectra(data, plot_priors=True)
    fig.canvas.draw()


def test_existing_axes_can_be_customised(tmp_path):
    data = DataContainer([write_run(tmp_path, k=(.1,))])
    fig, axes = plt.subplots(1, 2)
    returned_fig, ax = plot_power_spectra(data, ax=axes[0], colors=["red"])
    assert returned_fig is fig and ax is axes[0]
    returned_fig, post_axes = plot_posteriors(data, axes=axes[1], colors=["red"])
    assert returned_fig is fig and post_axes[0] is axes[1]
    ax.set_title("Custom title")
    fig.canvas.draw()


def test_missing_k_metadata_is_not_replaced_by_bin_indices(tmp_path):
    np.savetxt(tmp_path / "data-.txt", [[1., 6., 1.]])
    with pytest.raises(FileNotFoundError, match="k-vals.txt"):
        DataContainer([tmp_path])


def test_analytic_run_has_clear_error(tmp_path):
    (tmp_path / "summary.json").write_text('{"sampler": "analytic", "logZ": 1.0}')
    with pytest.raises(ValueError, match="no power-spectrum posterior"):
        DataContainer([tmp_path])


@pytest.mark.parametrize("chain", [
    [[-1., 2., 0.]], [[0., 2., 0.]], [[1., np.nan, 0.]],
    [[1., 2., 400.]], [[1., 2., -400.]], [[1., 2., 0., 1.]],
])
def test_invalid_chain_rejected(tmp_path, chain):
    with pytest.raises(ValueError):
        DataContainer([write_run(tmp_path, k=(.1,), chain=chain)])


@pytest.mark.parametrize("options", [
    {"labels": []}, {"cred_intervals": 101}, {"uplim_quantile": 2},
    {"expected_ps": 1., "expected_dmps": 2.}, {"expected_dmps": -1.},
])
def test_invalid_options_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        DataContainer([write_run(tmp_path)], **options)


def test_plot_validation(tmp_path):
    data = DataContainer([write_run(tmp_path)])
    with pytest.raises(ValueError, match="masks must match"):
        plot_power_spectra(data, uplim_inds=[True])
    with pytest.raises(ValueError, match="positive integer"):
        plot_posteriors(data, bins=0)
    data.args[0] = {}
    with pytest.raises(ValueError, match="Prior shading requires"):
        plot_posteriors(data, plot_priors=True)


def test_prior_boxes_use_each_bins_saved_bounds(tmp_path):
    from matplotlib.patches import Rectangle

    data = DataContainer([write_run(tmp_path)])
    data.args[0]["priors"] = [[-1., 2.], [0., 3.]]
    fig, ax = plot_power_spectra(data, plot_priors=True)
    boxes = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    assert len(boxes) == 2
    for column, (box, lo, hi) in enumerate(zip(boxes, [.1, 1.], [100., 1000.])):
        assert box.get_y() == pytest.approx(lo)
        assert box.get_height() == pytest.approx(hi-lo)
        assert box.get_x() == pytest.approx(data.k_vals_bins[0][column])
        assert box.get_width() == pytest.approx(np.diff(data.k_vals_bins[0])[column])
    fig.canvas.draw()


def test_pipeline_writes_summaries_and_titled_figures(tmp_path, monkeypatch):
    from bayeseor_jax.analyze import save_analysis
    from matplotlib.figure import Figure

    titles = []
    original_savefig = Figure.savefig
    def capture_savefig(fig, *args, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_box = fig._suptitle.get_window_extent(renderer)
        assert all(title_box.y0 > ax.get_window_extent(renderer).y1 for ax in fig.axes)
        titles.append(fig._suptitle.get_text())
        return original_savefig(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", capture_savefig)
    sampler = write_run(tmp_path / "nested_sampling")
    before = (sampler / 'data-.txt').read_bytes()
    summary = save_analysis(sampler, output_dir=tmp_path / "plots", label="21-cm only")
    np.testing.assert_allclose(summary["medians"], np.sqrt(10.))
    assert (tmp_path / "plots" / "posterior_summary.json").exists()
    assert len(list((tmp_path / "plots").glob("*.png"))) == 3
    assert len(list((tmp_path / "plots").glob("*.pdf"))) == 3
    assert len(titles) == 6 and all("21-cm only" in title for title in titles)
    assert (sampler / 'data-.txt').read_bytes() == before


def test_separate_setup_metadata_and_example(tmp_path):
    from bayeseor_jax.analyze import save_analysis
    run = write_run(tmp_path / "multinest")
    metadata = tmp_path / "setup"
    metadata.mkdir()
    for name in ("k-vals.txt", "k-vals-bins.txt", "args.json"):
        (run / name).rename(metadata / name)
    data = DataContainer([run], metadata_dirs=metadata)
    np.testing.assert_allclose(data.k_vals[0], [.1, .2])
    assert data.args[0]["priors"] == [[-1., 2.]] * 2
    summary = save_analysis(run, output_dir=tmp_path / "plots", metadata_dir=metadata)
    np.testing.assert_allclose(summary["medians"], data.medians[0])
    assert not (run / "args.json").exists()
    with pytest.raises(ValueError, match="metadata_dirs"):
        DataContainer([run], metadata_dirs=[])


def test_per_run_metadata_overrides_local_metadata(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", k=(.3, .4))
    data = DataContainer([a, b], metadata_dirs=[b, a])
    np.testing.assert_allclose(data.k_vals[0], [.3, .4])
    np.testing.assert_allclose(data.k_vals[1], [.1, .2])


def test_roundoff_does_not_prevent_overlay(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", k=(.1 + 6e-15, .2 + 6e-15))
    data = DataContainer([a, b])
    assert data.k_vals_identical and data.bins_correspond
    fig, _ = plot_power_spectra_and_posteriors(data)
    fig.canvas.draw()


def test_different_mean_k_requires_explicit_correspondence(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", k=(.1003, .2002))
    data = DataContainer([a, b])
    assert not data.bins_correspond
    with pytest.raises(ValueError, match="bins_correspond"):
        plot_posteriors(data)
    matched = DataContainer([a, b], bins_correspond=True)
    fig, axes = plot_power_spectra_and_posteriors(matched)
    assert r"\simeq" in axes[1][0].texts[0].get_text()
    fig.canvas.draw()
    np.testing.assert_array_equal(matched.medians[1], data.medians[1])
    np.testing.assert_array_equal(matched.k_vals[1], data.k_vals[1])
    c = write_run(tmp_path / "c", k=(.1,))
    with pytest.raises(ValueError, match="same count"):
        DataContainer([a, c], bins_correspond=True)


def test_publication_layout_and_peak_normalisation(tmp_path):
    k = np.arange(1, 10) / 10
    data = DataContainer([write_run(tmp_path, k=k)], expected_ps=1e4)
    before = matplotlib.rcParams.copy()
    fig, (power, axes) = plot_power_spectra_and_posteriors(data, plot_priors=True)
    np.testing.assert_allclose(fig.get_size_inches(), [7., 2.55])
    assert axes.shape == (9,)
    assert power.get_legend() is not None and axes[0].get_legend() is None
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[-1])
    assert not axes[0].get_xticklabels()
    for ax in axes:
        histogram = next(p for p in ax.patches if isinstance(p, matplotlib.patches.StepPatch))
        assert histogram.get_data().values.max() == 1.
        np.testing.assert_allclose(ax.get_ylim(), [0., 1.05])
        np.testing.assert_allclose(ax.get_xlim(), axes[0].get_xlim())
    for key in ("font.family", "font.size", "mathtext.fontset", "axes.linewidth", "axes.prop_cycle"):
        assert matplotlib.rcParams[key] == before[key]
    fig.canvas.draw()
    with pytest.raises(ValueError, match="normalisation"):
        plot_posteriors(data, normalisation="unknown")


def test_packaged_style_preserves_run_colours_and_markers(tmp_path):
    data = DataContainer([write_run(tmp_path / str(i)) for i in range(6)])
    fig, (power, axes) = plot_power_spectra_and_posteriors(data)
    colours = ["#153b5c", "#2b83ba", "#6f5596", "#d6604d", "#8b1e2d", "#153b5c"]
    markers = ["s", "^", "D", "v", "P", "s"]
    for container, colour, marker in zip(power.containers, colours, markers):
        line = container.lines[0]
        assert matplotlib.colors.to_hex(line.get_color()) == colour
        assert line.get_marker() == marker
    for patch, colour in zip(axes[0].patches, colours):
        assert matplotlib.colors.to_hex(patch.get_edgecolor()) == colour
    assert power.xaxis.label.get_fontsize() == 9.9
    assert power.xaxis.label.get_fontfamily() == ["serif"]
    fig.canvas.draw()
