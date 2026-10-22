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
    fig, axes = plot_posteriors(data, bins=3)
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
        with pytest.raises(ValueError, match="identical physical k grids"):
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


def test_example_writes_summaries_and_figures(tmp_path):
    from test_examples.run_analysis_example import run_output_analysis
    sampler = write_run(tmp_path / "nested_sampling")
    summary = run_output_analysis(sampler, output_dir=tmp_path)
    np.testing.assert_allclose(summary["medians"], np.sqrt(10.))
    assert (tmp_path / "analysis" / "posterior_summary.json").exists()
    assert len(list((tmp_path / "analysis").glob("*.png"))) == 3
