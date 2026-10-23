"""Diagnostic plots of weighted cosmological power-spectrum constraints."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
import numpy as np

from .results import DataContainer

_STYLE_PATH = Path(__file__).with_name("bayeseor.mplstyle")


def _prior_bounds(data, index):
    """Read the log10 prior bounds and convert them to linear mK² for plotting."""
    bounds = np.asarray(data.args[index].get("priors"), dtype=float)
    if bounds.shape != (data.k_vals[index].size, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Prior shading requires args.json with one ordered pair of log10 bounds per k bin")
    with np.errstate(over="ignore", under="ignore"):
        bounds = 10.0**bounds
    if not np.all(np.isfinite(bounds) & (bounds > 0)):
        raise ValueError("Prior bounds must convert to finite positive powers")
    return bounds


@style.context(_STYLE_PATH)
def plot_power_spectra(data, *, cred_interval=68, uplim_inds=None,
                      plot_priors=False, colors=None, ax=None):
    """Plot Δ²(k) medians, credible intervals and selected upper limits.

    Parameters
    ----------
    data : DataContainer
        Weighted posterior results, optionally including reference spectra.
    cred_interval : float, optional
        Stored central interval to display, in percent.
    uplim_inds : sequence of bool or sequence of sequences, optional
        True for bins displayed as upper limits. Supply one shared mask or one
        mask per run. By default all bins display central credible intervals.
    plot_priors : bool, optional
        Shade the prior support, requiring prior bounds and physical bin edges.
    colors : sequence of colors, optional
        One Matplotlib color per run; defaults to the publication palette.
    ax : matplotlib.axes.Axes, optional
        Existing axes; otherwise create a new figure.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Plot objects for subsequent layout and styling changes.

    Notes
    -----
    Compared runs receive small display-only horizontal offsets to separate
    error bars. Physical k values and calculated constraints remain unchanged.
    """
    if colors is not None and len(colors) != data.Ndirs:
        raise ValueError("colors must contain one entry per run")
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
    offsets = 10.0**np.linspace(-.025, .025, data.Ndirs) if data.Ndirs > 1 else [1.]
    shaded = []
    reference_labelled = False
    cycle = list(plt.rcParams["axes.prop_cycle"])
    for i, k in enumerate(data.k_vals):
        properties = cycle[i % len(cycle)]
        color = colors[i] if colors is not None else properties["color"]
        marker = properties["marker"]
        plotted_k = k * offsets[i]
        upper = np.zeros(k.size, dtype=bool)
        if uplim_inds is not None:
            shared = all(np.ndim(item) == 0 for item in uplim_inds)
            if not shared and len(uplim_inds) != data.Ndirs:
                raise ValueError("Supply one upper-limit mask per run")
            upper = np.asarray(uplim_inds if shared else uplim_inds[i], dtype=bool)
            if upper.shape != k.shape:
                raise ValueError("Upper-limit masks must match the corresponding k bins")
        interval = data.cred_intervals[i][cred_interval]
        errors = np.array([data.medians[i] - interval["lo"], interval["hi"] - data.medians[i]])
        ax.errorbar(plotted_k[~upper], data.medians[i][~upper], yerr=errors[:, ~upper],
                    fmt=marker, markersize=3, capsize=1.5, linewidth=.9,
                    color=color, label=data.labels[i])
        if np.any(upper):
            limits = data.uplims[i][upper]
            ax.errorbar(plotted_k[upper], limits, yerr=limits * 0.3, uplims=True,
                        fmt=marker, markersize=3, capsize=1.5, linewidth=.9, color=color)
        if data.expected_spectra[i] is not None:
            ax.plot(k, data.expected_spectra[i], "--", color="black", linewidth=1.4,
                    label=None if reference_labelled else "Reference")
            reference_labelled = True
        if plot_priors:
            bounds = _prior_bounds(data, i)
            if data.k_vals_bins[i] is None:
                raise ValueError("Power-spectrum prior shading requires k-vals-bins.txt")
            edges = data.k_vals_bins[i]
            if not any(np.array_equal(bounds, old) and np.array_equal(edges, old_edges)
                       for old, old_edges in shaded):
                for column, (lo, hi) in enumerate(bounds):
                    ax.add_patch(Rectangle(
                        (edges[column], lo), edges[column+1]-edges[column], hi-lo,
                        facecolor="0.85", edgecolor="0.55", linewidth=.6,
                        alpha=.35, zorder=0,
                        label="Prior support" if not shaded and column == 0 else None,
                    ))
                shaded.append((bounds, edges))
    ax.set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]",
           ylabel=r"$\Delta^2(k)$ [mK$^2$]")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(color="0.85", linestyle=":", linewidth=.6)
    ax.legend(loc="lower right", frameon=False, ncols=1)
    return ax.figure, ax


@style.context(_STYLE_PATH)
def plot_posteriors(data, *, bins=31, plot_priors=False, colors=None, axes=None,
                    normalisation="peak"):
    """Plot compact, stacked marginal posteriors on a shared logarithmic axis.

    Parameters
    ----------
    data : DataContainer
        Runs with matching grids or explicitly confirmed corresponding bins.
    bins : int, optional
        Number of equal-width log10-power histogram bins. Compared runs use the
        same edges within each k bin, spanning priors when shaded and all
        positive-weight samples. Curves are peak-normalised by default.
    plot_priors : bool, optional
        Shade each run's prior support from args.json.
    colors : sequence of colors, optional
        One Matplotlib color per run.
    axes : sequence of matplotlib.axes.Axes, optional
        One existing axis per k bin; otherwise create a vertical stack.
    normalisation : {"peak", "density"}, optional
        Peak normalisation compares posterior shapes, not probability heights.
        Choose density for a unit-integral density per dex. Neither changes
        the weighted samples or the reported constraints.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, numpy.ndarray
        Figure and a one-dimensional axes array, including for a single k bin.
    """
    if not data.bins_correspond:
        raise ValueError("Posterior overlays require matching grids or bins_correspond=True for verified corresponding bins")
    if normalisation not in ("peak", "density"):
        raise ValueError("normalisation must be 'peak' or 'density'")
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if colors is not None and len(colors) != data.Ndirs:
        raise ValueError("colors must contain one entry per run")
    count = data.k_vals[0].size
    if axes is None:
        fig, axes = plt.subplots(count, 1, figsize=(3.5, max(1.5, count * .283)),
                                 squeeze=False, sharex=True, gridspec_kw={"hspace": 0})
        fig.subplots_adjust(left=.17, right=.98, bottom=.18, top=.98)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    if axes.size != count or any(ax.figure is not axes[0].figure for ax in axes):
        raise ValueError("Supply one axis per k bin, all in the same figure")
    priors = [_prior_bounds(data, i) for i in range(data.Ndirs)] if plot_priors else []
    all_edges = []
    cycle = list(plt.rcParams["axes.prop_cycle"])
    for column, ax in enumerate(axes):
        # Shared edges make histogram heights comparable between different runs.
        support = np.concatenate([s[w > 0, column] for s, w in zip(data.samples, data.weights)])
        if priors:
            support = np.r_[support, np.log10([p[column] for p in priors]).ravel()]
        edges = np.histogram_bin_edges(support, bins=bins)
        all_edges.extend([edges[0], edges[-1]])
        shaded = []
        for i, samples in enumerate(data.samples):
            color = colors[i] if colors is not None else cycle[i % len(cycle)]["color"]
            density, _ = np.histogram(samples[:, column], bins=edges, weights=data.weights[i], density=True)
            height = density / density.max() if normalisation == "peak" else density
            ax.stairs(height, 10.0**edges, color=color, linewidth=1.15, label=data.labels[i])
            reference = data.expected_spectra[i]
            if reference is not None:
                ax.axvline(reference[column], color="black", ls="--", linewidth=1.)
            if plot_priors:
                lo, hi = priors[i][column]
                if (lo, hi) not in shaded:
                    ax.axvspan(lo, hi, color="0.85", alpha=.35, zorder=0)
                    shaded.append((lo, hi))
        ax.set_xscale("log")
        if normalisation == "peak":
            ax.set(ylim=(0, 1.05), yticks=[])
        else:
            ax.locator_params(axis="y", nbins=2)
            ax.tick_params(axis="y", labelsize=6)
        ax.set_ylabel(rf"$\varphi_{{{column}}}$", rotation=0, labelpad=7)
        relation = "=" if data.k_vals_identical else r"\simeq"
        ax.annotate(rf"$k{relation}{data.k_vals[0][column]:.2g}\,[\mathrm{{Mpc}}^{{-1}}]$",
                    (0.98, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=7.4)
        ax.grid(axis="x", color="0.85", linestyle=":", linewidth=.6)
        ax.tick_params(axis="x", labelbottom=column == count - 1)
    for ax in axes:
        ax.set_xlim(10.0**min(all_edges) / 1.05, 10.0**max(all_edges) * 1.1)
    if data.Ndirs > 1:
        axes[0].legend(loc="upper left", fontsize=6, frameon=False, ncols=1)
    axes[-1].set_xlabel(r"$\Delta^2(k)$ [mK$^2$]")
    return axes[0].figure, axes


@style.context(_STYLE_PATH)
def plot_power_spectra_and_posteriors(data, *, cred_interval=68, uplim_inds=None,
                                    plot_priors=False, colors=None, bins=31,
                                    normalisation="peak"):
    """Place the power-spectrum constraints beside their marginal posteriors.

    Parameters
    ----------
    data : DataContainer
        Runs with matching grids or explicitly confirmed corresponding bins.
    cred_interval : float, optional
        Central credible interval in percent.
    uplim_inds : sequence, optional
        Upper-limit masks, as accepted by plot_power_spectra.
    plot_priors : bool, optional
        Shade the stored prior bounds in both panels.
    colors : sequence of colors, optional
        One color per run, shared by both panels.
    bins : int, optional
        Number of logarithmic histogram bins per marginal posterior.
    normalisation : {"peak", "density"}, optional
        Histogram normalisation, as accepted by plot_posteriors.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, tuple
        Figure and (power-spectrum axis, posterior axes array).
    """
    if not data.bins_correspond:
        raise ValueError("Combined plots require matching grids or bins_correspond=True for verified corresponding bins")
    count = data.k_vals[0].size
    fig = plt.figure(figsize=(7., max(2.55, count * .283)))
    grid = fig.add_gridspec(1, 2, width_ratios=(.95, 1.08), left=.08,
                           right=.995, bottom=.18, top=.98, wspace=.23)
    power_ax = fig.add_subplot(grid[0])
    stack = grid[1].subgridspec(count, 1, hspace=0)
    posterior_axes = [fig.add_subplot(stack[0])]
    posterior_axes.extend(fig.add_subplot(stack[i], sharex=posterior_axes[0]) for i in range(1, count))
    posterior_axes = np.asarray(posterior_axes)
    plot_power_spectra(data, cred_interval=cred_interval, uplim_inds=uplim_inds,
                      plot_priors=plot_priors, colors=colors, ax=power_ax)
    plot_posteriors(data, bins=bins, plot_priors=plot_priors, colors=colors,
                    axes=posterior_axes, normalisation=normalisation)
    if posterior_axes[0].get_legend() is not None:
        posterior_axes[0].get_legend().remove()
    description = "Peak-normalised marginal posterior" if normalisation == "peak" else "Marginal posterior density / dex"
    fig.text(.51, .58, description, rotation=90, va="center", ha="center", fontsize=8.)
    return fig, (power_ax, posterior_axes)


@style.context(_STYLE_PATH)
def save_analysis(sampler_dir, *, output_dir, label="BayesEoR", metadata_dir=None,
                  expected_ps=None):
    """Save weighted posterior summaries and the combined analysis PNG.

    Parameters
    ----------
    sampler_dir : pathlib.Path or str
        Completed MultiNest output directory. Samples and weights are read only.
    output_dir : pathlib.Path or str
        Destination for posterior_summary.json and the combined PNG. Existing
        analysis products with these names are replaced, not sampler outputs.
    label : str, optional
        Model description included in every figure title and legend.
    metadata_dir : pathlib.Path or str, optional
        Setup directory containing args.json and k-bin metadata, if separate
        from the sampler products. Prior shading uses these saved bounds.
    expected_ps : float or array-like, optional
        Reference dimensional spectrum in mK² Mpc³, displayed without changing
        posterior constraints. Omit for observations without a known reference.

    Returns
    -------
    dict
        Weighted means, medians, credible intervals and upper limits in mK²,
        together with the representative wavenumbers and sampler path.
    """
    data = DataContainer([sampler_dir], labels=[label], metadata_dirs=metadata_dir,
                         expected_ps=expected_ps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "sampler": "multinest", "sampler_dir": str(sampler_dir),
        "k_vals": data.k_vals[0].tolist(),
        "averages": data.avgs[0].tolist(), "medians": data.medians[0].tolist(),
        "credible_intervals": {
            str(level): {side: values.tolist() for side, values in interval.items()}
            for level, interval in data.cred_intervals[0].items()
        },
        "upper_limits": data.uplims[0].tolist(),
    }
    (output_dir / "posterior_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, _ = plot_power_spectra_and_posteriors(data, plot_priors=True)
    try:
        fig.suptitle(f"21-cm power-spectrum inference\n{label}", fontsize=10, y=1.03, va="bottom")
        fig.savefig(output_dir / "power_spectrum_and_posteriors.png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    return summary
