"""Diagnostic plots of weighted cosmological power-spectrum constraints."""

import matplotlib.pyplot as plt
import numpy as np


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
        One Matplotlib color per run; defaults to the standard cycle.
    ax : matplotlib.axes.Axes, optional
        Existing axes; otherwise create a new figure.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Plot objects for subsequent layout and styling changes.
    """
    if colors is not None and len(colors) != data.Ndirs:
        raise ValueError("colors must contain one entry per run")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for i, k in enumerate(data.k_vals):
        color = colors[i] if colors is not None else f"C{i % 10}"
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
        ax.errorbar(k[~upper], data.medians[i][~upper], yerr=errors[:, ~upper],
                    fmt="o", capsize=3, color=color, label=data.labels[i])
        if np.any(upper):
            limits = data.uplims[i][upper]
            ax.errorbar(k[upper], limits, yerr=limits * 0.3, uplims=True,
                        fmt="o", capsize=3, color=color)
        if data.expected_spectra[i] is not None:
            ax.plot(k, data.expected_spectra[i], "--", color=color, label=f"{data.labels[i]} reference")
        if plot_priors:
            bounds = _prior_bounds(data, i)
            if data.k_vals_bins[i] is None:
                raise ValueError("Power-spectrum prior shading requires k-vals-bins.txt")
            ax.stairs(bounds[:, 1], data.k_vals_bins[i], baseline=bounds[:, 0],
                      fill=True, color=color, alpha=0.12, zorder=0)
    ax.set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]",
           ylabel=r"$\Delta^2(k)$ [mK$^2$]")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax.figure, ax


def plot_posteriors(data, *, bins=31, plot_priors=False, colors=None, axes=None):
    """Plot weighted marginal posterior densities in logarithmic power.

    Parameters
    ----------
    data : DataContainer
        Runs with identical physical k grids, optionally including references.
    bins : int, optional
        Number of equal-width log10-power histogram bins. Compared runs use the
        same edges within each k bin. Heights are densities per dex.
    plot_priors : bool, optional
        Shade each run's prior support from args.json.
    colors : sequence of colors, optional
        One Matplotlib color per run.
    axes : sequence of matplotlib.axes.Axes, optional
        One existing axis per k bin; otherwise create a vertical stack.

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, numpy.ndarray
        Figure and a one-dimensional axes array, including for a single k bin.
    """
    if not data.k_vals_identical:
        raise ValueError("Posterior overlays require identical physical k grids")
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if colors is not None and len(colors) != data.Ndirs:
        raise ValueError("colors must contain one entry per run")
    count = data.k_vals[0].size
    if axes is None:
        _, axes = plt.subplots(count, 1, figsize=(6, max(2, count * 1.3)),
                               squeeze=False, layout="constrained")
    axes = np.asarray(axes, dtype=object).reshape(-1)
    if axes.size != count or any(ax.figure is not axes[0].figure for ax in axes):
        raise ValueError("Supply one axis per k bin, all in the same figure")
    for column, ax in enumerate(axes):
        # Shared edges make histogram heights comparable between different runs.
        support = np.concatenate([s[w > 0, column] for s, w in zip(data.samples, data.weights)])
        edges = np.histogram_bin_edges(support, bins=bins)
        for i, samples in enumerate(data.samples):
            color = colors[i] if colors is not None else f"C{i % 10}"
            density, _ = np.histogram(samples[:, column], bins=edges, weights=data.weights[i], density=True)
            ax.stairs(density, 10.0**edges, color=color, label=data.labels[i])
            reference = data.expected_spectra[i]
            if reference is not None:
                ax.axvline(reference[column], color=color, ls="--")
            if plot_priors:
                lo, hi = _prior_bounds(data, i)[column]
                ax.axvspan(lo, hi, color=color, alpha=0.12, zorder=0)
        ax.set_xscale("log")
        if column == count // 2:
            ax.set_ylabel("Density / dex")
        ax.annotate(rf"$k={data.k_vals[0][column]:.3g}\,\mathrm{{Mpc}}^{{-1}}$",
                    (0.98, 0.92), xycoords="axes fraction", ha="right", va="top")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize="small")
    axes[-1].set_xlabel(r"$\Delta^2(k)$ [mK$^2$]")
    return axes[0].figure, axes


def plot_power_spectra_and_posteriors(data, *, cred_interval=68, uplim_inds=None,
                                    plot_priors=False, colors=None, bins=31):
    """Place the power-spectrum constraints beside their marginal posteriors.

    Parameters
    ----------
    data : DataContainer
        Runs on an identical physical k grid.
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

    Returns
    -------
    fig, axes : matplotlib.figure.Figure, tuple
        Figure and (power-spectrum axis, posterior axes array).
    """
    if not data.k_vals_identical:
        raise ValueError("Combined plots require identical physical k grids")
    count = data.k_vals[0].size
    fig = plt.figure(figsize=(12, max(4, count * 1.1)), layout="constrained")
    grid = fig.add_gridspec(count, 2)
    power_ax = fig.add_subplot(grid[:, 0])
    posterior_axes = np.array([fig.add_subplot(grid[i, 1]) for i in range(count)])
    plot_power_spectra(data, cred_interval=cred_interval, uplim_inds=uplim_inds,
                      plot_priors=plot_priors, colors=colors, ax=power_ax)
    plot_posteriors(data, bins=bins, plot_priors=plot_priors, colors=colors, axes=posterior_axes)
    return fig, (power_ax, posterior_axes)
