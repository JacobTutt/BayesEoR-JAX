"""Read MultiNest power-spectrum samples and calculate weighted constraints."""

import json
from pathlib import Path

import numpy as np


def ps_to_dmps(power, k):
    """Convert P(k) in mK² Mpc³ to Δ²(k) in mK² at k in Mpc⁻¹.

    Parameters
    ----------
    power, k : array-like
        Power and wavenumbers with broadcast-compatible shapes.

    Returns
    -------
    numpy.ndarray
        Dimensionless power, P(k) k³ / (2π²).
    """
    return np.asarray(power) * np.asarray(k)**3 / (2 * np.pi**2)


def dmps_to_ps(power, k):
    """Convert Δ²(k) in mK² to P(k) in mK² Mpc³ at k in Mpc⁻¹.

    Parameters
    ----------
    power, k : array-like
        Power and wavenumbers with broadcast-compatible shapes.

    Returns
    -------
    numpy.ndarray
        Dimensional power, Δ²(k) 2π² / k³.
    """
    return np.asarray(power) * (2 * np.pi**2) / np.asarray(k)**3


def _weighted_quantiles(samples, quantiles, weights):
    """Interpolate each column's weighted empirical CDF at the given quantiles."""
    # Zero-weight points must not change the interpolation or its endpoints.
    keep = weights > 0
    values = samples[keep]
    weights = weights[keep]
    result = []
    for column in values.T:
        order = np.argsort(column)
        cdf = np.cumsum(weights[order]) / weights.sum()
        result.append(np.interp(quantiles, cdf, column[order]))
    return np.asarray(result).T


def _chain_path(path, summary):
    """Resolve the recorded MultiNest chain, including shortened file roots."""
    basename = summary.get("outputfiles_basename")
    if basename:
        candidate = path / (Path(basename).name + ".txt")
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Recorded MultiNest chain is missing: {candidate}")
    default = path / "data-.txt"
    if default.exists():
        return default
    candidates = sorted(path.glob("*-.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one MultiNest weighted chain in {path}; found {len(candidates)}")
    return candidates[0]


class DataContainer:
    """Load runs and summarise their weighted Δ²(k) posteriors in mK².

    Parameters
    ----------
    dirnames : sequence of str or pathlib.Path
        MultiNest output directories containing a weighted chain and k-vals.txt.
        Optional args.json supplies prior bounds; k-vals-bins.txt supplies edges.
    labels : sequence of str, optional
        One plotting label per run; defaults to directory names.
    cred_intervals : float or sequence of float, optional
        Central credible intervals in percent, defaulting to 68 and 95.
    uplim_quantile : float, optional
        One-sided upper-limit quantile, defaulting to 0.95.
    expected_ps, expected_dmps : float or array-like, optional
        Reference P(k) or Δ²(k), respectively. Supply at most one: a scalar or
        one value per bin. P(k) is converted separately on each run's k grid.

    Attributes
    ----------
    samples, weights, log_likelihoods : list of numpy.ndarray
        Log10 power samples, normalised posterior weights and log likelihoods
        from MultiNest columns 2 onward, 0 and -0.5 times column 1, respectively.
    k_vals, k_vals_bins : list
        Physical wavenumbers and optional bin edges for each run, in Mpc⁻¹.
    avgs, medians, uplims : list of numpy.ndarray
        Weighted means, medians and upper limits in linear mK².
    cred_intervals : list of dict
        Per-run interval bounds, indexed by percentage and then 'lo' or 'hi'.
    expected_spectra : list
        Reference Δ²(k) arrays, or None for each run if no reference was given.
    args, summaries : list of dict
        Optional configuration and runner metadata, including evidence values.

    Notes
    -----
    This reader always honours MultiNest weights. It does not classify bins as
    detections: callers choose which constraints to display as upper limits.
    Foreground-only analytic runs have no power-spectrum chain; read their
    summary.json and map_coefficients.npy directly instead.
    """

    def __init__(self, dirnames, *, labels=None, cred_intervals=(68, 95),
                 uplim_quantile=0.95, expected_ps=None, expected_dmps=None):
        self.dirnames = [Path(path) for path in dirnames]
        self.Ndirs = len(self.dirnames)
        if not self.Ndirs:
            raise ValueError("At least one run directory is required")
        self.labels = list(labels) if labels is not None else [p.name for p in self.dirnames]
        if len(self.labels) != self.Ndirs:
            raise ValueError("labels must contain one entry per run")
        intervals = np.atleast_1d(cred_intervals).astype(float)
        if intervals.ndim != 1 or not np.all(np.isfinite(intervals) & (intervals > 0) & (intervals < 100)):
            raise ValueError("Credible intervals must lie strictly between 0 and 100 percent")
        if not 0 < uplim_quantile < 1:
            raise ValueError("uplim_quantile must lie strictly between 0 and 1")
        if expected_ps is not None and expected_dmps is not None:
            raise ValueError("Supply expected_ps or expected_dmps, not both")
        self.uplim_quantile = uplim_quantile
        self.k_vals, self.k_vals_bins, self.args, self.summaries = [], [], [], []
        self.samples, self.weights, self.log_likelihoods = [], [], []
        self.avgs, self.medians, self.uplims, self.cred_intervals = [], [], [], []
        self.expected_spectra = []

        for path in self.dirnames:
            summary_path = path / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if summary.get("sampler") == "analytic":
                raise ValueError("Analytic foreground-only runs have no power-spectrum posterior; read summary.json directly")
            # Never label invented bin indices as physical cosmological scales.
            k = np.loadtxt(path / "k-vals.txt", ndmin=1)
            if k.ndim != 1 or not k.size or not np.all(np.isfinite(k) & (k > 0)) or np.any(np.diff(k) <= 0):
                raise ValueError("k-vals.txt must contain positive, increasing physical wavenumbers")
            edges_path = path / "k-vals-bins.txt"
            edges = np.loadtxt(edges_path, ndmin=1) if edges_path.exists() else None
            if edges is not None and (edges.shape != (k.size + 1,) or not np.all(np.isfinite(edges))
                                      or np.any(np.diff(edges) < 0) or np.any(edges < 0)
                                      or np.any(k < edges[:-1]) or np.any(k > edges[1:])):
                raise ValueError("k-vals-bins.txt must contain ordered edges enclosing each k bin")
            args_path = path / "args.json"
            args = json.loads(args_path.read_text()) if args_path.exists() else {}
            chain = np.loadtxt(_chain_path(path, summary), ndmin=2)
            if chain.shape[1] != k.size + 2 or not chain.shape[0] or not np.all(np.isfinite(chain)):
                raise ValueError("Weighted chain must contain finite weights, -2 log likelihood and one log power per k bin")
            weights = chain[:, 0]
            total = weights.sum()
            if np.any(weights < 0) or not np.isfinite(total) or total <= 0:
                raise ValueError("Posterior weights must be non-negative with a finite positive sum")
            weights = weights / total
            with np.errstate(over="ignore", under="ignore"):
                powers = 10.0**chain[:, 2:]
            if not np.all(np.isfinite(powers) & (powers > 0)):
                raise ValueError("Log-power samples must convert to finite positive powers")

            quantiles = [0.5, uplim_quantile]
            for interval in intervals:
                quantiles.extend([(1 - interval / 100) / 2, (1 + interval / 100) / 2])
            estimates = _weighted_quantiles(powers, quantiles, weights)
            self.medians.append(estimates[0])
            self.uplims.append(estimates[1])
            self.cred_intervals.append({ci: {"lo": estimates[2 + 2*i], "hi": estimates[3 + 2*i]}
                                        for i, ci in enumerate(intervals)})
            self.avgs.append(np.average(powers, axis=0, weights=weights))
            self.samples.append(chain[:, 2:])
            self.weights.append(weights)
            self.log_likelihoods.append(-0.5 * chain[:, 1])
            self.k_vals.append(k)
            self.k_vals_bins.append(edges)
            self.args.append(args)
            self.summaries.append(summary)

            reference = expected_dmps
            if expected_ps is not None:
                reference = ps_to_dmps(expected_ps, k)
            if reference is not None:
                reference = np.broadcast_to(np.asarray(reference, dtype=float), k.shape).copy()
                if not np.all(np.isfinite(reference) & (reference > 0)):
                    raise ValueError("Reference power must be finite and positive")
            self.expected_spectra.append(reference)

        self.k_vals_identical = all(np.array_equal(self.k_vals[0], k) for k in self.k_vals[1:])
