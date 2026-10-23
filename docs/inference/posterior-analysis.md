# Posterior analysis

After sampling, the weighted MultiNest chains give the posterior on
$\Delta^2_{21}(k)$ in each bin. `DataContainer` reads them together with the setup
metadata and summarises each bin, while the plotting helpers produce the standard
power-spectrum and posterior figures.

## Output files

Setup and sampling write the following to the run's output directories.

`summary.json`
:   The sampler settings, dimensions, output paths and evidence, `logZ` and
    `logZerr`.

`data-*.txt`
:   The standard MultiNest products. The weighted chain is usually `data-.txt`;
    a shorter file root may be used to stay within MultiNest's path-length limit.

`args.json`
:   The full run configuration, used for prior ranges when plotting.

`k-vals.txt`, `k-vals-bins.txt`, `k-vals-nsamples.txt`
:   The representative $k$, edges and number of modes of each
    [power-spectrum bin](power-spectrum.md).

    - **Note:** only written when the model includes the EoR.

`map_coefficients.npy`
:   The MAP foreground coefficients, written instead of a chain for `model: fg`.

## Saving the standard analysis

The example pipeline calls `save_analysis` automatically after sampling. It
uses the same weighted statistics and plotting functions described below,
writing a JSON summary and a single titled PNG combining the power spectrum
and marginal posteriors.

Colours default to blue for 21-cm-only models and red for joint 21-cm +
foreground models, using the saved model configuration. Pass `uplim_inds`
to `save_analysis` to display selected bins as 95% upper limits instead of
central credible intervals.

```python
from bayeseor_jax.analyze import save_analysis

save_analysis(
    "outputs/run_a/nested_sampling",
    metadata_dir="outputs/run_a",
    output_dir="outputs/run_a/plots",
    label="Cosmological signal + foregrounds",
)
```

This can also regenerate figures from completed runs without resampling.
Existing analysis files are replaced; the chains remain unchanged.

## Reading results

=== "Code"

    ```python
    from bayeseor_jax import DataContainer

    dc = DataContainer(
        ["outputs/run_a/multinest", "outputs/run_b/multinest"],
        labels=["Run A", "Run B"],
    )
    dc.medians[0]          # posterior median of each bin, in mK²
    dc.cred_intervals[0]   # {68: {"lo": ..., "hi": ...}, 95: {...}}
    dc.uplims[0]           # 95% upper limits, in mK²
    ```

=== "Attributes"

    `samples`, `weights`
    :   The sampled $\log_{10}$ powers and their normalised posterior weights.

    `avgs`, `medians`, `uplims`
    :   Weighted means, medians and upper limits of $\Delta^2_{21}(k)$, in mK².

    `cred_intervals`
    :   Central credible intervals, by percentage, then `lo` or `hi`.

    `k_vals`, `k_vals_bins`
    :   The representative $k$ and bin edges, in Mpc⁻¹.

    `summaries`, `args`
    :   Each run's `summary.json`, including its evidence, and `args.json`.

    Each is a list with one entry per run.

`DataContainer` takes the following options:

`labels`
:   One label per run, used in the plots.

    - **Default:** the directory names.

`metadata_dirs`
:   A setup directory holding `k-vals*.txt` and `args.json`, if these are not
    beside the chains.

    - **Format:** one directory shared by every run, or one per run.

`cred_intervals`, `uplim_quantile`
:   The central intervals and the upper-limit quantile.

    - **Default:** 68 and 95 per cent; 0.95.

`expected_dmps` or `expected_ps`
:   A reference spectrum to plot against, as $\Delta^2(k)$ in mK² or $P(k)$ in
    mK² Mpc³.

`bins_correspond`
:   Allows posterior overlays between runs whose bins describe the same shells but
    have slightly different mean $k$, for example after changing the Fourier grid.

    - **Default:** `false`.
    - **Note:** only set this after checking the bins correspond; it does not rebin
      the samples.

## Plotting

The plotting helpers take a `DataContainer` and return Matplotlib figures and
axes, styled for publication without changing global Matplotlib settings.

```python
from bayeseor_jax.analyze import (
    plot_power_spectra, plot_posteriors, plot_power_spectra_and_posteriors,
)

fig, ax = plot_power_spectra(dc, plot_priors=True)
fig, axes = plot_posteriors(dc, plot_priors=True)
fig, (power_ax, posterior_axes) = plot_power_spectra_and_posteriors(dc)
fig.savefig("power_spectrum.png", dpi=300, bbox_inches="tight")
```

`plot_priors`
:   Shades the prior range of each bin.

    - **Requires:** `args.json`, and `k-vals-bins.txt` for the spectrum panel.

`cred_interval`
:   The credible interval shown on the power spectrum.

    - **Default:** 68.

`uplim_inds`
:   Bins to show as upper limits rather than constraints.

    - **Format:** one boolean mask shared by every run, or one per run.
    - **Note:** bins are not classified automatically.

`normalisation`
:   How the posterior curves are scaled.

    - **Default:** `peak`, normalising each curve to a peak of one for comparing
      shapes.
    - **Alternative:** `density`, for a unit-integral density per dex.

## Reporting results

For each bin, report its representative $k$, the posterior median, the credible
interval used, and whether it is a constraint or an upper limit, with the quantile
stated. To make a result reproducible, archive:

- the configuration file and `args.json`;
- `summary.json`, the chains and the `k-vals*.txt` files;
- the Git commit and the JAX, JAXMg and CUDA versions;
- the configuration needed to rebuild the saved arrays.
