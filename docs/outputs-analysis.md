# Outputs And Analysis

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX writes two broad classes of outputs: matrix caches from setup and posterior outputs from samplers.

## Matrix Cache

The setup path writes dense arrays under `array_save_directory`, including:

- `T.npy`
- `Ninv.npy`
- `T_Ninv_T.npy`
- `dbar.npy`
- `d_ninv_d.npy`

These files encode the forward model, noise weighting, and data products for a specific configuration. They should not be reused across runs with different data selection, model grid, beam, foreground model, or noise settings unless that reuse is intentional.

Use `clobber: true` only when you explicitly want to rebuild existing products.

## Run Metadata

The output directory records metadata needed by the analysis tools:

- `args.json`: serialized run arguments.
- `version.txt`: optional package/version marker.
- `k-vals.txt`: representative k values for compact non-empty bins.
- `k-vals-bins.txt`: k-bin edges.
- `k-vals-nsamples.txt`: number of modes in each bin.

These files are written by `write_analysis_metadata` during setup when the
cosmological model is included. Foreground-only setup writes configuration
metadata but no cosmological k-bin files.

## Sampler Outputs

MultiNest writes its standard `data-*` products and a BayesEoR-JAX `summary.json`. The main weighted-chain file is usually `data-.txt`, though a shorter file root may be used to avoid MultiNest path-length limits.

The `summary.json` file records sampler settings, dimensions, output paths, and evidence-related quantities where available.

For `model: fg`, the inference runner instead writes an analytic `summary.json`
and `map_coefficients.npy`. There is no posterior chain to analyse. Read these
files directly; the evidence uses the same fixed-normalisation convention as
the sampled likelihood (see [Inference](inference.md#evidence-normalisation)).

## Reading Results

The example script and CSD3 Slurm scripts run a small analysis step after sampling. This writes `analysis/posterior_summary.json` and diagnostic plots under the run output directory.

Use `DataContainer` directly for custom posterior analysis:

```python
from bayeseor_jax import DataContainer

dc = DataContainer(
    ["outputs/nested_sampling"],
)
```

`DataContainer`, defined in `analyze/results.py`, reads k-bin metadata, weighted
MultiNest chains and optional run metadata. It calculates means, medians, central
credible intervals and upper limits in linear mK². `samples` retain the sampled
log10 powers, `weights` contain normalised posterior weights, and
`log_likelihoods` contain `-0.5` times MultiNest's second column—not the weights
and not a joint posterior density. All are lists indexed by run.

The physical `k-vals.txt` file is required; missing scales are not replaced with
bin indices. For comparisons, pass several directories and matching `labels`.
Power-spectrum plots can compare different k grids; posterior overlays require
identical grids. An optional `expected_dmps` reference is given in mK², or
`expected_ps` in mK² Mpc³ and converted separately on each grid.

Plotting lives in `analyze/plotting.py` and returns Matplotlib figures and axes:

```python
from bayeseor_jax.analyze import (
    plot_power_spectra, plot_posteriors, plot_power_spectra_and_posteriors,
)

fig, ax = plot_power_spectra(dc, plot_priors=True)
ax.set_title("Recovered cosmological power")
fig.savefig("power_spectrum.pdf", bbox_inches="tight")

fig, axes = plot_posteriors(dc, plot_priors=True)
fig, (power_ax, posterior_axes) = plot_power_spectra_and_posteriors(dc)
```

Prior shading requires `args.json`; power-spectrum shading also requires
`k-vals-bins.txt`. Posterior histograms use shared log-power edges across runs,
with weighted densities per dex. Supply `uplim_inds` to the power-spectrum or
combined plot to display selected bins as upper limits: either a shared boolean
mask or one mask per run. The reader does not automatically classify detections.

### Changes from the previous analysis interface

- Import `DataContainer` from `bayeseor_jax` or `bayeseor_jax.analyze`, not the
  removed `analyze.analyze` module.
- Replace `dc.plot_*(...)` with `plot_*(dc, ...)`, unpacking the returned figure
  and axes. Set labels when constructing `DataContainer`; customise layout,
  legends and styles through the returned Matplotlib objects.
- Weighted-chain analysis always uses the weights. `posterior_weighted`,
  `store_samples`, histogram settings in the constructor, kurtosis, difference
  panels and speculative unit-label switches have been removed. Samples are
  always available; histogram bins are a plotting option.
- Output spectra use Δ²(k) in mK². Standalone `ps_to_dmps` and `dmps_to_ps`
  functions remain available for explicit conversions. `args` entries are
  dictionaries, and the reference arrays are stored in `expected_spectra`.
- Build directory paths directly instead of using `dir_prefix`. Upper limits
  are always calculated at `uplim_quantile`; plotting masks choose how to show
  them. Version information remains in the run's metadata files.

## Interpreting Power-Spectrum Outputs

The default output kind is the dimensionless power spectrum, `Delta^2(k)`. For each k bin, report:

- representative k value;
- posterior median or mean;
- chosen credibility interval;
- whether the bin is treated as a detection or upper limit;
- sampler and stopping settings;
- data/model configuration.

For non-detections, report upper limits using a clearly stated quantile, commonly 95 percent.

## Reproducibility Checklist

Archive these items with any scientific result:

- YAML configuration file.
- Git commit hash.
- installed JAX, JAXMg, and CUDA environment.
- `args.json`.
- `summary.json`.
- k-bin metadata files.
- sampler chain files.
- matrix cache provenance or instructions to rebuild it.

Large matrix caches do not always need to be archived permanently, but the exact configuration required to reproduce them should be.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
