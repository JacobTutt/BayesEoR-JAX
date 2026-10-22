# Outputs And Analysis

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

For standalone workflows, `write_log_files(config, out_dir=output_dir)` accepts
a `RunConfig` or metadata dictionary. It writes `args.json` and, when supplied,
`version.txt`, leaving existing files unchanged. No parser object is needed.

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
Power-spectrum plots can compare different k grids. Posterior overlays accept
grids that agree to numerical precision (`rtol=1e-10`), or explicitly confirmed
corresponding bins as described below. An optional `expected_dmps` reference is given in mK², or
`expected_ps` in mK² Mpc³ and converted separately on each grid.

### Separate setup metadata and field-of-view sweeps

Archived Isambard sweeps place chains in `model_eor_foregrounds/multinest/`
and the corresponding priors and k metadata in `setup/`. Read them directly
without copying or modifying either directory:

```python
dc = DataContainer(
    ["run_10/model_eor_foregrounds/multinest", "run_25/model_eor_foregrounds/multinest"],
    metadata_dirs=["run_10/setup", "run_25/setup"],
    labels=["10 degrees", "25 degrees"],
    bins_correspond=True,
)
```

`metadata_dirs` accepts one shared directory or a directory per run. When omitted,
metadata is read beside each chain, as before. The example's `run_output_analysis`
also accepts `metadata_dir`; the current inference workflow supplies its setup
directory directly, rather than duplicating metadata into the sampler directory.

Set `bins_correspond=True` only after verifying that bin indices represent the
same intended spherical shells across runs. Changing the cosmological Fourier
grid can slightly change the mean k of each shell without changing that
correspondence. The flag requires equal bin counts but does not itself establish
physical equivalence, rebin samples or change the stored k values. Overlay panels
then show approximate k labels using the first run's means; the spectrum panel
still uses each run's own k values. Different numbers of bins remain unsupported
in posterior overlays.

Plotting lives in `analyze/plotting.py` and returns Matplotlib figures and axes:

Fonts, tick styling, colours and markers are defined in the packaged
`analyze/bayeseor.mplstyle` file. The plotting helpers apply it locally without
changing Matplotlib's global defaults.

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

The defaults follow the compact publication layout: a seven-inch-wide combined
figure, serif typography, a consistent colour/marker palette, small display-only
offsets between power-spectrum constraints, and tightly stacked posterior panels
with a shared logarithmic axis. The combined legend sits in the lower right of
the spectrum panel. Styling is scoped to these plots and does not change global
Matplotlib settings; Times New Roman falls back to installed serif fonts.

Prior shading requires `args.json`; power-spectrum shading also requires
`k-vals-bins.txt`. Posterior histograms use shared log-power edges across runs,
spanning the displayed prior support and all positive-weight samples. By default
each curve is **peak-normalised**, for comparing shapes as in the publication
figures; its height is not a probability density. Pass `normalisation="density"`
to the posterior or combined plotting function for unit-integral densities per
dex. Neither choice changes weighted medians, intervals or upper limits. No
run-specific manual curve scaling is applied.

Supply `uplim_inds` to the power-spectrum or
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
