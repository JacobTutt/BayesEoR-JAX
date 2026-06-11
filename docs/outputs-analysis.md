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

These files are written by `write_analysis_metadata` during setup.

## Sampler Outputs

BlackJAX nested sampling writes:

- `nested_sampling_results.csv`
- `summary.json`

MultiNest writes its standard `data-*` products and a BayesEoR-JAX `summary.json`. The main weighted-chain file is usually `data-.txt`, though a shorter file root may be used to avoid MultiNest path-length limits.

The `summary.json` file records sampler settings, dimensions, output paths, and evidence-related quantities where available.

## Reading Results

Use `DataContainer` for posterior analysis:

```python
from bayeseor_jax import DataContainer

dc = DataContainer(
    ["outputs/nested_sampling"],
    sampler="blackjax",
    store_samples=True,
)
```

For MultiNest outputs:

```python
dc = DataContainer(
    ["outputs/nested_sampling"],
    sampler="multinest",
)
```

`DataContainer` reads k-bin metadata, sampler chains, posterior weights, and optional run metadata. It can calculate posterior summaries, credibility intervals, upper limits, and plotting inputs.

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
