# Basics

BayesEoR-JAX estimates the 21-cm Epoch of Reionization power spectrum from interferometric visibility data using a Bayesian forward model.

The signal of interest is faint. Foregrounds are much brighter, and the instrument mixes sky structure across angle and frequency. BayesEoR therefore models the data directly in visibility space rather than subtracting foregrounds and then estimating a power spectrum from residuals.

## Core Idea

The analysis starts from visibility data, `d`, and a model for how sky components are observed by the instrument. The main model components are:

- EoR signal coefficients, represented in a 3D Fourier-like model space.
- Spherical image-domain foreground coefficients with smooth spectral fields.
- The instrument response, including baseline sampling and the primary beam.
- A noise model.
- Power-spectrum parameters, `theta`, that control the prior covariance of the EoR coefficients.

The inference target is the posterior over power-spectrum amplitudes in spherical k bins. In most user-facing outputs this is the dimensionless power spectrum, `Delta^2(k)`.

## Workflow

A typical run has five stages:

1. Prepare a YAML configuration file.
2. Load or preprocess visibility data.
3. Build and cache the forward-model matrices.
4. Run an inference backend.
5. Analyse the posterior samples and k-bin metadata.

The central setup function is `bayeseor_jax.run_setup(config)`, where `config` is
a `RunConfig` constructed in Python or returned by `BayesEoRParser`. It returns
a `SetupResult` containing the requested configuration, matrix builder, prior,
precision model, likelihood object, visibility metadata and k-bin metadata used
by the samplers. Resolved observation and geometry inputs remain separate from
the requested settings.

Choose `model: eor`, `fg`, or `eor+fg` in the configuration to fit either component independently or jointly. The default is joint inference. With fixed foreground priors and noise, foreground-only inference instead evaluates the evidence and MAP coefficients analytically, without posterior sampling. See [model selection](inference.md#selecting-the-fitted-components).

## Example Data

The repository includes two small test datasets:

- `test_examples/eor/config.yaml`: EoR-only visibility data.
- `test_examples/eor_fgs/config.yaml`: EoR plus foreground visibility data.

Run setup for either example with:

```bash
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml
python test_examples/run_analysis_example.py --config test_examples/eor_fgs/config.yaml
```

Run nested sampling by adding `--sample`:

```bash
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml --sample
```

The setup-only mode is designed as a quick installation and configuration check. Sampling should be treated as the scientific run.

## Important Terms

`T`
: The forward-model operator mapping latent sky/model coefficients to observed visibilities.

`N`
: The visibility noise covariance. The implementation primarily works with `N^{-1}`.

`Phi`
: The prior covariance of the latent model coefficients. Its EoR block depends on the sampled power-spectrum parameters.

`T_Ninv_T`
: Cached matrix product `T^H N^{-1} T`, used by the marginalised likelihood.

`dbar`
: Cached vector product `T^H N^{-1} d`.

`Delta^2(k)`
: Dimensionless 21-cm power spectrum estimate reported in spherical k bins.

## Relation To Original BayesEoR

The original BayesEoR documentation remains valuable for the scientific model, visibility workflow, and output interpretation. BayesEoR-JAX changes the implementation backend:

- MAGMA and PyCUDA are replaced by JAX, XLA, and JAXMg.
- Forward operators are constructed in distributed CPU blocks; likelihood evaluation uses GPU linear algebra.
- The code is structured around JAX arrays, JIT-compiled kernels, and device sharding.
- The foreground model directly supports spherical, wide-field image-domain emission.

The mathematical vocabulary is intentionally similar so existing BayesEoR users can transfer concepts across.
