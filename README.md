# BayesEoR-JAX

**Navigation:** [README](README.md) · [Installation](docs/installation.md) · [CSD3](docs/csd3.md) · [Basics](docs/basics.md) · [Forward Model](docs/forward-model.md) · [Inference](docs/inference.md) · [References](docs/references.md)

BayesEoR-JAX is a JAX-compatible implementation and extension of [BayesEoR](https://github.com/PSims/BayesEoR), a Bayesian framework for estimating the redshifted 21-cm Epoch of Reionization power spectrum from interferometric visibilities.

The package keeps the BayesEoR modelling philosophy: jointly forward model the cosmological signal, spherical image-domain foreground emission, instrumental response, and noise, then infer the posterior distribution of the dimensionless power spectrum, `Delta^2(k)`. The JAX implementation targets hardware-accelerated, differentiable, and multi-device execution, with native JAXMg linear algebra used by the marginalised likelihood path.

## What This Version Adds

BayesEoR-JAX is not a drop-in documentation refresh of the original code. It is a downstream implementation focused on:

- JAX/XLA execution for matrix construction, likelihood evaluation, and sampler integration.
- A JAXMg-backed marginalised likelihood using GPU-native dense solves and log determinants.
- MultiNest as the default nested-sampling workflow, with BlackJAX retained as an experimental backend.
- A spherical image-domain foreground model propagated through the full beam and fringe response.
- A configuration-first workflow using YAML files and `jsonargparse`.
- Test datasets for EoR-only and EoR-plus-foreground runs.

The Python package is installed as `bayeseor-jax` and imported as `bayeseor_jax`.

## Documentation

- [Installation](docs/installation.md): install paths for core development, supported JAXMg wheels, source builds, JAX checks, and MultiNest setup.
- [CSD3 Getting Started](docs/csd3.md): CSD3 environment setup, MultiNest library setup, and one-GPU Slurm scripts for the test analyses.
- [Basics](docs/basics.md): the scientific and workflow overview for new users.
- [Forward Model](docs/forward-model.md): how visibility data are modelled through the linear operator `T`.
- [EoR Model](docs/eor.md): k-space grids, spherical bins, priors, and `Delta^2(k)`.
- [Foregrounds](docs/foregrounds.md): spherical image-domain foreground modelling and distributed operator construction.
- [Instrument Model](docs/instrument.md): visibility inputs, beam models, time/frequency selection, and noise.
- [Inference](docs/inference.md): marginalised likelihood, MultiNest, BlackJAX nested sampling, and MAP helpers.
- [Configuration](docs/configuration.md): grouped reference for YAML and CLI parameters.
- [Outputs and Analysis](docs/outputs-analysis.md): cached matrices, sampler outputs, metadata, and plotting helpers.
- [HPC and Scaling](docs/hpc-scaling.md): JAX devices, precision, memory scaling, and cluster notes.
- [References](docs/references.md): BayesEoR papers and sampler/software references.

## Quick Start

For a supported CUDA system with a prebuilt JAXMg wheel:

```bash
pip install "bayeseor-jax[csd3] @ git+https://github.com/PSims/BayesEoR-v2.git@main"
```

or, on an ARM64 CUDA system matching the Isambard wheel:

```bash
pip install "bayeseor-jax[isambard] @ git+https://github.com/PSims/BayesEoR-v2.git@main"
```

Then check the package and JAX backend:

```bash
python -c "import jax, bayeseor_jax; print(jax.devices())"
```

For local development from a clone:

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
cd BayesEoR-v2
pip install -e ".[csd3]"
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml
```

Use `[isambard]` instead of `[csd3]` on systems matching the Isambard wheel. Unsupported systems should build JAXMg from source first; see [Installation](docs/installation.md).

For a CSD3 Slurm workflow, including example one-GPU nested-sampling jobs, see [CSD3 Getting Started](docs/csd3.md).

## Citation

Please cite the relevant BayesEoR methodology and software papers when using this package. The core references are listed in [References](docs/references.md), including Sims et al. (2016, 2017), Sims & Pober (2019), Burba, Sims & Pober (2023), and Sims, Burba & Pober (2025).

---

**Navigation:** [README](README.md) · [Installation](docs/installation.md) · [CSD3](docs/csd3.md) · [Basics](docs/basics.md) · [Forward Model](docs/forward-model.md) · [Inference](docs/inference.md) · [References](docs/references.md)
