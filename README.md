# BayesEoR-JAX

**Navigation:** [README](README.md) · [Installation](docs/installation.md) · [CSD3](docs/csd3.md) · [Basics](docs/basics.md) · [Forward Model](docs/forward-model.md) · [Inference](docs/inference.md) · [References](docs/references.md)

BayesEoR-JAX is a JAX-compatible implementation and extension of [BayesEoR](https://github.com/PSims/BayesEoR), a Bayesian framework for estimating the redshifted 21-cm Epoch of Reionization power spectrum from interferometric visibilities.

The package keeps the BayesEoR modelling philosophy: jointly forward model the cosmological signal, spherical image-domain foreground emission, instrumental response, and noise, then infer the posterior distribution of the dimensionless power spectrum, `Delta^2(k)`. Standard JAX supports single-device CPU or GPU inference; the optional HPC installation uses JAXMG for distributed inference and GPU KL reduction.

## What This Version Adds

BayesEoR-JAX is not a drop-in documentation refresh of the original code. It is a downstream implementation focused on:

- JAX/XLA execution for matrix construction, likelihood evaluation, and sampler integration.
- A marginalised likelihood with native JAX Cholesky on one CPU or GPU and JAXMG solves and log determinants across GPUs.
- MultiNest for nested sampling, posterior recovery and Bayesian-evidence evaluation.
- A spherical image-domain foreground model propagated through the full beam and fringe response.
- A configuration-first workflow using YAML files and `jsonargparse`.
- Test datasets for EoR-only and EoR-plus-foreground runs.

The Python package is installed as `bayeseor-jax` and imported as `bayeseor_jax`.

## Documentation

The documentation uses MkDocs Material with a generated Python API reference.
For a local preview, run `python -m pip install -e ".[docs]"` followed by
`python -m mkdocs serve`. See [building and publishing the documentation](docs/documentation.md)
for the strict build check and optional versioned GitHub Pages deployment.

- [Installation](docs/installation.md): CPU and GPU environments, JAX checks, and MultiNest setup.
- [CSD3 Getting Started](docs/csd3.md): CSD3 environment setup, MultiNest library setup, and one-GPU Slurm scripts for the test analyses.
- [Basics](docs/basics.md): the scientific and workflow overview for new users.
- [Model](docs/model.md): step-by-step cosmological and foreground mappings, with diagrams, equations and implementation links.
- [KL Foreground Basis](docs/kl-foregrounds.md): prior/noise weighting, mode selection and reduced-operator construction.
- [Forward Model](docs/forward-model.md): how visibility data are modelled through the linear operator `T`.
- [EoR Model](docs/eor.md): k-space grids, spherical bins, priors, and `Delta^2(k)`.
- [Foregrounds](docs/foregrounds.md): spherical image-domain foreground modelling and distributed operator construction.
- [Instrument Model](docs/instrument.md): visibility inputs, beam models, time/frequency selection, and noise.
- [Inference](docs/inference.md): marginalised likelihood, MultiNest, and MAP helpers.
- [Configuration](docs/configuration.md): grouped reference for YAML and CLI parameters.
- [Outputs and Analysis](docs/outputs-analysis.md): cached matrices, sampler outputs, metadata, and plotting helpers.
- [HPC and Scaling](docs/hpc-scaling.md): JAX devices, precision, memory scaling, and cluster notes.
- [References](docs/references.md): BayesEoR papers and sampler/software references.

## Quick Start

Clone the repository and choose one installation. Use Miniforge/Conda so the
native MultiNest library is installed alongside its Python interface.
Both installations use Python 3.12+ and JAX 0.11.1+.

```bash
git clone https://github.com/JacobTutt/BayesEoR-JAX.git
cd BayesEoR-JAX
```

**Laptop / CPU:** no CUDA or JAXMG dependencies.

```bash
conda env create -f environment.yml
conda activate bayeseor
```

**HPC / NVIDIA GPU:** Python 3.12 and `jaxmg[cuda12]` from PyPI.

```bash
conda env create -f environment-hpc.yml
conda activate bayeseor-hpc
```

Check the installation:

```bash
python -c "import jax, bayeseor_jax, pymultinest; print(jax.devices())"
```

The HPC installation supports Linux x86_64 and aarch64 with a compatible NVIDIA
driver. See [Installation](docs/installation.md) for requirements,
existing environments and running the examples. Laptop KL
construction uses native JAX SVD; larger distributed decompositions use JAXMG.
Both produce the same reduced-operator format for inference.

For a CSD3 Slurm workflow, including example one-GPU nested-sampling jobs, see [CSD3 Getting Started](docs/csd3.md).

## Citation

Please cite the relevant BayesEoR methodology and software papers when using this package. The core references are listed in [References](docs/references.md), including Sims et al. (2016, 2017), Sims & Pober (2019), Burba, Sims & Pober (2023), and Sims, Burba & Pober (2025).

---

**Navigation:** [README](README.md) · [Installation](docs/installation.md) · [CSD3](docs/csd3.md) · [Basics](docs/basics.md) · [Forward Model](docs/forward-model.md) · [Inference](docs/inference.md) · [References](docs/references.md)
