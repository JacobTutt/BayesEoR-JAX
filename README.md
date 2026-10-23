# BayesEoR-JAX

**Navigation:** [README](README.md) · [Installation](docs/installation.md) · [Basics](docs/basics.md) · [Forward Model](docs/model/construction.md) · [Inference](docs/inference/running.md) · [Citation](#citation)

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
`python -m mkdocs serve`. Check the site with `python -m mkdocs build --strict`.

- [Installation](docs/installation.md): CPU and GPU environments, JAX checks, and MultiNest setup.
- [Basics](docs/basics.md): the scientific and workflow overview for new users.
- [Model](docs/model/index.md): step-by-step cosmological and foreground mappings, with diagrams, equations and implementation links.
- [KL Reduced Basis](docs/model/kl-reduced-basis.md): prior/noise weighting, mode selection and reduced-operator construction.
- [Forward Model](docs/model/construction.md): how visibility data are modelled through the linear operator `T`.
- [EoR Model](docs/model/eor.md): k-space grids, spherical bins, priors, and `Delta^2(k)`.
- [Foregrounds](docs/model/foregrounds.md): spherical image-domain foreground modelling and distributed operator construction.
- [Instrument Model](docs/data/instrument.md): visibility inputs, beam models, time/frequency selection, and noise.
- [Inference](docs/inference/running.md): marginalised likelihood, MultiNest, and MAP helpers.
- [Configuration](docs/examples/configuration.md): grouped reference for YAML and CLI parameters.
- [Outputs and Analysis](docs/inference/posterior-analysis.md): cached matrices, sampler outputs, metadata, and plotting helpers.

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

## Citation

Please cite the relevant BayesEoR methodology and software papers when using this
package: [Sims et al. (2016)](https://arxiv.org/abs/1607.07628),
[Sims et al. (2017)](https://arxiv.org/abs/1701.03384),
[Sims & Pober (2019)](https://arxiv.org/abs/1907.02608),
[Burba, Sims & Pober (2023)](https://arxiv.org/abs/2302.04058), and
[Sims, Burba & Pober (2025)](https://arxiv.org/abs/2501.12928).
For MultiNest, cite [Feroz, Hobson & Bridges (2009)](https://arxiv.org/abs/0809.3437).

---

**Navigation:** [README](README.md) · [Installation](docs/installation.md) · [Basics](docs/basics.md) · [Forward Model](docs/model/construction.md) · [Inference](docs/inference/running.md) · [Citation](#citation)
