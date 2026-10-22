# Installation

BayesEoR-JAX offers **laptop/CPU** and **HPC/NVIDIA GPU** installations. Both
use the same physical models, priors, marginalised likelihood and MultiNest
sampler; only the numerical solver and accelerator dependencies differ.

Use Miniforge (Conda with conda-forge) and work from a clone:

```bash
git clone https://github.com/JacobTutt/BayesEoR-JAX.git
cd BayesEoR-JAX
```

## 1. Normal installation: laptop or CPU workstation

```bash
conda env create -f environment.yml
conda activate bayeseor
```

This installs CPU JAX, BayesEoR-JAX, native MultiNest and PyMultiNest.
It does **not** install CUDA or JAXMG. Likelihood and MAP calculations use a
standard JAX Cholesky factorisation on a single CPU device, which can use
multiple CPU threads.

This path supports small analyses, development and plotting, including native
JAX KL construction through `T_fg_svd_build_save`. The dense SVD input, thin
singular-vector matrices and workspace must fit in local RAM. Larger distributed
decompositions require the HPC installation; both paths save the same reduced
operator and diagnostics. See [KL construction](forward-model.md#foreground-svd-and-truncation).

## 2. HPC installation: supported Linux NVIDIA systems

```bash
conda env create -f environment-hpc.yml
conda activate bayeseor-hpc
```

This uses Python 3.12 and installs `jaxmg[cuda12]` from PyPI, which supplies
compatible JAX/CUDA dependencies and the cuSOLVERMp runtime. Pip selects the
Linux x86_64 or aarch64 wheel automatically; no cluster-specific extra or
GitHub wheel URL is required. A compatible NVIDIA driver is still required.
See the [JAXMG installation guide](https://flatironinstitute.github.io/jaxmg/latest/install/)
for supported hardware and CUDA versions.

In an existing Python 3.12+ environment, use `python -m pip install -e ".[hpc]"`.
The former `[csd3]` and `[isambard]` extras have been replaced by `[hpc]`.

Likelihood and MAP solves use native JAX on one CPU or GPU and JAXMG across
multiple GPUs. Single-GPU inference alone can use
`python -m pip install -e . "jax[cuda12]"` without JAXMG; GPU KL construction
still uses JAXMG. Distributed JAXMG execution requires one Python process per
GPU and initialisation of distributed JAX before constructing the device mesh.

## Verify the installation

```bash
python -c "import jax, bayeseor_jax, pymultinest; print(jax.devices())"
```

The normal environment should report CPU. Check the HPC environment inside a
GPU allocation, where it should report GPU devices. On HPC also run:

```bash
python -c "from jaxmg import potrs, gesvd; print('JAXMG available')"
```

The likelihood selects its solver from the device mesh. A missing GPU library
raises an error, rather than silently moving the solve to CPU. To deliberately
use CPU in an HPC environment, set `JAX_PLATFORMS=cpu` before starting Python.

## Existing environments and MultiNest

The Conda files install both native MultiNest and its Python interface.
`pip install pymultinest` alone installs only the wrapper. In an existing
Conda environment use:

```bash
conda install --override-channels -c conda-forge "numpy>=2.0,<2.4" multinest pymultinest
python -m pip install -e .
```

Use `-e ".[hpc]"` for a supported GPU system. NumPy is constrained to
`>=2.0,<2.4` for the Numba-dependent dependencies.

If Miniconda's default-channel terms plugin blocks environment creation despite
`nodefaults`, use Miniforge or this explicit conda-forge-only route:

```bash
conda create -n bayeseor --override-channels -c conda-forge python=3.11 pip "numpy>=2.0,<2.4" multinest pymultinest
conda activate bayeseor
python -m pip install -e .
```

This does not require accepting unrelated channel terms. For manually compiled
MultiNest, make its library directory visible through `LD_LIBRARY_PATH` on
Linux or `DYLD_LIBRARY_PATH` on macOS. Keep MultiNest output paths short.

## Run an analysis

```bash
python test_examples/run_analysis_example.py --config test_examples/eor/config.yaml
```

This constructs example matrices without sampling. Add `--sample` to run
MultiNest and produce posterior summaries and plots. The scientific examples
can still be expensive on a laptop: CPU support does not reduce matrix sizes.

For regression tests:

```bash
python -m pip install pytest
python -m pytest -q
```
