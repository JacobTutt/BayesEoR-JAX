# Installation

Through JAX and [JAXMg](https://flatironinstitute.github.io/jaxmg/latest/),
BayesEoR scales from a single desktop computer to multi-GPU HPC clusters. The
models, likelihood and sampler are the same in every case; only the
linear-algebra backend changes.

Clone the repository, then create the environment matching your system:

```bash
git clone https://github.com/JacobTutt/BayesEoR-JAX.git
cd BayesEoR-JAX
```

**Desktop or CPU workstation**

```bash
conda env create -f environment.yml
conda activate bayeseor
```

**Linux NVIDIA GPU cluster**

```bash
conda env create -f environment-hpc.yml
conda activate bayeseor-hpc
```

!!! note

    Conda is used to install the native MultiNest library alongside its Python
    interface, PyMultiNest. `pip install pymultinest` alone installs only the
    wrapper. We recommend [Miniforge](https://github.com/conda-forge/miniforge).

## Requirements

Both environments use Python 3.12 or newer and JAX 0.11.1 or newer. On GPU clusters,
the dense linear algebra is distributed across devices with JAXMg, which is built on
NVIDIA's cuSOLVERMp, so the GPU environment is restricted to NVIDIA systems. It
installs `jaxmg[cuda12]`, which provides CUDA-enabled JAX; see the
[JAXMg installation guide](https://flatironinstitute.github.io/jaxmg/latest/install/)
for supported GPUs and drivers.
