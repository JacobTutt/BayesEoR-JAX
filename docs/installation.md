# Installation

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

BayesEoR-JAX is installed as a Python package named `bayeseor-jax` and imported as `bayeseor_jax`. The package uses JAX for accelerated numerical work. The current marginalised likelihood path also uses Jacob's downstream JAXMg branch, which adds the log-determinant support needed by BayesEoR-JAX.

The original BayesEoR package used a MAGMA/PyCUDA backend. This implementation does not use that installation path. Install and debug it as a JAX CUDA package plus JAXMg.

## Requirements

The recommended runtime is:

- Python 3.11 for the prebuilt JAXMg wheels.
- A CUDA 12 compatible NVIDIA GPU environment.
- JAX with CUDA support.
- JAXMg, either from one of the hosted wheels or from a local source build.
- MultiNest native libraries for the default `multinest` sampler through `pymultinest`.

The package pins NumPy to `>=2.0,<2.4` because runtime dependencies that use Numba currently reject NumPy 2.4.

## Option 1: Supported Hardware Wheels

Use this path when your system matches one of the provided JAXMg wheels. These extras install BayesEoR-JAX plus the corresponding JAXMg wheel.

For CSD3-like Linux x86_64 CUDA systems:

```bash
pip install "bayeseor-jax[csd3] @ git+https://github.com/PSims/BayesEoR-v2.git@main"
```

For a complete CSD3 workflow with conda, MultiNest, and Slurm examples, see [CSD3 Getting Started](csd3.md).

For Isambard-like Linux ARM64 CUDA systems:

```bash
pip install "bayeseor-jax[isambard] @ git+https://github.com/PSims/BayesEoR-v2.git@main"
```

These names describe the hardware/software environment used to build the JAXMg wheel. They are not intended to mean the package only works on those named machines.

For development from a local clone:

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
cd BayesEoR-v2
pip install -e ".[csd3]"
```

or:

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
cd BayesEoR-v2
pip install -e ".[isambard]"
```

## Option 2: Build JAXMg From Source

Use this path on unsupported systems where no prebuilt JAXMg wheel is available. The Python editable install and the native CUDA build are separate steps.

```bash
git clone https://github.com/JacobTutt/jaxmg.git
cd jaxmg

pip install "jax[cuda12]"
pip install --no-deps -e .

cmake -S . -B build
cmake --build build --target install
```

Then install BayesEoR-JAX without a JAXMg wheel extra:

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
cd BayesEoR-v2
pip install -e .
```

On managed clusters you may need to load compiler, CUDA, and cuDNN modules before configuring JAXMg. On CSD3, previous successful source builds used a CUDA 12.1 stack with `gcc/11`, `cuda/12.1`, and `cudnn/8.9_cuda-12.1`. CSD3's cuDNN module also exposed versioned header names, so a local cuDNN compatibility directory was needed during that source build. Treat those as site-specific JAXMg build notes, not BayesEoR-JAX package changes.

## Verify JAX

Check that JAX sees the expected accelerator:

```bash
python - <<'PY'
import jax
jax.config.update("jax_enable_x64", True)
print(jax.devices())
print(jax.default_backend())
PY
```

For a GPU run, the backend should be `gpu`, and the device list should contain CUDA devices. If JAX reports only CPU devices, fix the JAX/CUDA environment before running BayesEoR-JAX.

## Verify JAXMg

The BayesEoR-JAX marginalised likelihood imports JAXMg through `jaxmg`. A minimal import check is:

```bash
python - <<'PY'
import jaxmg
from jaxmg import potrs
print("jaxmg import ok")
PY
```

The JAXMg fork used here includes log-determinant support required by the BayesEoR-JAX likelihood. Installing upstream JAXMg from PyPI will not provide downstream branch-specific changes unless the upstream package has since incorporated them.

## MultiNest

`pymultinest` is a Python wrapper around the native MultiNest library. Installing the Python package is not always enough; the dynamic linker must also be able to find `libmultinest`.

If `import pymultinest` fails with a missing library error, set `LD_LIBRARY_PATH` to the directory containing the MultiNest shared library before running BayesEoR-JAX:

```bash
export LD_LIBRARY_PATH=/path/to/MultiNest/lib:$LD_LIBRARY_PATH
python -c "from pymultinest.solve import solve; print('pymultinest ok')"
```

MultiNest output paths should also be kept short. Very long output paths can fail in the underlying MultiNest library.

## Quick Functional Check

From a BayesEoR-JAX clone:

```bash
python examples/run_analysis_example.py --config test_data/eor/config.yaml
python examples/run_analysis_example.py --config test_data/eor_fgs/config.yaml
```

These commands run setup only by default. Add `--sample` to run nested sampling with the default MultiNest backend and then run the example analysis step:

```bash
python examples/run_analysis_example.py --config test_data/eor/config.yaml --sample
```

Sampling is substantially more expensive than setup and should normally be run on a suitable GPU node. After sampling, the example writes posterior summaries and diagnostic plots under the output directory's `analysis/` subdirectory.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
