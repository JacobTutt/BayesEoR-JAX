# Installation

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Docs Index](index.md) · [Original BayesEoR](https://github.com/PSims/BayesEoR)

BayesEoR-JAX is installed as a Python package named `bayeseor-jax` and imported
as `bayeseor_jax`. The package uses JAX for accelerated numerical work and the
current likelihood path uses JAXMg for native CUDA linear algebra.

## Requirements

Use Python 3.10 or newer. Python 3.11 is recommended on the supported HPC
systems below because the current prebuilt JAXMg wheels are CPython 3.11 wheels.

The core Python package installs:

- JAX with CUDA 12 support
- NumPy 2.x with an upper bound below 2.4, SciPy, Astropy, h5py, and pyuvdata
- JAX modelling dependencies including `distrax` and `jax-healpy`
- analysis and sampling support including `anesthetic`, `matplotlib`, `tqdm`,
  `pymultinest`, and a pinned BlackJAX development revision

JAXMg is not installed by the core dependency set. Install it either through one
of the supported wheel extras below or by building JAXMg from source.

The NumPy upper bound is intentional. Some runtime dependencies currently import
Numba, and current Numba releases used in this environment reject NumPy 2.4 with
an import-time error. Keeping `numpy<2.4` avoids that resolver failure while
remaining compatible with the JAX 0.10.1 CUDA stack used by the supported JAXMg
wheels.

## Core Development Install

Clone the repository and install the package into an existing virtual
environment:

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
cd BayesEoR-v2
pip install -e .
```

This installs BayesEoR-JAX and its Python dependencies, but it does not install
JAXMg. Use this path when you need to build JAXMg yourself or when you are
developing the BayesEoR-JAX package.

## Install With A Supported JAXMg Wheel

The package provides optional extras for systems where a compatible JAXMg wheel
is hosted on GitHub. These extras install the BayesEoR-JAX package plus the
matching JAXMg wheel and its JAX CUDA runtime dependency.

For CSD3-compatible Linux x86_64 systems with CPython 3.11:

```bash
pip install "bayeseor-jax[csd3] @ git+https://github.com/PSims/BayesEoR-v2.git@main"
```

For Isambard-compatible Linux aarch64 systems with CPython 3.11:

```bash
pip install "bayeseor-jax[isambard] @ git+https://github.com/PSims/BayesEoR-v2.git@main"
```

From a local checkout, the equivalent commands are:

```bash
pip install -e ".[csd3]"
pip install -e ".[isambard]"
```

The extras currently point at the downstream JAXMg logdet wheels:

- `csd3`: `jaxmg[csd3]` from
  `https://github.com/JacobTutt/jaxmg/raw/main/wheels/csd3/jaxmg-0.0.7-cp311-cp311-linux_x86_64.whl`
- `isambard`: `jaxmg[isambard]` from
  `https://github.com/JacobTutt/jaxmg/raw/main/wheels/isambard/jaxmg-0.0.7-cp311-cp311-linux_aarch64.whl`

Both JAXMg extras install `jax[cuda12]==0.10.1`, so the BayesEoR-JAX JAX
runtime resolves to the version used to build and test the wheels.

## Build JAXMg From Source

Use a source build when no prebuilt JAXMg wheel matches your hardware, Python
version, CUDA stack, or architecture.

First install BayesEoR-JAX without a JAXMg wheel:

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
cd BayesEoR-v2
pip install -e .
```

Then build and install the downstream JAXMg branch:

```bash
git clone https://github.com/JacobTutt/jaxmg.git
cd jaxmg

mkdir build
cd build
cmake .. -DJAX_VERSION=0.10.1
cmake --build . --target install
cd ..

pip install --no-deps -e .
```

The CMake step builds native shared libraries into `src/jaxmg/cu12` or
`src/jaxmg/cu13`, depending on the CUDA toolkit used. Rebuild these libraries
whenever the JAXMg C++ or CUDA sources change.

The source build needs a working CUDA toolkit, cuDNN, cuSolverMg, CMake, and a
host compiler supported by the selected NVCC version. On some HPC systems those
paths must be supplied explicitly through modules, `CUDA_HOME`, `CUDA_ROOT`,
`CUDNN_INCLUDE_DIR`, `CUDNN_LIBRARY`, or a small compatibility symlink
directory.

## Runtime Checks

After installation, verify that JAX sees the expected GPU backend:

```bash
python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

The backend should report `gpu` and list at least one `CudaDevice` on a GPU
node. If JAX reports only CPU devices on a GPU node, check that the environment
has not been contaminated by an incompatible system CUDA module or manually
prepended CUDA library path.

You can also check that JAXMg is installed and contains native CUDA libraries:

```bash
python - <<'PY'
from pathlib import Path
import jaxmg

root = Path(jaxmg.__file__).parent
print(root)
print(sorted(p.name for p in (root / "cu12").glob("*.so")))
PY
```

## MultiNest

BayesEoR-JAX includes nested-sampling workflows that use MultiNest through
`pymultinest`. The Python wrapper is installed with the package, but the
MultiNest shared library itself is usually installed as a system or user library
on HPC systems.

If `pymultinest` cannot find `libmultinest.so`, add the directory containing
that library to `LD_LIBRARY_PATH` before running nested-sampling jobs:

```bash
export LD_LIBRARY_PATH=/path/to/MultiNest/lib:${LD_LIBRARY_PATH}
```

Keep MultiNest output path prefixes short. MultiNest has strict path-length
limits and can fail when output paths are deeply nested.

## HPC Notes

For JAX-based runs, prefer a clean Python environment with the pip-installed
JAX CUDA stack. Avoid reusing launcher scripts from the original BayesEoR MAGMA
or PyCUDA workflow unless you have checked that they do not prepend incompatible
CUDA libraries. On Isambard, loading an old NVHPC or CUDA module stack at run
time can make JAX fail its cuSPARSE probe and fall back to CPU.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [Docs Index](index.md) · [Original BayesEoR](https://github.com/PSims/BayesEoR)
