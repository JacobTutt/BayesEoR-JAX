# CSD3 Getting Started

This is a step-by-step CSD3 setup guide for running the BayesEoR-JAX test examples. It uses a Python 3.12 conda environment, JAXMG from PyPI, and MultiNest as the default sampler.

The recommended [HPC installation](installation.md) now installs native MultiNest
through Conda as well as the Python wrapper. The manual instructions below are
an alternative for site-managed environments; skip the MultiNest build and
library-path steps when using the supplied Conda environment.

## 1. Choose A Working Directory

Choose a directory with enough space for the repository, conda environment, matrix caches, and sampler outputs. On CSD3 this should normally be on RDS project storage rather than the home directory.

```bash
mkdir -p /rds/user/$USER/hpc-work/BAYESEOR
cd /rds/user/$USER/hpc-work/BAYESEOR
```

All commands below assume you are inside this working directory.

## 2. Clone The Repository

```bash
git clone https://github.com/JacobTutt/BayesEoR-JAX.git BayesEoR-v2
```

The CSD3 examples use this layout:

```text
BAYESEOR/
  BayesEoR-v2/
  conda-env/
  MultiNest/
  r/
```

## 3. Create The Conda Environment

Create a fresh Python 3.12 environment. The examples use conda-forge by default:

```bash
module load miniconda/3

conda create -y \
  -p ./conda-env \
  -c conda-forge \
  python=3.12 pip
```

Activate it:

```bash
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./conda-env
set -u
```

## 4. Install BayesEoR-JAX

Install the package from the clone:

```bash
cd BayesEoR-v2
pip install -e ".[hpc]"
cd ..
```

The `hpc` extra installs `jaxmg[cuda12]` from PyPI and is shared across supported
Linux NVIDIA systems. The main package dependencies also install the Python
`pymultinest` wrapper.

## 5. Build The MultiNest Library

Build the native MultiNest library once in the working directory:

```bash
git clone https://github.com/farhanferoz/MultiNest.git
cmake -S MultiNest -B MultiNest/build
cmake --build MultiNest/build -j1
```

This should create:

```text
MultiNest/lib/libmultinest.so
```

If `cmake` is not available on your login shell, load the CMake module recommended for your project before running the build commands.

## 6. Make MultiNest Visible To The Environment

Add a conda activation hook so `pymultinest` can find `libmultinest.so` every time the environment is activated:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

MULTINEST_LIB="$(pwd)/MultiNest/lib"

cat > "$CONDA_PREFIX/etc/conda/activate.d/multinest.sh" <<EOF
export _BAYESEOR_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$MULTINEST_LIB:\${LD_LIBRARY_PATH:-}"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/multinest.sh" <<'EOF'
export LD_LIBRARY_PATH="${_BAYESEOR_OLD_LD_LIBRARY_PATH:-}"
unset _BAYESEOR_OLD_LD_LIBRARY_PATH
EOF
```

Restart the environment:

```bash
conda deactivate
conda activate ./conda-env
```

## 7. Check The Install

Run a login-node import check:

```bash
cd BayesEoR-v2

python - <<'PY'
import jax
import jaxmg
import bayeseor_jax
import pymultinest
print(jax.devices())
print(jax.default_backend())
print("BayesEoR-JAX import ok")
PY
```

On a login node, JAX may report CPU only. GPU visibility should be checked inside a Slurm GPU job.

## 8. Run The Example Script Directly

The test examples live in:

```text
examples/
```

By default the example script runs setup only. Add `--sample` to run MultiNest and then run the analysis step:

```bash
python examples/run_analysis_example.py \
  --config examples/eor/config.yaml \
  --sample \
  --nested-sampling-dirname ns \
  --analysis-label "EoR" \
  --array-save-directory ../r/manual_eor/m \
  --output-dir ../r/manual_eor/o
```

With `--sample`, the script runs:

```text
setup -> MultiNest nested sampling -> analysis plots and posterior summary
```

The analysis outputs are written under:

```text
../r/manual_eor/o/analysis/
```

## 9. Example Entry Points

The supplied examples are independent local runs; scheduler-specific scripts
are not included. See [Examples](examples.md) for their data, configurations
and commands. On allocated HPC resources, use the installed environment and
select the desired JAX platform explicitly.

Each run writes its evidence summary under `nested_sampling/summary.json`
and its posterior summary and plots under `analysis/` within the configured
output directory. Give independent configurations separate output and matrix
directories.
