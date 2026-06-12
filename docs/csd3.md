# CSD3 Getting Started

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

This is a step-by-step CSD3 setup guide for running the BayesEoR-JAX test examples. It uses a Python 3.11 conda environment, a prebuilt JAXMg wheel for a supported CUDA system, and MultiNest as the default sampler.

The Python package install provides the `pymultinest` wrapper. The native MultiNest shared library, `libmultinest.so`, still has to be built and made visible through `LD_LIBRARY_PATH`.

## 1. Choose A Working Directory

Choose a directory with enough space for the repository, conda environment, matrix caches, and sampler outputs. On CSD3 this should normally be on RDS project storage rather than the home directory.

```bash
mkdir -p /rds/user/$USER/hpc-work/BAYESEOR
cd /rds/user/$USER/hpc-work/BAYESEOR
```

All commands below assume you are inside this working directory.

## 2. Clone The Repository

```bash
git clone https://github.com/PSims/BayesEoR-v2.git
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

Create a fresh Python 3.11 environment. The examples use conda-forge by default:

```bash
module load miniconda/3

conda create -y \
  -p ./conda-env \
  -c conda-forge \
  python=3.11 pip
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
pip install -e ".[csd3]"
cd ..
```

The `csd3` extra installs the prebuilt JAXMg wheel for CSD3-like Linux x86_64 CUDA systems. The main package dependencies also install the Python `pymultinest` wrapper.

For a different supported system, use the matching extra instead of `csd3`.

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
test_examples/
```

By default the example script runs setup only. Add `--sample` to run MultiNest and then run the analysis step:

```bash
python test_examples/run_analysis_example.py \
  --config test_examples/eor/config.yaml \
  --sample \
  --sampler multinest \
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

## 9. Submit The CSD3 Slurm Examples

The Slurm scripts are thin wrappers around the example script. They set the CSD3 environment and output paths, then call `test_examples/run_analysis_example.py`. They do not duplicate the setup, sampler, or analysis code inline.

Submit the EoR-only test:

```bash
sbatch -A <project-account> --qos=<gpu-qos> test_examples/slurm/csd3_nested_eor.sbatch
```

Submit the EoR + foregrounds test:

```bash
sbatch -A <project-account> --qos=<gpu-qos> test_examples/slurm/csd3_nested_eor_fgs.sbatch
```

The account and QOS are deliberately not hardcoded in the repository scripts.

Each script requests one GPU and writes to a unique run directory:

```text
../r/eor_<job-id>/
../r/fgs_<job-id>/
```

The compact `r/`, `m`, `o`, and `ns` directory names keep MultiNest output paths short.

If your working directory is not `/rds/user/$USER/hpc-work/BAYESEOR`, pass it when submitting:

```bash
sbatch -A <project-account> --qos=<gpu-qos> \
  --export=ALL,BAYESEOR_PROJECT_ROOT=/path/to/BAYESEOR \
  test_examples/slurm/csd3_nested_eor.sbatch
```

## 10. Inspect The Results

Watch queued or running jobs:

```bash
squeue -u $USER
```

Check the MultiNest summaries after completion:

```bash
cat ../r/eor_<job-id>/o/ns/summary.json
cat ../r/fgs_<job-id>/o/ns/summary.json
```

Check the analysis outputs:

```text
../r/eor_<job-id>/o/analysis/posterior_summary.json
../r/eor_<job-id>/o/analysis/power_spectrum.png
../r/eor_<job-id>/o/analysis/posteriors.png
../r/eor_<job-id>/o/analysis/power_spectrum_and_posteriors.png
```

Repeat with `fgs_<job-id>` for the foreground run.

The setup logs should show that JAX uses a CUDA device. CPU fallback is a configuration error for production analyses.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
