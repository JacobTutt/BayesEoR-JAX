# CSD3 Getting Started

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)

This page records the recommended CSD3 workflow for running the BayesEoR-JAX test analyses. It assumes a CUDA GPU job on the `ampere` partition and the prebuilt Linux x86_64 JAXMg wheel installed through the `csd3` extra.

## Project Layout

Keep the clone, conda environment, matrix caches, and sampler outputs on RDS project storage:

```bash
PROJECT_ROOT=/rds/user/$USER/hpc-work/BAYESEOR
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git clone https://github.com/PSims/BayesEoR-v2.git
```

The Slurm scripts in `examples/slurm/` default to this layout:

```text
$PROJECT_ROOT/
  BayesEoR-v2/
  conda-env/
  r/
```

If you use a different location, pass `BAYESEOR_PROJECT_ROOT`, `BAYESEOR_REPO_DIR`, or `BAYESEOR_ENV_DIR` when submitting the job.

## Conda Environment

Create a Python 3.11 environment from the CSD3 login node:

```bash
module load miniconda/3

CONDA_SAFETY_CHECKS=disabled conda create -y \
  -p "$PROJECT_ROOT/conda-env" \
  -c conda-forge --override-channels \
  python=3.11 pip

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT_ROOT/conda-env"
set -u
```

`CONDA_SAFETY_CHECKS=disabled` avoids slow file-verification behaviour that can occur on CSD3 RDS-backed environments.

Install BayesEoR-JAX from the clone with the CSD3 JAXMg wheel extra:

```bash
cd "$PROJECT_ROOT/BayesEoR-v2"
pip install -e ".[csd3]"
```

## MultiNest Native Library

The Python dependency `pymultinest` needs the native `libmultinest.so` library to be visible at import time. Build MultiNest once inside the same project area:

```bash
cd "$PROJECT_ROOT"
git clone https://github.com/farhanferoz/MultiNest.git
cmake -S MultiNest -B MultiNest/build
cmake --build MultiNest/build -j1
```

Then add a conda activation hook so the environment can find the shared library:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

MULTINEST_LIB="$PROJECT_ROOT/MultiNest/lib"

cat > "$CONDA_PREFIX/etc/conda/activate.d/multinest.sh" <<EOF
export _BAYESEOR_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$MULTINEST_LIB:\${LD_LIBRARY_PATH:-}"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/multinest.sh" <<'EOF'
export LD_LIBRARY_PATH="${_BAYESEOR_OLD_LD_LIBRARY_PATH:-}"
unset _BAYESEOR_OLD_LD_LIBRARY_PATH
EOF
```

Restart the conda environment and check the imports:

```bash
conda deactivate
conda activate "$PROJECT_ROOT/conda-env"

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

On a login node JAX may report CPU only. Check GPU visibility inside a Slurm GPU job.

## Submit Test Runs

The repository includes CSD3 example Slurm scripts:

```bash
cd "$PROJECT_ROOT/BayesEoR-v2"
sbatch -A <project-account> --qos=<gpu-qos> examples/slurm/csd3_nested_eor.sbatch
sbatch -A <project-account> --qos=<gpu-qos> examples/slurm/csd3_nested_eor_fgs.sbatch
```

Use the account and QOS assigned to your CSD3 project. These are intentionally not hardcoded in the repository scripts.

Each script requests one GPU and writes to a unique run directory:

```text
$PROJECT_ROOT/r/eor_<job-id>/
$PROJECT_ROOT/r/fgs_<job-id>/
```

The compact `r/`, `m`, `o`, and `ns` directory names keep MultiNest output paths below its 100-character limit. The scripts use the package API directly and run the default MultiNest sampler through `run_setup` and `run_nestedsampling`.

To use a non-default project location:

```bash
sbatch -A <project-account> --qos=<gpu-qos> \
  --export=ALL,BAYESEOR_PROJECT_ROOT=/path/to/project \
  examples/slurm/csd3_nested_eor.sbatch
```

## Inspect Results

Watch the jobs with:

```bash
squeue -u $USER
```

After completion, check the run summary:

```bash
cat "$PROJECT_ROOT/r/eor_<job-id>/o/ns/summary.json"
cat "$PROJECT_ROOT/r/fgs_<job-id>/o/ns/summary.json"
```

The setup logs should show that JAX uses a CUDA device. Treat CPU fallback as a configuration error for production analyses.

---

**Navigation:** [README](../README.md) · [Installation](installation.md) · [CSD3](csd3.md) · [Basics](basics.md) · [Forward Model](forward-model.md) · [Inference](inference.md) · [References](references.md)
