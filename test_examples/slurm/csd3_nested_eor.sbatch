#!/bin/bash
#SBATCH -J bayeseor_ns_eor
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o bayeseor_nested_eor-%j.out
#SBATCH -e bayeseor_nested_eor-%j.err

set -euo pipefail

PROJECT_ROOT="${BAYESEOR_PROJECT_ROOT:-/rds/user/${USER}/hpc-work/BAYESEOR}"
REPO_DIR="${BAYESEOR_REPO_DIR:-${PROJECT_ROOT}/BayesEoR-v2}"
ENV_DIR="${BAYESEOR_ENV_DIR:-${PROJECT_ROOT}/conda-env}"
RUN_DIR="${BAYESEOR_RUN_DIR:-${PROJECT_ROOT}/r/eor_${SLURM_JOB_ID}}"

mkdir -p "${RUN_DIR}"

module load miniconda/3
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
set -u

export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${REPO_DIR}"

python test_examples/run_analysis_example.py \
  --config test_examples/eor/config.yaml \
  --sample \
  --sampler multinest \
  --nested-sampling-dirname ns \
  --analysis-label "EoR" \
  --array-save-directory "${RUN_DIR}/m" \
  --output-dir "${RUN_DIR}/o"
