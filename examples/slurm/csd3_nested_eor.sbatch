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
export BAYESEOR_RUN_DIR="${RUN_DIR}"

cd "${REPO_DIR}"

python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

from examples.run_analysis_example import run_output_analysis
from bayeseor_jax import (
    BayesEoRParser,
    run_nestedsampling,
    run_setup,
    write_log_files,
)


repo_dir = Path.cwd()
run_dir = Path(os.environ["BAYESEOR_RUN_DIR"])
config_path = repo_dir / "test_data" / "eor" / "config.yaml"

parser = BayesEoRParser()
args = parser.parse_args(
    [
        "--config",
        str(config_path),
        "--array-save-directory",
        str(run_dir / "m"),
        "--output-dir",
        str(run_dir / "o"),
        "--run",
    ]
)

setup_result = run_setup(**vars(args))
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
write_log_files(parser, args, out_dir=out_dir)

summary = run_nestedsampling(
    setup_result=setup_result,
    out_dir=out_dir / "ns",
    sampler="multinest",
    random_key=args.random_key,
    n_live_scale_factor=args.n_live_scale_factor,
)

print("Run summary:")
print(json.dumps(summary, indent=2))

run_output_analysis(
    sampler_dir=out_dir / "ns",
    output_dir=out_dir,
    sampler="multinest",
    label="EoR",
)
PY
