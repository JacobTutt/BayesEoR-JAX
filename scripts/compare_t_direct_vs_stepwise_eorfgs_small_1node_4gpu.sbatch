#!/bin/bash
#SBATCH --job-name=beor-fresh-compare-small
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/projects/u6n
#SBATCH --output=/projects/u6n/logs/%x-%j.out
#SBATCH --error=/projects/u6n/logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/benchmark-src/BayesEoR-v2-stepwise-t-overall}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/test_data/eor_fgs/config.yaml}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/test_data/eor_fgs/eor_and_fgs.uvh5}"
RUN_ROOT="${RUN_ROOT:-${PROJECTDIR:-/projects/u6n}/bayeseor_runs/fresh_compare_small_${SLURM_JOB_ID}}"
ARRAY_DIR="${ARRAY_DIR:-$RUN_ROOT/matrices}"
SUMMARY_JSON="${SUMMARY_JSON:-$RUN_ROOT/summary.json}"

mkdir -p "$ARRAY_DIR" "$(dirname "$SUMMARY_JSON")" /projects/u6n/logs

source "$HOME/miniforge3/bin/activate" bayeseor
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo "Repo:        $REPO_DIR"
echo "Config:      $CONFIG_PATH"
echo "Data path:   $DATA_PATH"
echo "Run root:    $RUN_ROOT"
echo "Array dir:   $ARRAY_DIR"
echo "Summary:     $SUMMARY_JSON"
echo "Job shape:   ${SLURM_NNODES} nodes, ${SLURM_NTASKS} tasks, ${SLURM_GPUS_ON_NODE:-4} gpus/node"
echo "Overrides:   n_u_eor=15 n_v_eor=15 fov_eor=12 n_u_fg=15 n_v_fg=15 fov_fg=12"
echo "Mode:        compare direct vs stepwise T"

cd "$REPO_DIR"

srun --cpu-bind=cores \
  python -u "$REPO_DIR/scripts/test_stepwise_t.py" \
    --mode compare_direct_vs_stepwise \
    --config "$CONFIG_PATH" \
    --data-path "$DATA_PATH" \
    --n-u-eor 15 \
    --n-v-eor 15 \
    --fov-eor 12 \
    --n-u-fg 15 \
    --n-v-fg 15 \
    --fov-fg 12 \
    --array-save-directory "$ARRAY_DIR" \
    --summary-json "$SUMMARY_JSON" \
    --direct-matrix-name T_direct \
    --stepwise-matrix-name T_stepwise
