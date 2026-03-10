#!/bin/bash
#SBATCH --job-name=eval_time_exp
#SBATCH --account=bdem-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_eval_time_all_%j.out
#SBATCH --error=run_eval_time_all_%j.err

# Run eval_time for all experiments (agcrn, astgcn, d2stgnn, ...) via sbatch.
# Requires: uv on PATH (e.g. conda base). Submit from experiments/:  cd TrafficFM/experiments && sbatch run_eval_time_all.sh

set -e

EXPERIMENTS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TRAFFICFM_ROOT="$(cd "$EXPERIMENTS_DIR/.." && pwd)"
cd "$TRAFFICFM_ROOT"

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found on PATH. Add uv to PATH or source conda before running." >&2
  exit 1
fi

echo "Started at $(date)"
echo "Working directory: $TRAFFICFM_ROOT"
echo "Running: uv run python experiments/run_eval_time_all.py"
uv run python experiments/run_eval_time_all.py
echo "Finished at $(date)"
