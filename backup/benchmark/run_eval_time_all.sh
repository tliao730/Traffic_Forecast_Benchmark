#!/bin/bash
#SBATCH --job-name=eval_time_all
#SBATCH --account=bdem-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_eval_time_all_%j.out
#SBATCH --error=run_eval_time_all_%j.err

# Run eval_time for all foundation models (survives disconnect when submitted via sbatch).
# Requires: uv on PATH (e.g. from conda base or install to PATH).
# Must be submitted from benchmark/:  cd TrafficFM/benchmark && sbatch run_eval_time_all.sh

set -e

# Slurm runs the script from the job spool dir; use submit dir so we run from benchmark/
BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BENCHMARK_DIR"

# Optional: if uv is from conda, uncomment and set your conda path:
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate base

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found on PATH. Add uv to PATH or source conda before running." >&2
  exit 1
fi

echo "Started at $(date)"
echo "Working directory: $BENCHMARK_DIR"
echo "Running: python run_eval_time_all.py"
python run_eval_time_all.py
echo "Finished at $(date)"
