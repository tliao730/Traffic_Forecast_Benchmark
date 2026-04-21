#!/bin/bash
#SBATCH --job-name=trafficfm_ml
#SBATCH --account=YOUR_ACCOUNT        # 改成你的 NCSA 帳號
#SBATCH --partition=gpuA100x4         # 改成你的 partition（問學長）
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ml_%j.out
#SBATCH --error=logs/ml_%j.err

# Usage: cd TrafficFM/benchmark && sbatch run_ml.sh

set -e

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BENCHMARK_DIR"

mkdir -p logs

if ! command -v uv &>/dev/null; then
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "Started at $(date)"
echo "Working directory: $BENCHMARK_DIR"

uv run --project ml python -m ml.ml_methods
uv run --project ml python -m ml.ml_ensemble

echo "Finished at $(date)"
