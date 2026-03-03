#!/bin/bash
#SBATCH --job-name=arima
#SBATCH --account=bdem-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_arima_%j.out
#SBATCH --error=run_arima_%j.err

# Run arima benchmark (submit via sbatch from benchmark/)
# Usage: cd TrafficFM/benchmark && sbatch run_arima.sh

set -e

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BENCHMARK_DIR"

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found on PATH." >&2
  exit 1
fi

echo "Started at $(date)"
echo "Working directory: $BENCHMARK_DIR"
uv run --project ../envs/arima python arima.py
echo "Finished at $(date)"
