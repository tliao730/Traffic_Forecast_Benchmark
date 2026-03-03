#!/bin/bash
#SBATCH --job-name=tabpfn_ts
#SBATCH --account=bdem-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=run_tabpfn_ts_%j.out
#SBATCH --error=run_tabpfn_ts_%j.err

# Run tabpfn_ts benchmark on H200 (submit via sbatch from benchmark/)
# Usage: cd TrafficFM/benchmark && sbatch run_tabpfn_ts.sh

set -e

BENCHMARK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BENCHMARK_DIR"

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found on PATH." >&2
  exit 1
fi

echo "Started at $(date)"
echo "Working directory: $BENCHMARK_DIR"
uv run --project ../envs/tabpfn_ts python tabpfn_ts.py
echo "Finished at $(date)"
