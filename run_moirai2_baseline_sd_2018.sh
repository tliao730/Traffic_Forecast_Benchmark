#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=moirai2_base_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=%j_moirai2_base.out
#SBATCH --error=%j_moirai2_base.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/sd_2018_finetune.yaml \
MOIRAI2_MODEL_NAME="Moirai2-Baseline-SD2018" \
uv run --project fm/moirai python -m fm.moirai.moirai2
