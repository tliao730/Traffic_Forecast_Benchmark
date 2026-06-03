#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=timesfm_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=%j_timesfm_sd_2018.out
#SBATCH --error=%j_timesfm_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/sd_2018_finetune.yaml \
uv run --project fm/timesfm python -m fm.timesfm.timesfm
