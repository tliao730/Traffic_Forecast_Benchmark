#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=flowstate_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --output=log/fm/%j_run_flowstate_sd_2018.out
#SBATCH --error=log/fm/%j_run_flowstate_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/sd_2018_finetune.yaml \
GRANITE_TSFM_PATH=/u/tliao2/TrafficFM/benchmark/fm/flowstate/granite-tsfm \
uv run --project fm/flowstate python -m fm.flowstate.flowstate
