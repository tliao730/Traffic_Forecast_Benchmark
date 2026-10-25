#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=timesfm_gla2019
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=/u/tliao2/TrafficFM/log/fm/timesfm/gla/2019/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/fm/timesfm/gla/2019/%j.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/fm_gla_2019.yaml \
HF_HOME=/u/tliao2/.cache/huggingface \
TRANSFORMERS_CACHE=/u/tliao2/.cache/huggingface/hub \
HF_HUB_CACHE=/u/tliao2/.cache/huggingface/hub \
uv run --project fm/timesfm python -m fm.timesfm.timesfm
