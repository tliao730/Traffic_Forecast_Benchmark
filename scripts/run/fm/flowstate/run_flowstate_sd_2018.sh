#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=flowstate_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --output=/u/tliao2/TrafficFM/log/fm/flowstate/sd/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/fm/flowstate/sd/2018/%j.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/fm_sd_2018.yaml \
GRANITE_TSFM_PATH=/u/tliao2/TrafficFM/benchmark/fm/flowstate/granite-tsfm \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/u/tliao2/.cache/huggingface \
TRANSFORMERS_CACHE=/u/tliao2/.cache/huggingface/hub \
HF_HUB_CACHE=/u/tliao2/.cache/huggingface/hub \
FLOWSTATE_CONTEXT_LENGTH=256 \
uv run --project fm/flowstate python -m fm.flowstate.flowstate --batch-size 1
