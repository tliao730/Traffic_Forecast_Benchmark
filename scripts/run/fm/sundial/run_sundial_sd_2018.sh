#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=sundial_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=/u/tliao2/TrafficFM/log/fm/sundial/sd/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/fm/sundial/sd/2018/%j.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/fm_sd_2018.yaml \
HF_HOME=/u/tliao2/.cache/huggingface \
TRANSFORMERS_CACHE=/u/tliao2/.cache/huggingface/hub \
HF_HUB_CACHE=/u/tliao2/.cache/huggingface/hub \
uv run --project fm/sundial python -m fm.sundial.sundial
