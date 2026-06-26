#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=sundial_gba2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=/u/tliao2/TrafficFM/log/fm/%j_run_sundial_gba_2018.out
#SBATCH --error=/u/tliao2/TrafficFM/log/fm/%j_run_sundial_gba_2018.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/fm_gba_2018.yaml \
HF_HOME=/u/tliao2/.cache/huggingface \
TRANSFORMERS_CACHE=/u/tliao2/.cache/huggingface/hub \
HF_HUB_CACHE=/u/tliao2/.cache/huggingface/hub \
uv run --project fm/sundial python -m fm.sundial.sundial
