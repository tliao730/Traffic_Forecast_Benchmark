#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=moirai_small_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=/u/tliao2/TrafficFM/log/fm/moirai_small/sd/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/fm/moirai_small/sd/2018/%j.err

# Moirai-Small = Salesforce/moirai-1.0-R-small (the paper's "Moirai-Small"),
# distinct from Moirai-2 = Salesforce/moirai-2.0-R-small in ../moirai2/.
cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/fm_sd_2018.yaml \
HF_HOME=/u/tliao2/.cache/huggingface \
TRANSFORMERS_CACHE=/u/tliao2/.cache/huggingface/hub \
HF_HUB_CACHE=/u/tliao2/.cache/huggingface/hub \
uv run --project fm/moirai python -m fm.moirai.moirai_small
