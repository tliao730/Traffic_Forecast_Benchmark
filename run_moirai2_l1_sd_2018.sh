#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=moirai2_l1_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=%j_moirai2_l1.out
#SBATCH --error=%j_moirai2_l1.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/sd_2018_finetune.yaml \
WANDB_PROJECT="TrafficFM" \
MOIRAI2_MODEL_NAME="Moirai2-L1-SD2018" \
MOIRAI2_L1_FINETUNE=1 \
MOIRAI2_L1_CACHE=0 \
MOIRAI2_L1_MAX_EPOCHS=5 \
MOIRAI2_L1_BATCH_SIZE=32 \
MOIRAI2_L1_NUM_BATCHES_PER_EPOCH=200 \
MOIRAI2_L1_LR=1e-5 \
MOIRAI2_L1_SERIES=256 \
MOIRAI2_L1_LOSS=mae_median \
MOIRAI2_CONTEXT_LENGTH=4000 \
uv run --project fm/moirai python ../Finetune/moirai2_finetune.py
