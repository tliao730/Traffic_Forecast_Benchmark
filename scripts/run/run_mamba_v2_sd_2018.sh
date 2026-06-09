#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=mamba_v2_2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=log/mamba/%j_run_mamba_v2_sd_2018.out
#SBATCH --error=log/mamba/%j_run_mamba_v2_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/sd_2018_finetune.yaml \
uv run --project fm/moirai python fm/mamba/mamba.py \
    --year 2018 \
    --model_version 2 \
    --context_length 48 \
    --num_sensors 0 \
    --windows_per_sensor 3000 \
    --bs 64 \
    --lrate 1e-3 \
    --max_epochs 50 \
    --patience 15 \
    --wandb_project TrafficFM \
    --log_dir /u/tliao2/TrafficFM/benchmark/experiments/mamba_fm/SD/2018/
