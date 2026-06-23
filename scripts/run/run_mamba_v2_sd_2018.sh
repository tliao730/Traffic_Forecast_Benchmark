#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=mamba_v2_2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=/u/tliao2/TrafficFM/log/mamba/%j_run_mamba_v2_sd_2018.out
#SBATCH --error=/u/tliao2/TrafficFM/log/mamba/%j_run_mamba_v2_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/mamba_train.yaml \
uv run --project fm/moirai python fm/mamba/mamba.py \
    --dataset sd \
    --year 2018 \
    --context_length 96 \
    --num_sensors 0 \
    --windows_per_sensor 3000 \
    --bs 64 \
    --lrate 1e-3 \
    --max_epochs 50 \
    --patience 15 \
    --compress_warmup 10 \
    --compress_every 5 \
    --compress_energy 0.99 \
    --wandb_project TrafficFM \
    --log_dir /u/tliao2/TrafficFM/benchmark/experiments/mamba_fm/SD/2018/
