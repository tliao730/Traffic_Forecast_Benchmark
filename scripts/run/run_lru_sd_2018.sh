#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=lru_2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=/u/tliao2/TrafficFM/log/mamba/%j_run_lru_sd_2018.out
#SBATCH --error=/u/tliao2/TrafficFM/log/mamba/%j_run_lru_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/mamba_train.yaml \
uv run --project fm/moirai python fm/mamba/lru.py \
    --dataset sd \
    --year 2018 \
    --context_length 96 \
    --windows_per_sensor 3000 \
    --d_model 64 \
    --d_state 16 \
    --num_layers 2 \
    --bs 256 \
    --lrate 1e-3 \
    --max_epochs 50 \
    --patience 15 \
    --log_dir /u/tliao2/TrafficFM/benchmark/experiments/ssm_bench/lru/SD/2018/
    # --force_retrain
