#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=dlinear_gla2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=/u/tliao2/TrafficFM/log/linear/dlinear/gla/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/linear/dlinear/gla/2018/%j.err

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_CONFIG=configs/mamba_train.yaml \
uv run --project fm/moirai python -m linear.dlinear \
    --dataset gla \
    --year 2018 \
    --context_length 672 \
    --kernel_size 25 \
    --windows_per_sensor 500 \
    --bs 256 \
    --num_workers 8 \
    --lrate 1e-3 \
    --max_epochs 50 \
    --patience 15 \
    --log_dir /u/tliao2/TrafficFM/benchmark/experiments/linear/dlinear/GLA/2018/ \
    --force_retrain
