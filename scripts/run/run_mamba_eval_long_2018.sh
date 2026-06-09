#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=mamba_eval_long
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=log/mamba/%j_run_mamba_eval_long_2018.out
#SBATCH --error=log/mamba/%j_run_mamba_eval_long_2018.err

cd /u/tliao2/TrafficFM/benchmark

uv run --project fm/moirai python fm/mamba/eval_only.py \
    --term long \
    --year 2018 \
    --context_length 48 \
    --num_sensors 0 \
    --windows_per_sensor 50 \
    --log_dir /u/tliao2/TrafficFM/benchmark/experiments/mamba_fm/SD/2018/ \
    --wandb_project TrafficFM
