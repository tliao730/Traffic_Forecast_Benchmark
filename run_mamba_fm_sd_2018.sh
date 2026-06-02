#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=mamba_fm_2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=%j_mamba_fm_sd_2018.out
#SBATCH --error=%j_mamba_fm_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark

for TERM in short medium long; do
    uv run --project fm/moirai python fm/mamba/mamba.py \
        --term ${TERM} \
        --year 2018 \
        --context_length 48 \
        --num_sensors 0 \
        --windows_per_sensor 1000 \
        --bs 64 \
        --lrate 1e-3 \
        --max_epochs 50 \
        --patience 15 \
        --wandb_project TrafficFM \
        --log_dir /u/tliao2/TrafficFM/benchmark/experiments/mamba_fm/SD/2018/
done
