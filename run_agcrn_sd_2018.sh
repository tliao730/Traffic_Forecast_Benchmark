#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=agcrn_sd_2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=%j_agcrn.out
#SBATCH --error=%j_agcrn.err

cd /u/tliao2/TrafficFM/benchmark

uv run --project gnn python -m gnn.agcrn \
    --device cuda \
    --dataset SD \
    --years 2018 \
    --model_name AGCRN \
    --log_dir /scratch/bcqc/tliao2/TrafficFM/experiments/AGCRN/sd/2018/
