#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=stgcn_gba2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=/u/tliao2/TrafficFM/log/gnn/%j_run_stgcn_gba_2018.out
#SBATCH --error=/u/tliao2/TrafficFM/log/gnn/%j_run_stgcn_gba_2018.err

cd /u/tliao2/TrafficFM/benchmark/gnn

uv run python stgcn.py \
    --device cuda \
    --dataset GBA \
    --years 2018 \
    --model_name STGCN \
    --seed 2023 \
    --mode train
