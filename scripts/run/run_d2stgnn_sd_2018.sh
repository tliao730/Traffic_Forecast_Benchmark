#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=d2stgnn_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --output=log/gnn/%j_run_d2stgnn_sd_2018.out
#SBATCH --error=log/gnn/%j_run_d2stgnn_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark/gnn

uv run python d2stgnn.py \
    --device cuda \
    --dataset SD \
    --years 2018 \
    --model_name D2STGNN \
    --seed 2023 \
    --bs 16 \
    --mode train
