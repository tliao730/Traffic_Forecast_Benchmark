#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=dcrnn_sd2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=log/gnn/%j_run_dcrnn_sd_2018.out
#SBATCH --error=log/gnn/%j_run_dcrnn_sd_2018.err

cd /u/tliao2/TrafficFM/benchmark/gnn

uv run python dcrnn.py \
    --device cuda \
    --dataset SD \
    --years 2018 \
    --model_name DCRNN \
    --seed 2023 \
    --mode train
