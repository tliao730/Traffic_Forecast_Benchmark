#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=d2stgnn_gba2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --output=/u/tliao2/TrafficFM/log/gnn/%j_run_d2stgnn_gba_2018.out
#SBATCH --error=/u/tliao2/TrafficFM/log/gnn/%j_run_d2stgnn_gba_2018.err

cd /u/tliao2/TrafficFM/benchmark/gnn

uv run python d2stgnn.py \
    --dataset GBA \
    --years 2018 \
    --model_name D2STGNN \
    --mode train
