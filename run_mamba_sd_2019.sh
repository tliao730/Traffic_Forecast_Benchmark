#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=mamba_sd
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=%j_mamba_sd_2019.out
#SBATCH --error=%j_mamba_sd_2019.err

cd /u/tliao2/TrafficFM/benchmark

uv run --project gnn python -m gnn.mamba \
    --device cuda \
    --dataset SD \
    --years 2019 \
    --model_name mamba \
    --max_epochs 100 \
    --patience 30 \
    --lrate 1e-3 \
    --max_train_samples 5000 \
    --wandb_project TrafficFM \
    --log_dir /scratch/bcqc/tliao2/TrafficFM/experiments/mamba/SD/2019/
