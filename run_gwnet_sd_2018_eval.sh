#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --job-name=gwnet_eval_sd_2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=%j_gwnet_eval.out
#SBATCH --error=%j_gwnet_eval.err

cd /u/tliao2/TrafficFM/benchmark/gnn

uv run python gwnet.py \
    --device cuda \
    --dataset SD \
    --years 2018 \
    --model_name GWNET \
    --mode test \
    --test_split 0.1 \
    --test_stride 12 \
    --test_num_windows 20 \
    --log_dir /scratch/bcqc/tliao2/TrafficFM/experiments/GWNET/sd/2018/
