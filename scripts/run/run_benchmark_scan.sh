#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --job-name=mamba_bench
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --output=log/mamba/%j_run_benchmark_scan.out
#SBATCH --error=log/mamba/%j_run_benchmark_scan.err

cd /u/tliao2/TrafficFM/benchmark

uv run --project fm/moirai python fm/mamba/benchmark_scan.py
