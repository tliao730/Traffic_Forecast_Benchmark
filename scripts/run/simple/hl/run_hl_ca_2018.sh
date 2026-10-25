#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=hl_ca2018
#SBATCH --account=bcqc-delta-cpu
#SBATCH --partition=cpu
#SBATCH --output=/u/tliao2/TrafficFM/log/simple/hl/ca/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/simple/hl/ca/2018/%j.err

# Historical-Last heuristic: HL_Engine.evaluate() overrides BaseEngine and does
# not load a checkpoint, so --mode test runs inference directly (no training).
cd /u/tliao2/TrafficFM/benchmark/gnn

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

uv run python hl.py \
    --device cpu \
    --dataset CA \
    --years 2018 \
    --model_name HL \
    --input_dim 1 \
    --mode test \
    --seed 2023
