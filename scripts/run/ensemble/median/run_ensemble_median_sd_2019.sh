#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --job-name=ens_median_sd2019
#SBATCH --account=bdem-delta-cpu
#SBATCH --partition=cpu
#SBATCH --output=/u/tliao2/TrafficFM/log/ensemble/median/sd/2019/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/ensemble/median/sd/2019/%j.err

# Pure-CPU baseline (sklearn / statsforecast): no GPU requested.
# Dataset + year come from BENCHMARK_CONFIG; results append to
# $result_root/<model_name>/all_results.csv (one row per region/year,
# already-done datasets are skipped on re-run).
cd /u/tliao2/TrafficFM/benchmark

# Confine BLAS/OpenMP to the cores SLURM actually allocated. Without this the
# libraries size their thread pools from the node's core count (128), so a
# 1-core job spawns 128 threads that thrash against each other inside the
# cgroup. sklearn's n_jobs does not control this.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS


BENCHMARK_CONFIG=configs/fm_sd_2019.yaml \
uv run --project ml python -m ml.ml_ensemble --strategy median --models seasonal_naive ridge linear_regression random_forest --n-jobs 3
