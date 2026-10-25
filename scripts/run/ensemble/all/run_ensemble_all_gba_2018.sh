#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --job-name=ens_all_gba2018
#SBATCH --account=bdem-delta-cpu
#SBATCH --partition=cpu
#SBATCH --output=/u/tliao2/TrafficFM/log/ensemble/all/gba/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/ensemble/all/gba/2018/%j.err

# Pure-CPU baseline (sklearn / statsforecast): no GPU requested.
#
# One job for all three strategies. mean/median/weighted differ only in how
# they combine the base forecasts, and ml_ensemble.py caches those per
# (window, model), so the second and third strategies cost almost nothing.
# Run as three separate jobs this was ~3x the CPU time for the same numbers.
#
# Results still land in one directory per strategy:
#   $result_root/ensemble_{mean,median,weighted}_<models>[_w...]/
# Rows append and already-done datasets are skipped, so a job killed by the
# time limit can simply be resubmitted -- it resumes at the next term.
#
# --cpus-per-task=3 comes from measurement: the ensembles averaged 54% CPU
# efficiency on 4 cores (~2.2 cores actually used) across 18 jobs. --mem=6G
# keeps the memory side from dominating the charge, which is
# max(cores, mem_GB/2) on Delta; measured peak RSS is ~0.4G, and CA needs
# ~1.3G just to materialise its 172000 test windows.
cd /u/tliao2/TrafficFM/benchmark

# Confine BLAS/OpenMP to the cores SLURM actually allocated. Without this the
# libraries size their thread pools from the node's core count (128), so a
# 1-core job spawns 128 threads that thrash against each other inside the
# cgroup. sklearn's n_jobs does not control this.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS


BENCHMARK_CONFIG=configs/fm_gba_2018.yaml \
uv run --project ml python -m ml.ml_ensemble \
    --strategy all \
    --models seasonal_naive ridge linear_regression random_forest \
    --weights 0.15 0.30 0.20 0.35 \
    --n-jobs 3
