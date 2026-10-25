#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=chronos_bolt_sd2018_p500
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=/u/tliao2/TrafficFM/log/preds/chronos_bolt/sd/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/preds/chronos_bolt/sd/2018/%j.err

# Profiling dump -- as the *_preds.sh script next to it, but with
# configs/profile_sd_2018_w500.yaml: 250 windows (~31 days) instead of the
# gift_eval-capped 20 (2.5 days), long term only. Everything lands under
# result_root/<model>_prof250/. These metrics span different days than the
# benchmark tables and must not be pasted into them.

# The dumps go to /work/nvme, not the result_root on /scratch: the bcqc
# /scratch allocation is at its quota and large writes fail there even though
# it reports free space (a full-quota write killed job 22050954/22050955 after
# inference had finished). /work/nvme had 246G free. Metrics still go to
# result_root; only the ~50MB gzipped dump moves.

cd /u/tliao2/TrafficFM/benchmark

BENCHMARK_FORCE_REEVAL=1 \
BENCHMARK_CONFIG=configs/profile_sd_2018_w500.yaml \
HF_HOME=/u/tliao2/.cache/huggingface \
TRANSFORMERS_CACHE=/u/tliao2/.cache/huggingface/hub \
HF_HUB_CACHE=/u/tliao2/.cache/huggingface/hub \
uv run --project fm/moirai python -m fm.moirai.chronos_bolt --save-predictions \
    --pred-out-dir /work/nvme/bcqc/tliao2/TrafficFM/predictions/chronos_bolt_base_prof500
