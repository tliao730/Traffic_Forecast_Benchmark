#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=linear_ssm_ca2018
#SBATCH --account=bcqc-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --output=/u/tliao2/TrafficFM/log/ssm/linear_ssm/ca/2018/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/ssm/linear_ssm/ca/2018/%j.err

cd /u/tliao2/TrafficFM/benchmark

# --- segmented-training support (auto-resume across SLURM jobs) ------------
# Training resumes automatically from last_checkpoint_<term>_s<seed>.pt in
# log_dir. A follow-up job is queued before training starts; if this job is
# killed (timeout) the follow-up resumes from the checkpoint. On clean finish
# the done-marker is written and the follow-up is cancelled.
SCRIPT_PATH=/u/tliao2/TrafficFM/scripts/run/ssm/linear_ssm/run_linear_ssm_ca_2018.sh
DONE_MARKER=/u/tliao2/TrafficFM/benchmark/experiments/done_markers/run_linear_ssm_ca_2018.done
mkdir -p "/u/tliao2/TrafficFM/benchmark/experiments/done_markers"
if [ -f "$DONE_MARKER" ]; then
    echo "Done marker exists ($DONE_MARKER), nothing to do."
    exit 0
fi
# Only auto-resume after a timeout/node failure; if the previous attempt
# crashed (e.g. OOM) or was cancelled, retrying would fail the same way.
if [ -n "$PREV_JOB_ID" ]; then
    PREV_STATE=$(sacct -j "$PREV_JOB_ID" -X -n -o State 2>/dev/null | head -1 | xargs)
    case "$PREV_STATE" in
        ""|TIMEOUT*|NODE_FAIL*|PREEMPTED*)
            echo "Previous job $PREV_JOB_ID state: '${PREV_STATE:-unknown}' — resuming." ;;
        *)
            echo "Previous job $PREV_JOB_ID state: $PREV_STATE — not auto-resuming. Fix the issue, then resubmit this script to continue from the last checkpoint."
            exit 0 ;;
    esac
fi
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
NEXT_JOB=""
if [ -n "$SLURM_JOB_ID" ] && [ "$RESUBMIT_COUNT" -lt 10 ]; then
    NEXT_JOB=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
        --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),PREV_JOB_ID=${SLURM_JOB_ID} \
        "$SCRIPT_PATH")
    echo "Queued follow-up job $NEXT_JOB (resubmit #$((RESUBMIT_COUNT + 1)))"
fi
# ---------------------------------------------------------------------------

BENCHMARK_CONFIG=configs/mamba_train.yaml \
uv run --project fm/moirai python ssm/linear_ssm.py \
    --dataset ca \
    --year 2018 \
    --context_length 96 \
    --windows_per_sensor 500 \
    --d_model 64 \
    --d_state 16 \
    --num_layers 2 \
    --bs 256 \
    --num_workers 8 \
    --lrate 1e-3 \
    --max_epochs 50 \
    --patience 15 \
    --log_dir /u/tliao2/TrafficFM/benchmark/experiments/ssm_bench/linear_ssm/CA/2018/
EXIT_CODE=$?

# Reaching this line means SLURM did not kill the job (a timeout kills the
# whole script first). Training either finished or crashed — stop the chain
# either way; a crash (e.g. OOM) would fail the same way on retry.
if [ "$EXIT_CODE" -eq 0 ]; then
    touch "$DONE_MARKER"
    echo "Training finished."
else
    echo "Training failed with exit code $EXIT_CODE — not retrying. Resubmit this script after fixing the issue to continue from the last checkpoint."
fi
if [ -n "$NEXT_JOB" ]; then
    scancel "$NEXT_JOB"
    echo "Cancelled follow-up job $NEXT_JOB"
fi
exit "$EXIT_CODE"
