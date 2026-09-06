#!/bin/bash
# Fast regression check for the GNN <-> gift_eval test-window alignment.
#
# Runs on the cpu partition with --device cpu: evaluation is one forward pass
# over 20 windows, so SD finishes in under a minute, while the GPU partitions
# sit ~2600 jobs deep. Use this after touching dataloader.py, args.py, the
# gnn environment, or anything that could break the alignment again.
#
#   sbatch scripts/analysis/verify_gnn_alignment.sh
#
# Expected: A_EXIT=0 with "aligned anchors (long): [34799, 35027], count=20";
# B_EXIT non-zero (refuses to silently use stride windows); C reaches the
# model-loading step, which then fails on the deliberate horizon mismatch --
# that failure is the proof the fallback engaged.
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=verify_gnn_align
#SBATCH --account=bcqc-delta-cpu
#SBATCH --partition=cpu
#SBATCH --output=/u/tliao2/TrafficFM/log/gnn_eval/_verify/%j.out
#SBATCH --error=/u/tliao2/TrafficFM/log/gnn_eval/_verify/%j.err

cd /u/tliao2/TrafficFM/benchmark/gnn
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

MODEL=${MODEL:-stgcn}
REGION=${REGION:-SD}
YEAR=${YEAR:-2018}
COMMON="--device cpu --dataset $REGION --years $YEAR --mode test --bs 16 --seed 2023"

echo "########## A) aligned path: must report gift_eval anchors ##########"
uv run python ${MODEL}.py $COMMON --model_name "${MODEL^^}"
echo "A_EXIT=$?"

echo ""
echo "########## B) alignment impossible: must fail, not degrade ##########"
uv run python ${MODEL}.py $COMMON --model_name "${MODEL^^}" --horizon 24
echo "B_EXIT=$?  (non-zero is correct)"

echo ""
echo "########## C) --allow_stride_fallback: must warn and continue ##########"
# Fails later, loading weights trained at horizon 12 -- reaching that point is
# what shows the fallback let the run through.
uv run python ${MODEL}.py $COMMON --model_name "${MODEL^^}" --horizon 24 \
    --allow_stride_fallback
echo "C_EXIT=$?"
