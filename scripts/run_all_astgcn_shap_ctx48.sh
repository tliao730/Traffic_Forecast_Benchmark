#!/usr/bin/env bash
set -euo pipefail

SHARED_DATA_DIR="/home/defu/workspace/LargeST_pretrain/shared_shap_data_experiments"
OUTPUT_ROOT="/home/defu/workspace/LargeST_pretrain/interpret/astgcn_shap_shared_results_ctx48"
LOG_DIR="/home/defu/workspace/LargeST_pretrain/experiments/astgcn/SD_48_12"
DEVICE="cpu"
SEED=2023
INPUT_DIM=3
OUTPUT_DIM=1
NUM_SAMPLES=20
BACKGROUND_SIZE=5
HORIZONS="3,6,9,12"

for dir in "$SHARED_DATA_DIR"/sd_2019_15T_ctx48_pred*; do
  name=$(basename "$dir")
  pred=$(echo "$name" | sed -n 's/.*_pred\([0-9][0-9]*\)$/\1/p')
  if [[ -z "$pred" ]]; then
    continue
  fi

  output_dir="$OUTPUT_ROOT/$name"
  echo "Running ctx48 pred${pred}: $name"

  python interpret/astgcn_shap_inter.py \
    --device "$DEVICE" \
    --dataset SD \
    --years 2019 \
    --seq_len 48 \
    --horizon "$pred" \
    --input_dim "$INPUT_DIM" \
    --output_dim "$OUTPUT_DIM" \
    --log_dir "$LOG_DIR" \
    --shared_data_dir "$SHARED_DATA_DIR" \
    --shared_data_name "$name" \
    --output_dir "$output_dir" \
    --num_samples "$NUM_SAMPLES" \
    --background_size "$BACKGROUND_SIZE" \
    --horizons "$HORIZONS" \
    --seed "$SEED"

done
