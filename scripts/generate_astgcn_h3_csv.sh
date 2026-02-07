#!/bin/bash
# Generate ASTGCN predictions for horizon=3 with last 20 windows and export to CSV

set -e

cd /home/defu/workspace/LargeST_pretrain

echo "Step 1: Running ASTGCN test with last 20 windows and horizon=3..."
python experiments/astgcn/main.py \
  --device cuda:0 \
  --dataset SD \
  --years 2019 \
  --seq_len 48 \
  --horizon 12 \
  --input_dim 3 \
  --output_dim 1 \
  --mode test \
  --log_dir experiments/astgcn/SD_48_12 \
  --save_pred \
  --pred_out outputs/astgcn_sd_pred_h3.h5 \
  --pred_horizon 3 \
  --num_windows 20

echo ""
echo "Step 2: Converting H5 to CSV for horizon 1, 2, 3..."
for h in 1 2 3; do
  echo "  Processing horizon ${h}..."
  python scripts/export_pred_h5_to_short_csv.py \
    --h5 outputs/astgcn_sd_pred_h3.h5 \
    --out outputs/astgcn_sd_h3_short_h${h}.csv \
    --horizon ${h} \
    --ds_config "sd/2019/short" \
    --chunk 200
done

echo ""
echo "Step 3: Combining all horizons into single CSV..."
# Combine all horizons (horizon 3 is the target, but we include all for analysis)
python scripts/export_pred_h5_to_short_csv.py \
  --h5 outputs/astgcn_sd_pred_h3.h5 \
  --out outputs/astgcn_sd_h3_short.csv \
  --horizon 3 \
  --ds_config "sd/2019/short" \
  --chunk 200

echo ""
echo "Done! Output saved to: outputs/astgcn_sd_h3_short.csv"
echo ""
echo "To analyze, run:"
echo "  python scripts/analyze_predictions.py --csv outputs/astgcn_sd_h3_short.csv --output-dir outputs/astgcn"
