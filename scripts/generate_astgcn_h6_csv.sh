#!/bin/bash
# Generate ASTGCN predictions for horizon=6 with last 20 windows and export to CSV

set -e

cd /home/defu/workspace/LargeST_pretrain

echo "Step 1: Running ASTGCN test with last 20 windows and horizon=6..."
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
  --pred_out outputs/astgcn_sd_pred_h6.h5 \
  --pred_horizon 6 \
  --num_windows 20 \
  --bs 32

echo ""
echo "Step 2: Converting H5 to CSV for horizon 1, 2, 3, 4, 5, 6..."
for h in 1 2 3 4 5 6; do
  echo "  Processing horizon ${h}..."
  python scripts/export_pred_h5_to_short_csv.py \
    --h5 outputs/astgcn_sd_pred_h6.h5 \
    --out outputs/astgcn_sd_h6_short_h${h}.csv \
    --horizon ${h} \
    --ds_config "sd/2019/short" \
    --chunk 200
done

echo ""
echo "Step 3: Combining all horizons into single CSV..."
# Combine all horizons (horizon 6 is the target, but we include all for analysis)
python scripts/export_pred_h5_to_short_csv.py \
  --h5 outputs/astgcn_sd_pred_h6.h5 \
  --out outputs/astgcn_sd_h6_short.csv \
  --horizon 6 \
  --ds_config "sd/2019/short" \
  --chunk 200

echo ""
echo "Step 4: Analyzing predictions..."
python scripts/analyze_predictions.py \
  --csv outputs/astgcn_sd_h6_short.csv \
  --output-dir outputs/astgcn_h6 \
  --top-k 20

echo ""
echo "Step 5: Generating Time-of-Day Difficulty Profiling (4 plots)..."
python scripts/plot_tod_profiling.py \
  --pred-csv outputs/astgcn_sd_h6_short.csv \
  --output outputs/astgcn_h6/tod_difficulty_profiling.pdf \
  --save-csv \
  --outlier-threshold 3.0

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - Prediction H5: outputs/astgcn_sd_pred_h6.h5"
echo "  - Prediction CSV: outputs/astgcn_sd_h6_short.csv"
echo "  - Analysis results: outputs/astgcn_h6/"
echo "    - time_mean_abs_error.csv/png"
echo "    - hourly_mean_abs_error.csv/png"
echo "    - id_mean_abs_error.csv"
echo "    - top_20_hardest_ids.csv"
echo "    - tod_mean_abs_error.csv/png"
echo "    - tod_difficulty_profiling.pdf/csv (4-panel time-of-day analysis)"
echo ""
