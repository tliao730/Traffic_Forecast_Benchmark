# Moirai2 vs ASTGCN Analysis Guide

## Data Description

| Model | Prediction File | item_id Meaning |
|------|----------|--------------|
| Moirai2 | `results/Moirai2/predictions/sd_2019_short_predictions.csv` | Sensor ID (e.g., 1114091) |
| ASTGCN | `results/ASTGCN/predictions/astgcn_sd_sd_2019_short_predictions.csv` | Node index (0..715) |

**Note**: The two models use different test periods (Moirai2 uses gift_eval's test windows, ASTGCN uses the last 192 windows of idx_test). The comparison is on the error distribution of their respective test sets.

---

## Step 1: Analyze Each Model Separately

Run `analyze_predictions.py` on each model's prediction CSV to obtain MAE and plots by time, hour, and item.

```bash
cd /projects/bevu/jliu55/TrafficFM

# Moirai2
uv run python scripts/analyze_predictions.py \
  --csv results/Moirai2/predictions/sd_2019_short_predictions.csv \
  --output-dir results/Moirai2/analysis

# ASTGCN
uv run python scripts/analyze_predictions.py \
  --csv results/ASTGCN/predictions/astgcn_sd_sd_2019_short_predictions.csv \
  --output-dir results/ASTGCN/analysis
```

Outputs include:
- `id_mean_abs_error.csv`: mean absolute error per item
- `time_mean_abs_error.csv` / `.png`: MAE by time series
- `hourly_mean_abs_error.csv` / `.png`: MAE by hour
- `tod_mean_abs_error.csv` / `.png`: MAE by 15-minute period of day
- `top_20_hardest_ids.csv`: top 20 hardest-to-predict items

---

## Step 2: Align item_id for Comparison

ASTGCN's `item_id` is the node index (0..715). It must be mapped to the sensor ID in `sd_meta.csv` for node-level comparison with Moirai2.

```bash
uv run python scripts/remap_astgcn_item_id.py \
  --err-csv results/ASTGCN/analysis/id_mean_abs_error.csv \
  --out results/ASTGCN/analysis/id_mean_abs_error_remapped.csv
```

---

## Step 3: Node-Level Error Comparison Between the Two Models

Use `compare_model_errors.py` to generate map and scatter plots:

```bash
uv run python scripts/compare_model_errors.py \
  --err1-path results/Moirai2/analysis/id_mean_abs_error.csv \
  --err2-path results/ASTGCN/analysis/id_mean_abs_error_remapped.csv \
  --model1-name Moirai2 \
  --model2-name ASTGCN \
  --out-dir results/comparison \
  --out-pdf moirai2_vs_astgcn.pdf
```

Outputs:
- `results/comparison/moirai2_vs_astgcn.pdf`: map (colored by node error difference) + scatter plot
- `moirai2_vs_astgcn_comparison.csv`: each node's MAE for both models and the difference

---

## Step 4: Other Optional Analyses

- **Spatial distribution**: `scripts/export_sd_region_error_pdf.py` exports error by County/District
- **Single-model map**: `scripts/export_sd_id_error_pdf.py` plots id_mean_abs_error for a given model
- **Temporal analysis**: `scripts/plot_hourly_mae.py`, etc.

---

## One-Shot Run (Optional)

```bash
cd /projects/bevu/jliu55/TrafficFM

# 1. Analyze
uv run python scripts/analyze_predictions.py --csv results/Moirai2/predictions/sd_2019_short_predictions.csv --output-dir results/Moirai2/analysis
uv run python scripts/analyze_predictions.py --csv results/ASTGCN/predictions/astgcn_sd_sd_2019_short_predictions.csv --output-dir results/ASTGCN/analysis

# 2. Remap ASTGCN item_id
uv run python scripts/remap_astgcn_item_id.py --err-csv results/ASTGCN/analysis/id_mean_abs_error.csv --out results/ASTGCN/analysis/id_mean_abs_error_remapped.csv

# 3. Compare
uv run python scripts/compare_model_errors.py \
  --err1-path results/Moirai2/analysis/id_mean_abs_error.csv \
  --err2-path results/ASTGCN/analysis/id_mean_abs_error_remapped.csv \
  --model1-name Moirai2 --model2-name ASTGCN \
  --out-dir results/comparison --out-pdf moirai2_vs_astgcn.pdf
```
