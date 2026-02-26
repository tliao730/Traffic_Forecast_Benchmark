# Moirai2 vs ASTGCN 分析指南

## 数据说明

| 模型 | 预测文件 | item_id 含义 |
|------|----------|--------------|
| Moirai2 | `results/Moirai2/predictions/sd_2019_short_predictions.csv` | 传感器 ID (如 1114091) |
| ASTGCN | `results/ASTGCN/predictions/astgcn_sd_sd_2019_short_predictions.csv` | 节点索引 (0..715) |

**注意**：两个模型测试集时间段不同（Moirai2 用 gift_eval 的测试窗口，ASTGCN 用 idx_test 的最后 192 个窗口），比较的是各自测试集上的误差分布。

---

## Step 1：分别分析各模型

对每个模型的预测 CSV 运行 `analyze_predictions.py`，得到按时间、小时、item 的 MAE 及图表。

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

输出包括：
- `id_mean_abs_error.csv`：每个 item 的平均绝对误差
- `time_mean_abs_error.csv` / `.png`：按时间序列的 MAE
- `hourly_mean_abs_error.csv` / `.png`：按小时
- `tod_mean_abs_error.csv` / `.png`：按一天内 15 分钟时段
- `top_20_hardest_ids.csv`：最难预测的 20 个 item

---

## Step 2：统一 item_id 以便对比

ASTGCN 的 `item_id` 是节点索引 (0..715)，需要映射到 `sd_meta.csv` 中的传感器 ID，才能与 Moirai2 做节点级对比。

```bash
uv run python scripts/remap_astgcn_item_id.py \
  --err-csv results/ASTGCN/analysis/id_mean_abs_error.csv \
  --out results/ASTGCN/analysis/id_mean_abs_error_remapped.csv
```

---

## Step 3：两模型节点级误差对比

使用 `compare_model_errors.py` 生成地图和散点图：

```bash
uv run python scripts/compare_model_errors.py \
  --err1-path results/Moirai2/analysis/id_mean_abs_error.csv \
  --err2-path results/ASTGCN/analysis/id_mean_abs_error_remapped.csv \
  --model1-name Moirai2 \
  --model2-name ASTGCN \
  --out-dir results/comparison \
  --out-pdf moirai2_vs_astgcn.pdf
```

输出：
- `results/comparison/moirai2_vs_astgcn.pdf`：地图（按节点误差差着色）+ 散点图
- `moirai2_vs_astgcn_comparison.csv`：每个节点两个模型的 MAE 及差值

---

## Step 4：其他可选分析

- **地域分布**：`scripts/export_sd_region_error_pdf.py` 可导出按 County/District 的误差
- **单个模型地图**：`scripts/export_sd_id_error_pdf.py` 可对某个模型的 id_mean_abs_error 画图
- **时段分析**：`scripts/plot_hourly_mae.py` 等

---

## 一键运行（可选）

```bash
cd /projects/bevu/jliu55/TrafficFM

# 1. 分析
uv run python scripts/analyze_predictions.py --csv results/Moirai2/predictions/sd_2019_short_predictions.csv --output-dir results/Moirai2/analysis
uv run python scripts/analyze_predictions.py --csv results/ASTGCN/predictions/astgcn_sd_sd_2019_short_predictions.csv --output-dir results/ASTGCN/analysis

# 2.  remap ASTGCN item_id
uv run python scripts/remap_astgcn_item_id.py --err-csv results/ASTGCN/analysis/id_mean_abs_error.csv --out results/ASTGCN/analysis/id_mean_abs_error_remapped.csv

# 3. 对比
uv run python scripts/compare_model_errors.py \
  --err1-path results/Moirai2/analysis/id_mean_abs_error.csv \
  --err2-path results/ASTGCN/analysis/id_mean_abs_error_remapped.csv \
  --model1-name Moirai2 --model2-name ASTGCN \
  --out-dir results/comparison --out-pdf moirai2_vs_astgcn.pdf
```
