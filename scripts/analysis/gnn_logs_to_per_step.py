#!/usr/bin/env python3
"""Extract per-horizon GNN test metrics from SLURM logs into per_step_results.csv.

The GNN family never writes to result_root; its numbers only exist as
"Horizon k, Test MAE: ..." lines in log/gnn/**. The gift_eval side now emits
per-step rows (common.py, axis=0), so pulling the GNN side into the same shape
is what lets one table hold both. Columns match common.py's PER_STEP_CSV_NAME
schema for the metrics GNN actually reports; the rest are left blank.

Usage:
    python scripts/analysis/gnn_logs_to_per_step.py [--log-root log/gnn] \
        [--out results/gnn_per_step_results.csv]
"""
import argparse
import csv
import os
import re

HORIZON_RE = re.compile(
    r"Horizon (\d+), Test MAE: ([0-9.]+), Test RMSE: ([0-9.]+), Test MAPE: ([0-9.]+)"
)
STRIDE_RE = re.compile(r"falling back to stride mode")
ALIGNED_RE = re.compile(r"gift_eval aligned anchors")

# Same header as common.py's per_step_results.csv, plus provenance columns that
# matter here: whether the run was window-aligned, and which log it came from.
HEADER = [
    "dataset", "model", "step",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]", "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]", "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS", "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
    "window_mode", "source_log",
]


def parse_log(path):
    """Return (rows, window_mode) for the LAST test block in the log, or None."""
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return None
    matches = HORIZON_RE.findall(text)
    if not matches:
        return None
    # A log may hold several runs (resumes append). Keep the final block: walk
    # backwards while the horizon index keeps decreasing.
    block = []
    for step, mae, rmse, mape in reversed(matches):
        step = int(step)
        if block and step >= block[-1][0]:
            break
        block.append((step, float(mae), float(rmse), float(mape)))
    block.reverse()
    if ALIGNED_RE.search(text):
        mode = "gift_eval_aligned"
    elif STRIDE_RE.search(text):
        mode = "stride_fallback"
    else:
        mode = "unknown"
    return block, mode


def main():
    ap = argparse.ArgumentParser()
    # Both trees share the {model}/{region}/{year}/{jobid}.out layout. Reading
    # them together lets a fresh log/gnn_eval run supersede the stride-mode
    # result the training run left in log/gnn, since the newest file per cell
    # wins and window_mode records which kind it was.
    ap.add_argument("--log-root", nargs="+", default=["log/gnn", "log/gnn_eval"])
    ap.add_argument("--out", default="results/gnn_per_step_results.csv")
    args = ap.parse_args()

    # log/gnn/{model}/{region}/{year}/{jobid}.out -> keep the newest job per cell
    # Collect every candidate per cell rather than only the newest file: a job
    # that is still running has already created its .out, so "newest" alone
    # would pick an empty log over a finished older one.
    candidates = {}
    for root in args.log_root:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith(".out"):
                    continue
                p = os.path.join(dirpath, fn)
                parts = os.path.relpath(p, root).split(os.sep)
                if len(parts) != 4:
                    continue  # _misc/, _probe/ and pre-migration layouts
                model, region, year, _ = parts
                candidates.setdefault((model, region, year), []).append(p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_cells = n_aligned = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for (model, region, year), paths in sorted(candidates.items()):
            # Newest first, but skip logs with no finished test block.
            parsed = path = None
            for p in sorted(paths, key=os.path.getmtime, reverse=True):
                parsed = parse_log(p)
                if parsed:
                    path = p
                    break
            if not parsed:
                continue
            block, mode = parsed
            n_cells += 1
            n_aligned += mode == "gift_eval_aligned"
            # 12 forecast steps == the gift_eval "long" term, so that is the
            # dataset key these rows belong under.
            ds_config = f"{region}/{year}/long"
            for step, mae, rmse, mape in block:
                w.writerow([
                    ds_config, model.upper(), step,
                    "", "", mae, "", mape, "", "", rmse, "", "", "",
                    mode, os.path.relpath(path),
                ])
    print(f"Wrote {args.out}: {n_cells} cells, {n_aligned} gift_eval-aligned, "
          f"{n_cells - n_aligned} still on stride windows.")


if __name__ == "__main__":
    main()
