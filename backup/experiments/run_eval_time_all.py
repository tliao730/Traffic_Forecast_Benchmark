#!/usr/bin/env python3
"""
Run eval_time for all experiment models and save a single summary file of running times.

**支持的运行方式（uv 或 conda 均可）：**

  在 **TrafficFM 项目根目录** 下执行（即包含 experiments/ 和 benchmark/ 的那一层）：

    cd TrafficFM
    uv run python experiments/run_eval_time_all.py

  或使用 conda 环境时：

    cd TrafficFM
    conda activate <your_env>
    python experiments/run_eval_time_all.py

  其他用法：
    uv run python experiments/run_eval_time_all.py --aggregate-only   # 只汇总已有结果
    uv run python experiments/run_eval_time_all.py --models agcrn,dcrnn,lstm   # 只跑指定模型

Output:
  - 每个模型: results/<model_name>/time_estimation.json
  - 汇总: results/experiments_time_summary.csv 和 .json

与 benchmark 使用相同 eval_time 协议，便于对比运行时间。
"""

import argparse
import csv
import json
import os
import subprocess
import sys

# All experiment model dirs under experiments/ (each has main.py)
EXPERIMENT_MODELS = [
    "agcrn",
    "astgcn",
    "d2stgnn",
    "dcrnn",
    "dgcrn",
    "dstagnn",
    "gwnet",
    "hl",
    "lstm",
    "stgcn",
    "stgode",
    "sttn",
]


def get_experiments_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_trafficfm_root():
    return os.path.dirname(get_experiments_dir())


def get_result_root():
    return os.path.join(get_trafficfm_root(), "results")


def run_eval_time_for_model(model_name: str) -> bool:
    """Run experiments/<model>/main.py --eval-time from TrafficFM root. Returns True if ok."""
    root = get_trafficfm_root()
    main_py = os.path.join("experiments", model_name, "main.py")
    if not os.path.isfile(os.path.join(root, main_py)):
        print(f"[skip] {main_py} not found", file=sys.stderr)
        return False
    cmd = [sys.executable, main_py, "--eval-time"]
    print(f"[run] {model_name} --eval-time ...")
    ret = subprocess.run(cmd, cwd=root)
    if ret.returncode != 0:
        print(f"[fail] {model_name} exited with {ret.returncode}", file=sys.stderr)
        return False
    print(f"[ok] {model_name}")
    return True


def aggregate_time_results(result_root: str):
    """Collect time_estimation.json for experiment models and write experiments_time_summary."""
    summary_rows = []
    summary_dict = {}

    if not os.path.isdir(result_root):
        os.makedirs(result_root, exist_ok=True)
        print(f"No results dir yet: {result_root}. Run eval_time for some models first.")
        return

    for name in sorted(os.listdir(result_root)):
        path = os.path.join(result_root, name, "time_estimation.json")
        if not os.path.isfile(path):
            continue
        # Only include experiment models (optional: could also include benchmark and merge)
        if name not in EXPERIMENT_MODELS:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] Could not load {path}: {e}", file=sys.stderr)
            continue
        total = data.get("total") or {}
        model_name = data.get("model_name", name)
        model_path = data.get("model_path", "")
        total_sec = total.get("total_seconds", 0)
        total_min = total.get("total_minutes", 0)
        total_hr = total.get("total_hours", 0)
        summary_rows.append(
            {
                "model_name": model_name,
                "model_path": model_path,
                "total_seconds": total_sec,
                "total_minutes": round(total_min, 2),
                "total_hours": round(total_hr, 2),
            }
        )
        summary_dict[model_name] = {
            "model_path": model_path,
            "total_seconds": total_sec,
            "total_minutes": round(total_min, 2),
            "total_hours": round(total_hr, 2),
        }

    if not summary_rows:
        print("No experiment time_estimation.json found. Run eval_time for at least one model.")
        return

    summary_csv = os.path.join(result_root, "experiments_time_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "model_path", "total_seconds", "total_minutes", "total_hours"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {os.path.abspath(summary_csv)}")

    summary_json = os.path.join(result_root, "experiments_time_summary.json")
    with open(summary_json, "w") as f:
        json.dump({"models": summary_dict, "rows": summary_rows}, f, indent=2)
    print(f"Summary JSON: {os.path.abspath(summary_json)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run eval_time for all experiment models and save running times to a summary file."
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing time_estimation.json; do not run any script.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model dir names (e.g. agcrn,dcrnn,lstm). Default: all.",
    )
    args = parser.parse_args()

    result_root = get_result_root()

    if not args.aggregate_only:
        models = EXPERIMENT_MODELS
        if args.models:
            models = [s.strip() for s in args.models.split(",") if s.strip()]
        for model_name in models:
            run_eval_time_for_model(model_name)

    aggregate_time_results(result_root)


if __name__ == "__main__":
    main()
