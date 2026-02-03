#!/usr/bin/env python3
"""
Run eval_time for all foundation models and save a single summary file of running times.

Each model runs in its own uv environment when one exists under envs/ (see UV_USAGE_GUIDE.md).
Otherwise the TrafficFM root project environment is used.

Usage (run from benchmark/; uv must be on PATH so each model runs in its own env):
  cd TrafficFM/benchmark

  # Run eval_time for all foundation models, then aggregate times into one file.
  # This script uses only the stdlib and invokes each model via uv run --project <env>.
  python run_eval_time_all.py

  # Only aggregate existing time_estimation.json files (no subprocess runs)
  python run_eval_time_all.py --aggregate-only

  # Run only specific models (comma-separated script names without .py)
  python run_eval_time_all.py --models chronos_1,moirai,naive

  # Run only the five models with dedicated uv envs
  python run_eval_time_all.py --models timesfm_eval,moirai,kairos,flowstate,feedforward

Output:
  - Each model's detailed timing: ../results/<model_name>/time_estimation.json (from common.eval_time)
  - Summary of all models: ../results/foundation_models_time_summary.csv and .json

Note: When running models (not --aggregate-only), `uv` must be on PATH. Each script is run
with `uv run --project <env>` so it uses the correct env (envs/<name>/ or root).
"""

import argparse
import json
import os
import subprocess
import sys

# Foundation model scripts that support --eval-time (script name without .py).
# Scripts with a dedicated uv env under envs/<name>/: use that env; others use TrafficFM root.
# Value: env dir relative to TrafficFM root (e.g. "envs/timesfm_eval") or None for root project.
FOUNDATION_MODEL_SCRIPTS = [
    "arima",          # root
    "chronos_1",      # root
    "timesfm_eval",   # envs/timesfm_eval
    "moirai",         # envs/moirai
    "kairos",         # envs/kairos
    "flowstate",      # envs/flowstate
    "feedforward",    # envs/feedforward
    "sundial",        # root
    "toto",           # root
    "tabpfn_ts",      # root
    "naive",          # root
    "moirai2",        # root
    "ml_methods",     # envs/tabpfn_ts
    "ml_ensemble",    # envs/tabpfn_ts
]

# Script stem -> env directory relative to TrafficFM root (None = use root project)
SCRIPT_TO_ENV: dict[str, str | None] = {
    "arima": "envs/arima",
    "chronos_1": "envs/moirai",
    "feedforward": "envs/feedforward",
    "flowstate": "envs/flowstate",
    "kairos": "envs/kairos",
    "moirai": "envs/moirai",
    "moirai2": "envs/moirai",
    "ml_ensemble": "envs/tabpfn_ts",
    "ml_methods": "envs/tabpfn_ts",
    "naive": "envs/feedforward",
    "sundial": "envs/sundial",
    "tabpfn_ts": "envs/tabpfn_ts",
    "timesfm_eval": "envs/timesfm_eval",
    "toto": "envs/toto",
}


def get_benchmark_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_result_root():
    # config.result_root is relative to benchmark/
    return os.path.normpath(os.path.join(get_benchmark_dir(), "..", "results"))


def run_eval_time_for_script(script_stem: str) -> bool:
    """
    Run script with --eval-time in the correct uv environment.
    - If script has a dedicated env in envs/<name>/, use: uv run --project ../envs/<name> python <script>.py --eval-time
    - Otherwise use root project: uv run --project .. python <script>.py --eval-time
    Always runs with cwd=benchmark/ so that relative paths (e.g. ../results) resolve correctly.
    Returns True if successful.
    """
    benchmark_dir = get_benchmark_dir()
    script_path = os.path.join(benchmark_dir, f"{script_stem}.py")
    if not os.path.isfile(script_path):
        print(f"[skip] {script_stem}.py not found", file=sys.stderr)
        return False

    env_dir = SCRIPT_TO_ENV.get(script_stem)
    if env_dir is not None:
        # Dedicated env under envs/<name>/; path from benchmark/ is ../envs/<name>
        project_arg = os.path.join("..", env_dir)
        env_label = env_dir
    else:
        # Use TrafficFM root project (parent of benchmark/)
        project_arg = ".."
        env_label = "root"

    # Prefer uv run so each model uses its own env; fall back to current Python if uv missing
    uv_cmd = ["uv", "run", "--project", project_arg, "python", f"{script_stem}.py", "--eval-time"]
    print(f"[run] {script_stem}.py --eval-time (env: {env_label}) ...")
    ret = subprocess.run(
        uv_cmd,
        cwd=benchmark_dir,
    )
    if ret.returncode != 0:
        print(f"[fail] {script_stem}.py exited with {ret.returncode}", file=sys.stderr)
        return False
    print(f"[ok] {script_stem}.py")
    return True


def aggregate_time_results(result_root: str):
    """
    Collect all time_estimation.json under result_root and write
    foundation_models_time_summary.csv and .json.
    """
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
        print("No time_estimation.json files found. Run eval_time for at least one model.")
        return

    # Write CSV
    import csv
    summary_csv = os.path.join(result_root, "foundation_models_time_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "model_path", "total_seconds", "total_minutes", "total_hours"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {os.path.abspath(summary_csv)}")

    # Write JSON
    summary_json = os.path.join(result_root, "foundation_models_time_summary.json")
    with open(summary_json, "w") as f:
        json.dump({"models": summary_dict, "rows": summary_rows}, f, indent=2)
    print(f"Summary JSON: {os.path.abspath(summary_json)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run eval_time for foundation models and save running times to a summary file."
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
        help="Comma-separated list of script names (e.g. chronos_1,moirai,naive). Default: all.",
    )
    args = parser.parse_args()

    result_root = get_result_root()

    if not args.aggregate_only:
        scripts = FOUNDATION_MODEL_SCRIPTS
        if args.models:
            scripts = [s.strip() for s in args.models.split(",") if s.strip()]
        for script_stem in scripts:
            run_eval_time_for_script(script_stem)

    aggregate_time_results(result_root)


if __name__ == "__main__":
    main()
