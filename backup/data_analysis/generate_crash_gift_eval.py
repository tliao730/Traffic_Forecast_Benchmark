"""
Build GIFT-eval (Arrow) dataset from crash_flow_after_2019-10-20.json.

Task (from advisor):
- History: 720 steps (values[:720] for crash_id column).
- Predict: future 12 steps (values[720:732]).
- Each record → one univariate series (crash_id only, column 0).
- Output: Arrow dataset at output_dir/crash_sd/2019/15T/ for Dataset(name="crash_sd/2019/15T", term="long").
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value

# Fixed lengths from task
HISTORY_LEN = 720
PREDICTION_LEN = 12
FREQ = "15T"


def load_crash_json(path: str) -> list:
    """Load crash JSON; return list of records."""
    with open(path) as f:
        return json.load(f)


def build_records_from_crash(data: list, use_crash_id_only: bool = True) -> list:
    """
    Build GIFT-style records from crash list.

    If use_crash_id_only: one record per crash (column 0 only).
    Else: one record per (crash, id) → 6 records per crash.
    """
    records = []
    for i, rec in enumerate(data):
        values = np.array(rec["values"], dtype=np.float32)
        n_steps, n_ids = values.shape
        if n_steps < HISTORY_LEN + PREDICTION_LEN:
            continue

        # Start time: first time step of the series (crash_time is at step 720)
        crash_time = pd.Timestamp(rec["crash_time"])
        # 15T = 15 min; history has 720 steps before crash
        start_time = crash_time - pd.Timedelta(minutes=15 * HISTORY_LEN)

        if use_crash_id_only:
            # One series per crash: crash_id (column 0)
            target = values[: HISTORY_LEN + PREDICTION_LEN, 0].tolist()
            item_id = f"{rec['crash_id']}_{rec['crash_time'].replace(' ', '_')}"
            records.append({
                "item_id": item_id,
                "start": start_time.isoformat(),
                "freq": FREQ,
                "target": target,
            })
        else:
            # One series per (crash, id) → 6 per crash
            for j, sid in enumerate(rec["ids_order"]):
                target = values[: HISTORY_LEN + PREDICTION_LEN, j].tolist()
                item_id = f"{rec['crash_id']}_{rec['crash_time'].replace(' ', '_')}_{sid}"
                records.append({
                    "item_id": item_id,
                    "start": start_time.isoformat(),
                    "freq": FREQ,
                    "target": target,
                })
    return records


def save_as_arrow(records: list, output_path: str) -> "Dataset":
    """Save records as HuggingFace Arrow dataset (same schema as generate_data_for_gift_eval)."""
    features = Features({
        "item_id": Value("string"),
        "start": Value("string"),
        "freq": Value("string"),
        "target": Sequence(Value("float32")),
    })
    dataset = Dataset.from_list(records, features=features)
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    return dataset


def prepare_output_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{path}' already exists. Use --overwrite to replace."
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build GIFT-eval Arrow dataset from crash_flow_after_2019-10-20.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="crash_flow_after_2019-10-20.json",
        help="Path to crash JSON (relative to cwd or absolute)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/gift_eval_datasets",
        help="Root directory for GIFT datasets; must match benchmark config (data/gift_eval_datasets)",
    )
    parser.add_argument(
        "--all_ids",
        action="store_true",
        help="If set, emit 6 series per crash (one per ID); else 1 per crash (crash_id only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    args = parser.parse_args()

    # Resolve paths (root = TrafficFM project root)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Crash JSON not found: {input_path}")

    # Output dir: if relative, resolve relative to project root so it works from any cwd
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(root, args.output_dir)
    output_path = os.path.join(output_dir, "crash_sd", "2019", FREQ)
    prepare_output_dir(output_path, args.overwrite)

    data = load_crash_json(input_path)
    records = build_records_from_crash(data, use_crash_id_only=not args.all_ids)
    if not records:
        raise ValueError("No valid records (need at least 720+12 time steps per series).")

    ds = save_as_arrow(records, output_path)
    print(f"Saved {ds.num_rows} series to {os.path.abspath(output_path)}")
    print(f"Dataset name for benchmark: crash_sd/2019/15T")
    print(f"Use term='long' for prediction_length=12.")


if __name__ == "__main__":
    main()
