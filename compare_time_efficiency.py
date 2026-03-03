#!/usr/bin/env python3
"""
Compare eval_time results between foundation models (benchmark) and
graph/multivariate models (experiments) on a per-node basis.

The key idea (matching your advisor's explanation):
  - Foundation models (benchmark) in this project see ONE node per sample
    for the traffic datasets (sd/gba/gla/ca).
    → their avg_time_per_sample is per-node time.
  - Experiment models (GNNs, LSTM, etc. under experiments/) see ALL nodes
    of a dataset at once (bs=1) for each sample.
    → their avg_time_per_sample is time for the whole graph (all nodes).

So for a dataset with N nodes, we can derive:
  - foundation: time_per_node = avg_time_per_sample
               time_full_graph_step = N * avg_time_per_sample
  - experiments: time_per_node = avg_time_per_sample / N
                 time_full_graph_step = avg_time_per_sample

This script reads all results/*/time_estimation.json files and produces
an aggregated CSV with per-node and per-graph timings that can be used
to make tables or plots for your advisor.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Number of nodes per traffic dataset (from src/utils/dataloader.get_dataset_info)
NODE_COUNTS: Dict[str, int] = {
    "sd": 716,
    "gba": 2352,
    "gla": 3834,
    "ca": 8600,
}

# We only care about these traffic datasets (short/medium/long terms)
TRAFFIC_DS_PREFIXES = set(NODE_COUNTS.keys())

Family = Literal["foundation_benchmark", "graph_experiment"]


@dataclass
class DatasetTiming:
    model_name: str
    model_path: str
    family: Family
    dataset: str  # e.g. "sd/2019/short"
    num_samples: int
    measured_samples: int
    avg_time_per_sample: float  # seconds
    nodes_per_sample: int
    time_per_node_ms: float
    nodes_per_second: float
    time_full_graph_step_s: float


def classify_family(model_path: str) -> Family:
    """
    Simple heuristic:
      - results coming from experiments/<model>/... are graph/multivariate models
      - everything else we treat as benchmark/foundation (per-node) models
    """
    if model_path.startswith("experiments/"):
        return "graph_experiment"
    return "foundation_benchmark"


def nodes_for_sample(family: Family, ds_config: str) -> int:
    """
    For traffic datasets:
      - foundation_benchmark: one sample = one node's series → 1 node
      - graph_experiment: one sample = full graph → N nodes
    For any non-traffic dataset (should not appear here), fall back to 1.
    """
    ds_key = ds_config.split("/")[0].lower()
    if ds_key not in NODE_COUNTS:
        return 1
    if family == "foundation_benchmark":
        return 1
    return NODE_COUNTS[ds_key]


def iter_time_estimation_files(results_dir: str) -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(results_dir):
        return paths
    for name in sorted(os.listdir(results_dir)):
        sub = os.path.join(results_dir, name)
        if not os.path.isdir(sub):
            continue
        json_path = os.path.join(sub, "time_estimation.json")
        if os.path.isfile(json_path):
            paths.append(json_path)
    return paths


def load_timings_for_file(path: str) -> List[DatasetTiming]:
    with open(path) as f:
        data = json.load(f)

    model_name: str = data.get("model_name", os.path.basename(os.path.dirname(path)))
    model_path: str = data.get("model_path", "")
    family: Family = classify_family(model_path)

    timings: List[DatasetTiming] = []
    datasets: Dict[str, Dict[str, float]] = data.get("datasets", {})

    for ds_config, d in datasets.items():
        # Only keep our traffic datasets sd/gba/gla/ca
        ds_key = ds_config.split("/")[0].lower()
        if ds_key not in TRAFFIC_DS_PREFIXES:
            continue

        num_samples = int(d.get("num_samples", 0))
        measured_samples = int(d.get("measured_samples", 0))
        avg_time = float(d.get("avg_time_per_sample", 0.0))
        if avg_time <= 0 or measured_samples <= 0 or num_samples <= 0:
            continue

        n_sample = nodes_for_sample(family, ds_config)
        time_per_node_ms = (avg_time / n_sample) * 1000.0
        nodes_per_second = (n_sample / avg_time) if avg_time > 0 else 0.0

        # "One full-graph prediction step" means: forecast once for all N nodes
        if family == "graph_experiment":
            time_full_graph_step_s = avg_time
        else:
            # foundation: each sample sees one node, need N sequential predictions
            n_nodes = NODE_COUNTS.get(ds_key, 1)
            time_full_graph_step_s = avg_time * n_nodes

        timings.append(
            DatasetTiming(
                model_name=model_name,
                model_path=model_path,
                family=family,
                dataset=ds_config,
                num_samples=num_samples,
                measured_samples=measured_samples,
                avg_time_per_sample=avg_time,
                nodes_per_sample=n_sample,
                time_per_node_ms=time_per_node_ms,
                nodes_per_second=nodes_per_second,
                time_full_graph_step_s=time_full_graph_step_s,
            )
        )

    return timings


def collect_all_timings(results_dir: str) -> List[DatasetTiming]:
    all_timings: List[DatasetTiming] = []
    for json_path in iter_time_estimation_files(results_dir):
        all_timings.extend(load_timings_for_file(json_path))
    return all_timings


def write_csv(timings: List[DatasetTiming], out_path: str) -> None:
    fieldnames = [
        "model_name",
        "model_path",
        "family",
        "dataset",
        "num_samples",
        "measured_samples",
        "avg_time_per_sample_s",
        "nodes_per_sample",
        "time_per_node_ms",
        "nodes_per_second",
        "time_full_graph_step_s",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in timings:
            writer.writerow(
                {
                    "model_name": t.model_name,
                    "model_path": t.model_path,
                    "family": t.family,
                    "dataset": t.dataset,
                    "num_samples": t.num_samples,
                    "measured_samples": t.measured_samples,
                    "avg_time_per_sample_s": f"{t.avg_time_per_sample:.6f}",
                    "nodes_per_sample": t.nodes_per_sample,
                    "time_per_node_ms": f"{t.time_per_node_ms:.6f}",
                    "nodes_per_second": f"{t.nodes_per_second:.2f}",
                    "time_full_graph_step_s": f"{t.time_full_graph_step_s:.6f}",
                }
            )


def main(dataset_filter: Optional[str] = None) -> None:
    """
    If dataset_filter is provided (e.g. "sd/2019/short"), only keep that dataset.
    Otherwise, keep all traffic datasets.
    """
    timings = collect_all_timings(RESULTS_DIR)
    if dataset_filter is not None:
        timings = [t for t in timings if t.dataset == dataset_filter]

    if not timings:
        print("No timing records found for the specified filter.")
        return

    # Sort for easier reading: by dataset, then family, then model_name
    timings.sort(key=lambda t: (t.dataset, t.family, t.model_name))

    # Write aggregated CSV for further analysis / plotting
    if dataset_filter:
        safe_name = dataset_filter.replace("/", "_")
        out_csv = os.path.join(RESULTS_DIR, f"time_per_node_{safe_name}.csv")
    else:
        out_csv = os.path.join(RESULTS_DIR, "time_per_node_all_traffic.csv")
    write_csv(timings, out_csv)

    print(f"Wrote aggregated timing table to: {os.path.abspath(out_csv)}")
    print()

    # Also print a compact text table for quick inspection (per dataset)
    current_ds: Optional[str] = None
    for t in timings:
        if t.dataset != current_ds:
            current_ds = t.dataset
            print(f"=== Dataset: {current_ds} ===")
            print(
                "model_name".ljust(24),
                "family".ljust(22),
                "avg_s/sample".rjust(12),
                "nodes/sample".rjust(12),
                "ms/node".rjust(12),
                "nodes/s".rjust(12),
                "full_graph_s".rjust(14),
            )
        print(
            t.model_name.ljust(24),
            t.family.ljust(22),
            f"{t.avg_time_per_sample:>12.4f}",
            f"{t.nodes_per_sample:>12d}",
            f"{t.time_per_node_ms:>12.4f}",
            f"{t.nodes_per_second:>12.2f}",
            f"{t.time_full_graph_step_s:>14.4f}",
        )


if __name__ == "__main__":
    # Default: all traffic datasets. You can also run, e.g.:
    #   python compare_time_efficiency.py   # all
    #   python compare_time_efficiency.py sd/2019/short
    import sys

    ds_filter = sys.argv[1] if len(sys.argv) > 1 else None
    main(ds_filter)

