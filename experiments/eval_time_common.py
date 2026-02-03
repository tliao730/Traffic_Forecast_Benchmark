"""
Shared eval_time logic for experiments: same protocol as benchmark/common.eval_time
(same datasets, same estimation_samples, same output JSON format) so that
benchmark vs experiments runtimes can be compared.

Each experiment main.py implements get_model_and_batches_for_eval_time(dataset_key, seq_len, horizon, device, estimation_samples)
returning (model, list of (x, y) batches). This module then times forward passes and writes
results/<model_name>/time_estimation.json in the same format as benchmark.
"""

import json
import os
import sys
import time

# Ensure TrafficFM root and benchmark/ are on path so "config" and "benchmark" resolve
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAFFICFM_ROOT = os.path.dirname(_SCRIPT_DIR)
_BENCHMARK_DIR = os.path.join(_TRAFFICFM_ROOT, "benchmark")
if _TRAFFICFM_ROOT not in sys.path:
    sys.path.insert(0, _TRAFFICFM_ROOT)
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(1, _BENCHMARK_DIR)

# Load config as "config" (same name benchmark.common uses) and patch to absolute paths
# so setup_dataset() finds dataset_properties.json when cwd is TrafficFM root
import config
config.dataset_properties_path = os.path.join(_TRAFFICFM_ROOT, "benchmark", "dataset_properties.json")
config.gift_eval_datasets_path = os.path.join(_TRAFFICFM_ROOT, "data", "gift_eval_datasets")
config.result_root = os.path.join(_TRAFFICFM_ROOT, "results")

from benchmark.common import get_prediction_length, setup_dataset
gift_eval_datasets_path = config.gift_eval_datasets_path
med_long_datasets = config.med_long_datasets
short_datasets = config.short_datasets

# gift_eval for num_test_samples per (ds_name, term)
from gift_eval.data import Dataset as GiftEvalDataset


# gift_eval dataset name (e.g. sd/2019/15T) -> experiment dataset key (SD, GLA, GBA, CA)
GIFT_EVAL_TO_EXPERIMENT_DATASET = {
    "sd/2019/15T": "SD",
    "gba/2019/15T": "GBA",
    "gla/2019/15T": "GLA",
    "ca/2019/15T": "CA",
}


def run_eval_time_for_experiment(
    model_name: str,
    model_path: str,
    get_model_and_batches_fn,
    estimation_samples: int = 10,
):
    """
    Run eval_time protocol for an experiment model: same datasets and output format as benchmark.

    get_model_and_batches_fn(dataset_key, seq_len, horizon, device, estimation_samples)
    must return (model, list of (x, y) batches) where model(x, y) is the forward pass.
    """
    os.environ["GIFT_EVAL"] = gift_eval_datasets_path

    short_ds, med_long_ds, all_datasets, dataset_properties_map = setup_dataset()

    timing_results = {
        "model_name": model_name,
        "model_path": model_path,
        "estimation_samples": estimation_samples,
        "datasets": {},
    }
    total_estimated_time = 0.0

    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    for ds_num, ds_name in enumerate(all_datasets):
        if "/" in ds_name:
            ds_key = ds_name.split("/")[0].lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = ds_name.split("/")[1]
        else:
            ds_key = ds_name.lower()
            ds_freq = dataset_properties_map.get(ds_key, {}).get("frequency", "15T")

        print(f"\nDataset {ds_num + 1}/{len(all_datasets)}: {ds_name}")

        terms = ["short", "medium", "long"]
        for term in terms:
            if (term == "medium" or term == "long") and ds_name not in med_long_ds.split():
                continue

            ds_config = f"{ds_key}/{ds_freq}/{term}"

            # Num test samples from gift_eval (same as benchmark)
            try:
                to_uni = (
                    False
                    if GiftEvalDataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                    else True
                )
                gift_ds = GiftEvalDataset(name=ds_name, term=term, to_univariate=to_uni)
                num_test_samples = len(gift_ds.test_data)
            except Exception as e:
                print(f"  [skip] {term}: gift_eval failed: {e}", file=sys.stderr)
                continue

            prediction_length = get_prediction_length(term)
            horizon = prediction_length
            seq_len = 12

            # Map to experiment dataset key (SD, GLA, GBA, CA)
            exp_dataset_key = GIFT_EVAL_TO_EXPERIMENT_DATASET.get(ds_name)
            if exp_dataset_key is None:
                print(f"  [skip] {term}: no mapping for {ds_name}", file=sys.stderr)
                continue

            try:
                model, batches = get_model_and_batches_fn(
                    exp_dataset_key, seq_len, horizon, estimation_samples
                )
            except Exception as e:
                print(f"  [skip] {term}: get_model_and_batches failed: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue

            measure_samples = min(estimation_samples, len(batches), num_test_samples)
            if measure_samples == 0:
                print(f"  [skip] {term}: no batches", file=sys.stderr)
                continue

            # Time forward passes (no grad)
            model.eval()
            import torch
            device = next(model.parameters()).device

            def to_device(t):
                if isinstance(t, torch.Tensor):
                    return t.to(device)
                if isinstance(t, (list, tuple)):
                    return type(t)(to_device(x) for x in t)
                return torch.tensor(t, dtype=torch.float32, device=device)

            start_time = time.time()
            with torch.no_grad():
                for i in range(measure_samples):
                    x, y = batches[i]
                    x = to_device(x)
                    y = to_device(y)
                    _ = model(x, y)
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / measure_samples
            estimated_time = avg_time_per_sample * num_test_samples

            timing_results["datasets"][ds_config] = {
                "num_samples": num_test_samples,
                "measured_samples": measure_samples,
                "avg_time_per_sample": avg_time_per_sample,
                "estimated_total_seconds": estimated_time,
                "estimated_total_minutes": estimated_time / 60,
                "estimated_total_hours": estimated_time / 3600,
            }
            total_estimated_time += estimated_time
            print(
                f"  {term}: {num_test_samples} samples "
                f"→ {avg_time_per_sample:.4f}s/sample → ~{estimated_time:.1f}s (~{estimated_time/60:.1f}min)"
            )

    timing_results["total"] = {
        "total_seconds": total_estimated_time,
        "total_minutes": total_estimated_time / 60,
        "total_hours": total_estimated_time / 3600,
    }

    print("\n" + "=" * 70)
    print("TOTAL ESTIMATED TIME:")
    print(f"  {total_estimated_time:.2f} seconds")
    print(f"  {total_estimated_time/60:.2f} minutes")
    print(f"  {total_estimated_time/3600:.2f} hours")
    print("=" * 70)

    # result_root is relative to benchmark/; we run from TrafficFM so results/ is correct
    out_dir = os.path.join(_TRAFFICFM_ROOT, "results", model_name)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "time_estimation.json")
    with open(json_path, "w") as f:
        json.dump(timing_results, f, indent=2)
    print(f"\nTiming results saved to: {os.path.abspath(json_path)}")
    return timing_results
