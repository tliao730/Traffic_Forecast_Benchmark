import csv
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
from gift_eval.data import Dataset
from config import (
    dataset_properties_path,
    default_plot_dataset,
    default_plot_quantile,
    default_plot_sample_idx,
    default_plot_term,
    gift_eval_datasets_path,
    med_long_datasets,
    result_root,
    short_datasets,
)
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


def setup_logger():
    gts_logger = logging.getLogger("gluonts.model.forecast")
    gts_logger.addFilter(
        WarningFilter("The mean prediction is not stored in the forecast data")
    )


def setup_dataset():
    os.environ["GIFT_EVAL"] = gift_eval_datasets_path

    # Get union of short and med_long datasets
    all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

    if not os.path.exists(dataset_properties_path):
        raise FileNotFoundError("dataset_properties.json not found.")
    dataset_properties_map = json.load(open(dataset_properties_path))

    # Add properties for new datasets (ca, gba, gla)
    if "ca" not in dataset_properties_map:
        dataset_properties_map["ca"] = {
            "frequency": "15T",
            "domain": "Transport",
            "num_variates": 1,
        }
    if "gba" not in dataset_properties_map:
        dataset_properties_map["gba"] = {
            "frequency": "15T",
            "domain": "Transport",
            "num_variates": 1,
        }
    if "gla" not in dataset_properties_map:
        dataset_properties_map["gla"] = {
            "frequency": "15T",
            "domain": "Transport",
            "num_variates": 1,
        }
    if "crash_sd" not in dataset_properties_map:
        dataset_properties_map["crash_sd"] = {
            "frequency": "15T",
            "domain": "Transport",
            "num_variates": 1,
        }

    return short_datasets, med_long_datasets, all_datasets, dataset_properties_map


def get_prediction_length(term):
    """Get prediction length based on term type."""
    term_to_length = {
        "short": 3,
        "medium": 6,
        "long": 12,
    }
    return term_to_length.get(term, 3)


def get_metrics():
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]


def check_done_datasets(csv_file_path):
    done_datasets = []
    if os.path.exists(csv_file_path):
        df_done = pd.read_csv(csv_file_path)
        done_datasets = df_done["dataset"].values.tolist()
        print(f"Found {len(done_datasets)} completed datasets.")
    else:
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "eval_metrics/MSE[mean]",
                    "eval_metrics/MSE[0.5]",
                    "eval_metrics/MAE[0.5]",
                    "eval_metrics/MASE[0.5]",
                    "eval_metrics/MAPE[0.5]",
                    "eval_metrics/sMAPE[0.5]",
                    "eval_metrics/MSIS",
                    "eval_metrics/RMSE[mean]",
                    "eval_metrics/NRMSE[mean]",
                    "eval_metrics/ND[0.5]",
                    "eval_metrics/mean_weighted_sum_quantile_loss",
                    "domain",
                    "num_variates",
                ]
            )
    return done_datasets


def write_result_to_csv(
    res, csv_file_path, ds_config, ds_key, dataset_properties_map, model_name
):
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                ds_config,
                model_name,
                res["MSE[mean]"][0],
                res["MSE[0.5]"][0],
                res["MAE[0.5]"][0],
                res["MASE[0.5]"][0],
                res["MAPE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["MSIS"][0],
                res["RMSE[mean]"][0],
                res["NRMSE[mean]"][0],
                res["ND[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
                dataset_properties_map[ds_key]["domain"],
                dataset_properties_map[ds_key]["num_variates"],
            ]
        )


def show_results(model_name):
    results_file = f"{result_root}/{model_name}/all_results.csv"
    df = pd.read_csv(results_file)
    print("\nFinal aggregated results:")
    print(df)


def plot_forecast_vs_truth(
    predictor_factory,
    dataset_name: Optional[str] = None,
    term: Optional[str] = None,
    sample_idx: Optional[int] = None,
    quantile: Optional[float] = None,
    save_path: Optional[str] = None,
    history_length: Optional[int] = None,
):
    """
    Plot history, forecast and residuals (prediction - ground truth) for one test series.

    Args:
        predictor_factory: Callable that takes a dataset and returns a predictor (same as eval uses).
        dataset_name: Name used by Dataset (e.g., "sd/2019/15T"). Uses default from config if None.
        term: One of "short", "medium", "long". Uses default from config if None.
        sample_idx: Which sample in the dataset's test set to plot. Uses default from config if None.
        quantile: Which quantile to show as the main forecast line. Uses default from config if None.
        save_path: If provided, saves the figure instead of showing it.
        history_length: Number of historical points to show in the plot. If None, shows all history.
    """
    # Set up GIFT_EVAL environment variable
    os.environ["GIFT_EVAL"] = gift_eval_datasets_path
    
    # Use defaults from config if not provided
    dataset_name = dataset_name or default_plot_dataset
    term = term or default_plot_term
    sample_idx = sample_idx if sample_idx is not None else default_plot_sample_idx
    quantile = quantile if quantile is not None else default_plot_quantile
    # Align with eval's univariate handling
    probe_ds = Dataset(name=dataset_name, term=term, to_univariate=False)
    to_univariate = False if probe_ds.target_dim == 1 else True
    dataset = Dataset(name=dataset_name, term=term, to_univariate=to_univariate)
    
    # Override dataset's prediction_length with our custom values
    prediction_length = get_prediction_length(term)
    dataset.prediction_length = prediction_length

    predictor = predictor_factory(dataset)

    # Convert test_data to list to allow indexing
    test_data_list = list(dataset.test_data)
    series_data = test_data_list[sample_idx]
    
    # Handle tuple format (input, label) from test_data
    if isinstance(series_data, tuple):
        series = series_data[0]  # Extract the input dict from tuple
    else:
        series = series_data
    
    forecast = predictor.predict([series])[0]

    history_full = np.asarray(series["target"], dtype=float)
    if prediction_length > len(history_full):
        raise ValueError("prediction_length is longer than the available history.")

    future_truth = history_full[-prediction_length:]
    
    # Print experiment settings
    print("\n" + "="*60)
    print("Experiment Settings:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Term: {term}")
    print(f"  Sample Index: {sample_idx}")
    print(f"  Total History Length: {len(history_full)}")
    print(f"  Prediction Length: {prediction_length}")
    print(f"  Quantile: {quantile}")
    if history_length is not None:
        print(f"  Displayed History Length: {history_length}")
    else:
        print(f"  Displayed History Length: All ({len(history_full)})")
    print("="*60 + "\n")
    
    # Determine how much history to show
    if history_length is not None:
        # Show only the last `history_length` points (including the prediction horizon)
        start_idx = max(0, len(history_full) - history_length)
        history = history_full[start_idx:]
        offset = start_idx
    else:
        # Show all history
        history = history_full
        offset = 0
    
    history_idx = np.arange(offset, offset + len(history))
    horizon_idx = np.arange(len(history_full) - prediction_length, len(history_full))

    q_key = str(quantile)
    pred = forecast.quantile(q_key)
    lower = forecast.quantile("0.1")
    upper = forecast.quantile("0.9")

    fig, (ax_forecast, ax_resid) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, constrained_layout=True
    )

    ax_forecast.plot(history_idx, history, label="history + truth", color="black")
    ax_forecast.plot(horizon_idx, pred, label=f"pred q{quantile}", color="tab:blue")
    ax_forecast.fill_between(
        horizon_idx, lower, upper, color="tab:blue", alpha=0.15, label="p10-p90"
    )
    ax_forecast.axvline(
        len(history_full) - prediction_length - 0.5, color="gray", linestyle="--", linewidth=1
    )
    ax_forecast.set_ylabel("value")
    ax_forecast.legend(loc="upper left")
    
    # Enhanced title with experiment settings
    title = f"Dataset: {dataset_name} | Term: {term} | Sample: {sample_idx}\n"
    title += f"Prediction Length: {prediction_length} | Total History: {len(history_full)}"
    if history_length is not None:
        title += f" | Displayed: {history_length}"
    ax_forecast.set_title(title, fontsize=10)

    residuals = pred - future_truth
    ax_resid.axhline(0.0, color="gray", linewidth=1)
    ax_resid.bar(horizon_idx, residuals, width=0.8, color="tab:orange")
    ax_resid.set_ylabel("pred - truth")
    ax_resid.set_xlabel(f"time index (prediction horizon: {prediction_length} steps)")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def eval_time(model_name, model_path, predictor_factory, estimation_samples=10):
    """
    Estimate the time needed to run evaluation on all datasets.
    
    Args:
        model_name: Name of the model for display
        model_path: Path to model for display
        predictor_factory: Callable that takes a dataset and returns a predictor
        estimation_samples: Number of samples to measure (default: 10)
    
    Returns:
        dict: Dictionary with timing statistics for each dataset configuration
    """
    short_datasets, med_long_datasets, all_datasets, dataset_properties_map = (
        setup_dataset()
    )
    
    setup_logger()
    
    print(f"Estimating evaluation time for {model_name} from {model_path}")
    print(f"Using {estimation_samples} samples per dataset for estimation")
    print("="*70)
    
    timing_results = {
        'model_name': model_name,
        'model_path': model_path,
        'estimation_samples': estimation_samples,
        'datasets': {}
    }
    total_estimated_time = 0
    
    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        print(f"\nDataset {ds_num + 1}/{len(all_datasets)}: {ds_name}")
        
        terms = ["short", "medium", "long"]
        for term in terms:
            if (term == "medium" or term == "long") and ds_name not in med_long_datasets.split():
                continue
            
            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                pretty_names = {
                    "saugeenday": "saugeen",
                    "temperature_rain_with_missing": "temperature_rain",
                    "kdd_cup_2018_with_missing": "kdd_cup_2018",
                    "car_parts_with_missing": "car_parts",
                }
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            
            ds_config = f"{ds_key}/{ds_freq}/{term}"
            
            # Initialize the dataset
            to_univariate = (
                False
                if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                else True
            )
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            prediction_length = get_prediction_length(term)
            dataset.prediction_length = prediction_length
            
            num_test_samples = len(dataset.test_data)
            measure_samples = min(estimation_samples, num_test_samples)
            
            print(f"  {term}: {num_test_samples} samples", end=" ")
            
            # Create predictor
            predictor = predictor_factory(dataset)
            
            # Measure time for estimation samples
            test_data_list = list(dataset.test_data)
            start_time = time.time()
            
            for i in range(measure_samples):
                item = test_data_list[i]
                # test_data may yield (input_dict, label) tuples; predictor expects input dicts
                if isinstance(item, tuple):
                    item = item[0]
                _ = list(predictor.predict([item]))
            
            elapsed_time = time.time() - start_time
            avg_time_per_sample = elapsed_time / measure_samples
            estimated_time = avg_time_per_sample * num_test_samples
            
            timing_results['datasets'][ds_config] = {
                'num_samples': num_test_samples,
                'measured_samples': measure_samples,
                'avg_time_per_sample': avg_time_per_sample,
                'estimated_total_seconds': estimated_time,
                'estimated_total_minutes': estimated_time / 60,
                'estimated_total_hours': estimated_time / 3600
            }
            
            total_estimated_time += estimated_time
            
            print(f"→ {avg_time_per_sample:.4f}s/sample → ~{estimated_time:.1f}s (~{estimated_time/60:.1f}min)")
    
    # Add total statistics
    timing_results['total'] = {
        'total_seconds': total_estimated_time,
        'total_minutes': total_estimated_time / 60,
        'total_hours': total_estimated_time / 3600
    }
    
    print("\n" + "="*70)
    print("TOTAL ESTIMATED TIME:")
    print(f"  {total_estimated_time:.2f} seconds")
    print(f"  {total_estimated_time/60:.2f} minutes")
    print(f"  {total_estimated_time/3600:.2f} hours")
    print("="*70)
    
    # Save to JSON file
    output_dir = f"{result_root}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    json_file_path = os.path.join(output_dir, "time_estimation.json")
    
    with open(json_file_path, 'w') as f:
        json.dump(timing_results, f, indent=2)
    
    print(f"\nTiming results saved to: {os.path.abspath(json_file_path)}")
    
    return timing_results


def save_predictions_csv(predictor, test_data, output_path, ds_config, prediction_length, freq):
    """
    Save predictions and ground truth to CSV (compatible with analyze_predictions.py).
    test_data yields (input_dict, label) or dict; label has future values.
    """
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    test_list = list(test_data)
    forecasts = predictor.predict(test_list)

    rows = []
    freq_offset = to_offset(freq)
    for sample_idx, (entry, fc) in enumerate(zip(test_list, forecasts)):
        if isinstance(entry, tuple):
            inp, label = entry
        else:
            inp, label = entry, None
        item_id = inp.get("item_id", str(sample_idx))
        start_raw = inp["start"]
        if hasattr(start_raw, "to_timestamp"):
            start = start_raw.to_timestamp()
        else:
            start = pd.Timestamp(start_raw)
        # Ground truth: from label if tuple, else last pred_len of target (generate_instances format)
        if label is not None:
            if isinstance(label, dict) and "target" in label:
                truth = np.asarray(label["target"]).flatten()
            else:
                truth = np.asarray(label).flatten()
        else:
            truth = np.asarray(inp["target"]).flatten()[-prediction_length:]
        if len(truth) < prediction_length:
            truth = np.pad(truth, (0, prediction_length - len(truth)), constant_values=np.nan)
        pred_median = fc.quantile("0.5")
        if pred_median.ndim > 1:
            pred_median = pred_median[:, 0] if pred_median.shape[1] >= 1 else pred_median[:, 0]
        pred_median = np.asarray(pred_median).flatten()[:prediction_length]
        hist_len = len(np.asarray(inp["target"]).flatten())
        for h in range(min(prediction_length, len(pred_median), len(truth))):
            ts = start + (hist_len + h) * freq_offset
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
            rows.append({"ds_config": ds_config, "sample_idx": sample_idx, "item_id": item_id,
                        "timestamp": ts_str, "dim": 0, "stat": "truth", "value": float(truth[h])})
            rows.append({"ds_config": ds_config, "sample_idx": sample_idx, "item_id": item_id,
                        "timestamp": ts_str, "dim": 0, "stat": "mean", "value": float(pred_median[h])})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path} ({len(df)} rows)")


def eval(model_name, model_path, predictor_factory, batch_size=1024, save_predictions_dir=None):
    short_datasets, med_long_datasets, all_datasets, dataset_properties_map = (
        setup_dataset()
    )

    # Instantiate the metrics
    metrics = get_metrics()

    setup_logger()

    output_dir = f"{result_root}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }

    # Check if file exists and read completed datasets
    csv_file_path = os.path.join(output_dir, "all_results.csv")
    done_datasets = check_done_datasets(csv_file_path)

    print(f"Evaluating {model_name} from {model_path}")

    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        terms = ["short", "medium", "long"]
        for term in terms:
            if (
                term == "medium" or term == "long"
            ) and ds_name not in med_long_datasets.split():
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            # Skip if already completed
            if ds_config in done_datasets:
                print(f"Skipping already completed dataset: {ds_config}")
                continue

            # Initialize the dataset
            to_univariate = (
                False
                if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
                else True
            )
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)
            prediction_length = get_prediction_length(term)
            # Override dataset's prediction_length with our custom values
            dataset.prediction_length = prediction_length
            print(f"Prediction length: {prediction_length}")
            print(f"Dataset size: {len(dataset.test_data)}")

            predictor = predictor_factory(dataset)

            # Measure the time taken for evaluation
            res = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )

            # Append the results to the CSV file
            write_result_to_csv(
                res,
                csv_file_path,
                ds_config,
                ds_key,
                dataset_properties_map,
                model_name,
            )

            # Optionally save predictions and ground truth
            if save_predictions_dir is not None:
                pred_csv = os.path.join(
                    save_predictions_dir,
                    f"{ds_config.replace('/', '_')}_predictions.csv",
                )
                save_predictions_csv(
                    predictor,
                    dataset.test_data,
                    pred_csv,
                    ds_config,
                    prediction_length,
                    dataset.freq,
                )

            print(f"Results for {ds_name} have been written to {csv_file_path}")

    show_results(model_name)
