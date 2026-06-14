"""
SHAP Interpretability Analysis for Chronos-2 Time Series Model

This script performs SHAP (SHapley Additive exPlanations) analysis on Chronos-2,
a foundation model for time series forecasting.

It loads shared data (npy) to ensure identical samples with Ridge/Moirai2 and
runs multi-configuration experiments.
"""

import os
import sys
import numpy as np
import torch
import shap
from pathlib import Path
import matplotlib; matplotlib.use("Agg")

sys.path.append(str(Path(__file__).parent))
from plot.plot_utils import (
    plot_shap_summary,
    plot_temporal_importance,
    plot_time_windows,
    plot_shap_heatmap,
    plot_sample_with_shap_and_prediction,
)

try:
    from chronos import BaseChronosPipeline, Chronos2Pipeline
except ImportError as e:
    print(f"Warning: Could not import chronos: {e}")
    print("Install chronos-forecasting>=2.0 to use Chronos-2.")


class Chronos2Wrapper:
    """Wrapper for Chronos-2 to make it compatible with SHAP."""

    def __init__(self, model_path, prediction_length=48, device_map="cuda",
                 quantile_levels=None, batch_size=64, predict_batches_jointly=False):
        self.prediction_length = prediction_length
        self.device_map = device_map
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        self.batch_size = batch_size
        self.predict_batches_jointly = predict_batches_jointly

        self.pipeline = BaseChronosPipeline.from_pretrained(model_path, device_map=device_map)
        if not isinstance(self.pipeline, Chronos2Pipeline):
            raise ValueError("Loaded pipeline is not Chronos2Pipeline. Use a Chronos-2 model.")

    def _batched_predict_quantiles(self, X):
        all_outputs = []
        for start in range(0, len(X), self.batch_size):
            batch = X[start:start + self.batch_size]
            inputs = [torch.tensor(x) for x in batch]
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=inputs,
                prediction_length=self.prediction_length,
                batch_size=len(inputs),
                quantile_levels=self.quantile_levels,
                predict_batches_jointly=self.predict_batches_jointly,
            )

            if isinstance(quantiles, list):
                quantiles = np.stack([np.array(q) for q in quantiles], axis=0)
            else:
                quantiles = np.array(quantiles)

            if quantiles.ndim == 2:
                quantiles = quantiles[None, ...]

            all_outputs.append(quantiles)

        return np.concatenate(all_outputs, axis=0)

    def predict_full(self, X):
        """Return median forecast for each sample (batch, prediction_length)."""
        quantiles = self._batched_predict_quantiles(X)
        if quantiles.ndim == 2:
            return quantiles

        median_idx = int(np.argmin(np.abs(np.array(self.quantile_levels) - 0.5)))
        median_idx = min(median_idx, quantiles.shape[1] - 1)
        return quantiles[:, median_idx, :]

    def predict_mean(self, X):
        """Return mean over horizon for SHAP (batch,)."""
        preds = self.predict_full(X)
        if preds.ndim == 1:
            return preds
        if preds.ndim == 2:
            return preds.mean(axis=1)
        return preds.reshape(preds.shape[0], -1).mean(axis=1)

    def __call__(self, X):
        return self.predict_mean(X)


def load_shared_data(shared_dir):
    X = np.load(os.path.join(shared_dir, "X_context.npy"))
    y = np.load(os.path.join(shared_dir, "y_future.npy"))

    # If data has multiple windows: (num_windows, num_series, length)
    if X.ndim == 3:
        X = X[-1]
    if y.ndim == 3:
        y = y[-1]

    return X.astype(np.float32), y.astype(np.float32)


def analyze_shap_chronos2(wrapper, X, max_samples=20, sample_seed=42):
    """Kernel SHAP on Chronos-2 (expensive)."""
    if len(X) > max_samples:
        rng = np.random.default_rng(sample_seed)
        indices = rng.permutation(len(X))[:max_samples]
        X_sample = X[indices]
    else:
        indices = np.arange(len(X))
        X_sample = X

    background = shap.sample(X_sample, min(10, len(X_sample)))
    #import pdb; pdb.set_trace()
    explainer = shap.KernelExplainer(wrapper.predict_mean, background)
    #import pdb; pdb.set_trace()
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    #import pdb; pdb.set_trace()

    return explainer, shap_values, X_sample, indices


def run_single_experiment(dataset_name, year, freq, context_length, prediction_length,
                          model_path, device_map, num_samples, shap_samples,
                          base_output_dir, shared_data_base_dir):
    exp_name = f"{dataset_name}_{year}_{freq}_ctx{context_length}_pred{prediction_length}"
    output_dir = os.path.join(base_output_dir, exp_name)

    shared_dir = os.path.join(shared_data_base_dir, exp_name)
    if not os.path.exists(shared_dir):
        print(f"Shared data not found: {shared_dir}")
        return None

    print("\n" + "=" * 80)
    print(f"Running Experiment: {exp_name}")
    print("=" * 80)
    print(f"  Shared data: {shared_dir}")
    print(f"  Output dir: {output_dir}")

    X, y = load_shared_data(shared_dir)
    if num_samples is not None and num_samples < len(X):
        X = X[:num_samples]
        y = y[:num_samples]

    print(f"Loaded shared data: X shape {X.shape}, y shape {y.shape}")

    wrapper = Chronos2Wrapper(
        model_path=model_path,
        prediction_length=prediction_length,
        device_map=device_map,
        quantile_levels=[0.1, 0.5, 0.9],
        batch_size=64,
        predict_batches_jointly=False,
    )

    # Evaluate metrics (mean over horizon)
    y_true_mean = y.mean(axis=1)
    y_pred_mean = wrapper.predict_mean(X)
    if np.ndim(y_pred_mean) > 1:
        y_pred_mean = np.mean(y_pred_mean, axis=1)
    mse = np.mean((y_true_mean - y_pred_mean) ** 2)
    mae = np.mean(np.abs(y_true_mean - y_pred_mean))
    if X.shape[1] > 1:
        naive_denom = np.mean(np.abs(np.diff(X, axis=1)))
        mase = mae / naive_denom if naive_denom != 0 else np.nan
    else:
        mase = np.nan

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Context length: {context_length}\n")
        f.write(f"Prediction length: {prediction_length}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write("Evaluation: mean over horizon\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        if np.isnan(mase):
            f.write("MASE: NaN\n")
        else:
            f.write(f"MASE: {mase:.6f}\n")

    explainer, shap_values, X_sample, sample_indices = analyze_shap_chronos2(
        wrapper, X, max_samples=shap_samples, sample_seed=42
    )
    plot_shap_summary(shap_values, X_sample, explainer, output_dir=output_dir, model_name=exp_name)
    plot_temporal_importance(shap_values, context_length, output_dir=output_dir, model_name=exp_name)
    plot_time_windows(shap_values, context_length, output_dir=output_dir, model_name=exp_name)
    plot_shap_heatmap(shap_values, X_sample, output_dir=output_dir, model_name=exp_name)

    def predict_future_fn(history):
        preds = wrapper.predict_full(np.asarray(history)[None, :])[0]
        preds = np.array(preds)
        #import pdb; pdb.set_trace()
        if preds.ndim > 1:
            return preds.mean(axis=1)
        return preds

    y_sample = y[sample_indices]
    plot_sample_with_shap_and_prediction(
        X_sample,
        y_sample,
        shap_values,
        predict_future_fn,
        num_examples=min(3, len(X_sample)),
        output_dir=output_dir,
        model_name=exp_name,
    )

    np.save(f'{output_dir}/shap_values.npy', shap_values)
    np.save(f'{output_dir}/X_sample.npy', X_sample)
    np.save(f'{output_dir}/y_sample.npy', y_sample)

    return {
        'context_length': context_length,
        'prediction_length': prediction_length,
        'num_samples': len(X),
        'mse': float(mse),
        'mae': float(mae),
        'mase': float(mase) if not np.isnan(mase) else None,
    }


def main():
    print("=" * 80)
    print("SHAP Interpretability Analysis for Chronos-2")
    print("Multi-Configuration Experiments")
    print("=" * 80)

    DATASET_NAME = 'sd'
    YEAR = '2019'
    FREQ = '15T'

    MODEL_PATH = 'amazon/chronos-2'
    DEVICE_MAP = 'cuda' if torch.cuda.is_available() else 'cpu'

    CONTEXT_LENGTHS = [48, 98, 196, 336, 720]
    PREDICTION_LENGTHS = [3, 6, 12, 48]

    NUM_SAMPLES = None
    SHAP_SAMPLES = 10

    BASE_OUTPUT_DIR = './chronos2_shap_experiments'
    SHARED_DATA_BASE_DIR = '/home/defu/workspace/LargeST/results/shared_shap_data_experiments'

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    summary_file = os.path.join(BASE_OUTPUT_DIR, 'experiments_summary.txt')

    all_results = []
    total_experiments = len(CONTEXT_LENGTHS) * len(PREDICTION_LENGTHS)

    exp_count = 0
    for context_length in CONTEXT_LENGTHS:
        for prediction_length in PREDICTION_LENGTHS:
            exp_count += 1
            print(f"\n[Experiment {exp_count}/{total_experiments}]")
            results = run_single_experiment(
                dataset_name=DATASET_NAME,
                year=YEAR,
                freq=FREQ,
                context_length=context_length,
                prediction_length=prediction_length,
                model_path=MODEL_PATH,
                device_map=DEVICE_MAP,
                num_samples=NUM_SAMPLES,
                shap_samples=SHAP_SAMPLES,
                base_output_dir=BASE_OUTPUT_DIR,
                shared_data_base_dir=SHARED_DATA_BASE_DIR,
            )
            if results:
                all_results.append(results)

    with open(summary_file, 'w') as f:
        f.write("CHRONOS-2 SHAP Analysis - Experiments Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {DATASET_NAME}/{YEAR}\n")
        f.write(f"Frequency: {FREQ}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Device: {DEVICE_MAP}\n\n")
        f.write(f"Total experiments: {len(all_results)}/{total_experiments}\n")
        f.write(f"SHAP samples per experiment: {SHAP_SAMPLES}\n\n")
        f.write("=" * 80 + "\n\n")
        for i, result in enumerate(all_results, 1):
            f.write(f"Experiment {i}:\n")
            f.write(f"  Context length: {result['context_length']}\n")
            f.write(f"  Prediction length: {result['prediction_length']}\n")
            f.write(f"  Samples: {result['num_samples']}\n")
            f.write(f"  MSE: {result['mse']:.6f}\n")
            f.write(f"  MAE: {result['mae']:.6f}\n")
            f.write(f"  MASE: {result['mase']}\n")
            f.write("\n")

    print("\n" + "=" * 80)
    print("All experiments complete!")
    print(f"Total successful: {len(all_results)}/{total_experiments}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print(f"Summary: {summary_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
