"""
SHAP Interpretability Analysis for Moirai2 Time Series Model

This script performs SHAP (SHapley Additive exPlanations) analysis on Moirai2,
a pretrained transformer-based time series forecasting model.

Analyzes:
- Which historical time points (context) are most important for predictions
- Temporal attention patterns
- Feature importance across different time lags
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap
from pathlib import Path
from typing import List, Dict, Any

from plot.plot_utils import (
        plot_shap_summary,
        plot_temporal_importance,
        plot_time_windows,
        plot_shap_heatmap,
        plot_sample_with_shap_and_prediction,
    )

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'benchmark'))
sys.path.append(str(Path(__file__).parent.parent))

try:
    from benchmark.common import setup_dataset, get_prediction_length
    from gift_eval.data import Dataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure you have uni2ts and gift-eval installed")


class Moirai2Wrapper:
    """
    Wrapper for Moirai2 model to make it compatible with SHAP.
    """
    def __init__(self, model, prediction_length=48, device='cuda'):
        self.model = model
        self.prediction_length = prediction_length
        self.device = device
        
    def predict(self, X):
        """
        Predict function for SHAP.
        
        Args:
            X: numpy array of shape (batch_size, context_length)
        
        Returns:
            predictions: numpy array of shape (batch_size, prediction_length)
        """
        # Convert numpy to list of sequences (Moirai2 expects list)
        sequences = [x for x in X]
        
        # Get predictions
        with torch.no_grad():
            forecasts = self.model.predict(sequences)
        
        # Extract median forecast (quantile 0.5)
        # forecasts shape: (batch_size, num_quantiles, prediction_length)
        median_idx = len(forecasts[0]) // 2  # Assuming 0.5 is in the middle
        predictions = np.array([f[median_idx] for f in forecasts])
        
        # Return mean over prediction horizon for simplicity
        return predictions.mean(axis=1)
    
    def __call__(self, X):
        return self.predict(X)


def load_moirai2_model(model_path='Salesforce/moirai-2.0-R-small', 
                       prediction_length=48,
                       context_length=1000,
                       device='cuda'):
    """
    Load Moirai2 model.
    
    Args:
        model_path: Path to pretrained model
        prediction_length: Number of steps to forecast
        context_length: Context window size
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Moirai2 model
        wrapper: SHAP-compatible wrapper
    """
    print(f"Loading Moirai2 model from: {model_path}")
    print(f"  Prediction length: {prediction_length}")
    print(f"  Context length: {context_length}")
    print(f"  Device: {device}")
    
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load model
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(model_path),
        prediction_length=prediction_length,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    
    model.eval()
    
    # Create wrapper
    wrapper = Moirai2Wrapper(model, prediction_length=prediction_length, device=device)
    
    print("Model loaded successfully!")
    return model, wrapper


def load_data_from_dataset(dataset_name='sd/15T/short', 
                          context_length=1000,
                          prediction_length=48,
                          num_samples=100):
    """
    Load data from gift-eval dataset.
    
    Args:
        dataset_name: Dataset name
        context_length: Length of context window
        prediction_length: Length of prediction horizon
        num_samples: Number of time series to load
    
    Returns:
        X: Context data (num_samples, context_length)
        y: Future values for validation
        metadata: Additional information
    """
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Setup environment
        os.environ["GIFT_EVAL"] = str(Path(__file__).parent.parent / 
                                      'gift-eval' / 'datasets')
        
        # Load dataset
        dataset = Dataset.load(dataset_name)
        
        # Collect time series from test set
        X_list = []
        y_list = []
        
        for i, item in enumerate(dataset.test):
            if i >= num_samples:
                break
                
            target = item['target']
            if len(target.shape) > 1:
                target = target[:, 0]  # Take first dimension if multivariate
            
            # Extract context and future
            if len(target) >= context_length + prediction_length:
                context = target[-(context_length + prediction_length):-prediction_length]
                future = target[-prediction_length:]
                X_list.append(context)
                y_list.append(future)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Loaded {len(X)} time series")
        print(f"  Context shape: {X.shape}")
        print(f"  Future shape: {y.shape}")
        
        metadata = {
            'dataset_name': dataset_name,
            'context_length': context_length,
            'prediction_length': prediction_length,
            'num_samples': len(X)
        }
        
        return X, y, metadata
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data...")
        return load_synthetic_data(context_length, prediction_length, num_samples)


def load_synthetic_data(context_length=1000, prediction_length=48, num_samples=100):
    """
    Generate synthetic time series data.
    
    Args:
        context_length: Length of context window
        prediction_length: Length of prediction horizon
        num_samples: Number of time series
    
    Returns:
        X: Context data
        y: Future values
        metadata: Additional information
    """
    print(f"Generating synthetic data: {num_samples} series, context_length={context_length}")
    
    X_list = []
    y_list = []
    
    for _ in range(num_samples):
        # Generate time series with trend + seasonality + noise
        t = np.arange(context_length + prediction_length)
        trend = 0.05 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 96)  # Daily pattern (15min intervals)
        noise = np.random.normal(0, 2, len(t))
        ts = 50 + trend + seasonal + noise
        
        X_list.append(ts[:context_length])
        y_list.append(ts[context_length:context_length + prediction_length])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    metadata = {
        'dataset_name': 'synthetic',
        'context_length': context_length,
        'prediction_length': prediction_length,
        'num_samples': num_samples
    }
    
    return X, y, metadata


def analyze_shap_moirai2(wrapper, X, max_samples=20, context_length=1000, sample_seed=42):
    """
    Perform SHAP analysis on Moirai2 model.
    
    Args:
        wrapper: Model wrapper
        X: Context data
        max_samples: Number of samples to analyze (small number due to computation)
        context_length: Context window size
    
    Returns:
        shap_values: SHAP values
        X_sample: Sample data used
    """
    print(f"\nPerforming SHAP analysis on Moirai2...")
    
    # Sample data for analysis (SHAP on deep models is computationally expensive)
    if len(X) > max_samples:
        rng = np.random.default_rng(sample_seed)
        indices = rng.permutation(len(X))[:max_samples]
        X_sample = X[indices]
    else:
        indices = np.arange(len(X))
        X_sample = X
    
    print(f"Analyzing {len(X_sample)} samples...")
    print("Note: This may take several minutes for deep learning models...")
    
    # For deep learning models, we can use different explainers:
    # 1. GradientExplainer - uses gradients (requires model)
    # 2. KernelExplainer - model-agnostic but slow
    # 3. SamplingExplainer - faster approximation
    
    # We'll use Kernel SHAP with a small background dataset
    print("Using Kernel SHAP (model-agnostic method)...")
    background = shap.sample(X_sample, min(10, len(X_sample)))
    explainer = shap.KernelExplainer(wrapper.predict, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    print("SHAP analysis complete!")
    
    return explainer, shap_values, X_sample, indices




def run_single_experiment(dataset_name, year, freq, context_length, prediction_length,
                         model_path, device, num_samples, shap_samples, base_output_dir,
                         use_shared_data=False, shared_data_base_dir=None):
    """
    Run a single SHAP analysis experiment with given configuration.
    
    Args:
        dataset_name: Dataset name (e.g., 'sd', 'ca')
        year: Year (e.g., '2019')
        freq: Frequency (e.g., '15T')
        context_length: Historical context length
        prediction_length: Prediction horizon
        model_path: Path to Moirai2 model
        device: 'cuda' or 'cpu'
        num_samples: Number of samples to load
        shap_samples: Number of samples for SHAP analysis
        base_output_dir: Base directory for outputs
    
    Returns:
        results: Dictionary with experiment results
    """
    exp_name = f"{dataset_name}_{year}_{freq}_ctx{context_length}_pred{prediction_length}"
    output_dir = os.path.join(base_output_dir, exp_name)
    
    print("\n" + "=" * 80)
    print(f"Running Experiment: {exp_name}")
    print("=" * 80)
    print(f"  Dataset: {dataset_name}/{year}")
    print(f"  Frequency: {freq}")
    print(f"  Context length: {context_length}")
    print(f"  Prediction length: {prediction_length}")
    print(f"  SHAP samples: {shap_samples}")
    print(f"  Output dir: {output_dir}")
    
    # Load data (prefer shared npy to match Ridge)
    if use_shared_data and shared_data_base_dir:
        shared_dir = os.path.join(shared_data_base_dir, exp_name)
        x_path = os.path.join(shared_dir, 'X_context.npy')
        y_path = os.path.join(shared_dir, 'y_future.npy')
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"\nLoading shared data from: {shared_dir}")
            X = np.load(x_path)
            y = np.load(y_path)
            if X.ndim == 3 and y.ndim == 3:
                X = X[-1]
                y = y[-1]
            item_ids = []
            item_ids_path = os.path.join(shared_dir, 'item_ids.txt')
            if os.path.exists(item_ids_path):
                with open(item_ids_path, 'r') as f:
                    for line in f:
                        if line.strip() and line[0].isdigit() and '. ' in line:
                            item_ids.append(line.split('. ', 1)[1].strip())
            print(f"Loaded shared data: X shape {X.shape}, y shape {y.shape}")
        else:
            print(f"Shared data not found for config: {shared_dir}")
            print("Falling back to h5 loading...")
            use_shared_data = False

    if not use_shared_data:
        # Load data directly from h5 using same method as prepare_shap_data.py
        data_dir = Path(__file__).parent.parent / 'data' / dataset_name
        h5_file = data_dir / f"{dataset_name}_his_raw_{year}.h5"

        if not h5_file.exists():
            h5_file = data_dir / f"{dataset_name}_his_{year}.h5"

        if not h5_file.exists():
            print(f"ERROR: Cannot find h5 file: {h5_file}")
            return None

        print(f"\nLoading data from: {h5_file}")
        df = pd.read_hdf(h5_file)
        print(f"  Original shape: {df.shape}")

        # Resample
        df = df.resample(freq).mean().round(0)
        print(f"  After resampling to {freq}: {df.shape}")

        # Split into train/val/test (60/20/20)
        num_samples_total = len(df)
        num_train = round(num_samples_total * 0.6)
        num_val = round(num_samples_total * 0.2)

        # Use test split (last 20%)
        test_df = df.iloc[num_train + num_val:]
        print(f"  Test split shape: {test_df.shape}")
        print(f"  Test start timestamp: {test_df.index[0]}")

        # Remove columns with NaN
        test_df = test_df.dropna(axis=1, how='any')
        print(f"  After removing NaN columns: {test_df.shape}")

        if test_df.shape[1] == 0:
            print("ERROR: No columns left after removing NaN values")
            return None

        # Collect samples
        X_list = []
        y_list = []
        item_ids = []

        # Randomly sample from available time series
        available_columns = list(test_df.columns)
        np.random.seed(42)
        np.random.shuffle(available_columns)

        count = 0
        for col in available_columns:
            if count >= num_samples:
                break

            ts = test_df[col].values

            # Skip if not enough data points
            if len(ts) < context_length + prediction_length:
                continue

            # Extract from the end of the time series
            context = ts[-(context_length + prediction_length):-prediction_length]
            future = ts[-prediction_length:]

            # Verify no NaN
            if np.isnan(context).any() or np.isnan(future).any():
                continue

            X_list.append(context)
            y_list.append(future)
            item_ids.append(str(col))
            count += 1

        if len(X_list) == 0:
            print("ERROR: No valid time series found")
            return None

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
    
    print(f"\nLoaded {len(X)} time series samples")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Load Moirai2 model
    try:
        model, wrapper = load_moirai2_model(
            model_path, 
            prediction_length, 
            context_length,
            device
        )
    except Exception as e:
        print(f"ERROR loading Moirai2 model: {e}")
        return None

    # Evaluate model (MAE/MSE/MASE) on loaded samples using mean over horizon
    try:
        os.makedirs(output_dir, exist_ok=True)
        y_true_mean = y.mean(axis=1)
        y_pred_mean = wrapper.predict(X)
        mse = np.mean((y_true_mean - y_pred_mean) ** 2)
        mae = np.mean(np.abs(y_true_mean - y_pred_mean))
        # MASE: scale by in-sample naive forecast error from context
        if X.shape[1] > 1:
            naive_denom = np.mean(np.abs(np.diff(X, axis=1)))
            mase = mae / naive_denom if naive_denom != 0 else np.nan
        else:
            mase = np.nan
        print(f"\nModel performance on loaded samples:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MASE: {mase:.4f}" if not np.isnan(mase) else "  MASE: NaN")

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
        print(f"Metrics saved to: {metrics_path}")
    except Exception as e:
        print(f"WARNING: Failed to compute MAE/MSE/MASE: {e}")
    
    # SHAP analysis
    try:
        explainer, shap_values, X_sample, sample_indices = analyze_shap_moirai2(
            wrapper, X,
            max_samples=shap_samples,
            context_length=context_length,
            sample_seed=42
        )
    except Exception as e:
        print(f"ERROR in SHAP analysis: {e}")
        return None
   # import pdb; pdb.set_trace()
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_shap_summary(shap_values, X_sample, explainer, output_dir=output_dir, model_name=exp_name)
    plot_temporal_importance(shap_values, context_length, output_dir, exp_name)
    plot_time_windows(shap_values, context_length, output_dir, exp_name)
    plot_shap_heatmap(shap_values, X_sample, output_dir, exp_name)
    
    # Plot detailed examples with predictions
    y_sample = y[sample_indices]
    def predict_fn(history):
        predicted_future = wrapper.model.predict([history])
        median_idx = len(predicted_future[0]) // 2
        return np.array(predicted_future[0][median_idx])

    plot_sample_with_shap_and_prediction(X_sample, y_sample, shap_values,
                                         predict_fn, num_examples=min(3, len(X_sample)),
                                         output_dir=output_dir,
                                         model_name=exp_name)
    
    # Save SHAP values and metadata
    np.save(f'{output_dir}/shap_values.npy', shap_values)
    np.save(f'{output_dir}/X_sample.npy', X_sample)
    np.save(f'{output_dir}/y_sample.npy', y_sample)
    
    # Save item IDs
    with open(f'{output_dir}/item_ids.txt', 'w') as f:
        f.write(f"Total samples: {len(item_ids)}\n")
        f.write(f"SHAP analyzed samples: {shap_samples}\n\n")
        for i, idx in enumerate(sample_indices):
            f.write(f"{i+1}. {item_ids[idx]}\n")
    
    # Save configuration and results
    results = {
        'dataset': dataset_name,
        'year': year,
        'freq': freq,
        'context_length': context_length,
        'prediction_length': prediction_length,
        'num_samples': len(X),
        'shap_samples': shap_samples,
        'X_shape': X.shape,
        'y_shape': y.shape,
        'X_mean': float(X.mean()),
        'X_std': float(X.std()),
        'X_min': float(X.min()),
        'X_max': float(X.max()),
        'mse': float(mse) if 'mse' in locals() else None,
        'mae': float(mae) if 'mae' in locals() else None,
        'mase': float(mase) if 'mase' in locals() else None,
    }
    
    # Save metadata
    with open(f'{output_dir}/metadata.txt', 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write("=" * 60 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nExperiment {exp_name} completed successfully!")
    print(f"Results saved to: {output_dir}")
    
    return results


def main():
    """Main function to run Moirai2 SHAP analysis experiments."""
    print("=" * 80)
    print("SHAP Interpretability Analysis for Moirai2 Time Series Model")
    print("Multi-Configuration Experiments")
    print("=" * 80)
    
    # ===== CONFIGURATION =====
    # Dataset configuration
    DATASET_NAME = 'sd'
    YEAR = '2019'
    FREQ = '15T'
    
    # Model configuration
    MODEL_PATH = 'Salesforce/moirai-2.0-R-small'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Experiment configurations
    CONTEXT_LENGTHS = [48, 98, 196, 336, 720]  # Historical lengths to test
    PREDICTION_LENGTHS = [3, 6, 12, 48]  # Prediction horizons to test
    
    # Sampling configuration
    NUM_SAMPLES = 100  # Number of time series to load
    SHAP_SAMPLES = 10  # Number of samples for SHAP analysis (small for computational efficiency)
    
    # Output configuration
    BASE_OUTPUT_DIR = './moirai2_shap_experiments'

    # Shared data configuration (to match Ridge data)
    USE_SHARED_DATA = True
    SHARED_DATA_BASE_DIR = '/home/defu/workspace/LargeST/results/shared_shap_data_experiments'
    
    # ===== END CONFIGURATION =====
    
    # ===== END CONFIGURATION =====
    
    print(f"\nGlobal Configuration:")
    print(f"  Dataset: {DATASET_NAME}/{YEAR}")
    print(f"  Frequency: {FREQ}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Context lengths: {CONTEXT_LENGTHS}")
    print(f"  Prediction lengths: {PREDICTION_LENGTHS}")
    print(f"  Samples per experiment: {NUM_SAMPLES}")
    print(f"  SHAP samples per experiment: {SHAP_SAMPLES}")
    print(f"  Base output directory: {BASE_OUTPUT_DIR}")
    
    # Create summary file
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    summary_file = os.path.join(BASE_OUTPUT_DIR, 'experiments_summary.txt')
    
    all_results = []
    total_experiments = len(CONTEXT_LENGTHS) * len(PREDICTION_LENGTHS)
    
    print(f"\n{'='*80}")
    print(f"Running {total_experiments} experiments...")
    print(f"{'='*80}\n")
    
    experiment_count = 0
    for context_length in CONTEXT_LENGTHS:
        for prediction_length in PREDICTION_LENGTHS:
            experiment_count += 1
            print(f"\n[Experiment {experiment_count}/{total_experiments}]")
            
            results = run_single_experiment(
                dataset_name=DATASET_NAME,
                year=YEAR,
                freq=FREQ,
                context_length=context_length,
                prediction_length=prediction_length,
                model_path=MODEL_PATH,
                device=DEVICE,
                num_samples=NUM_SAMPLES,
                shap_samples=SHAP_SAMPLES,
                base_output_dir=BASE_OUTPUT_DIR,
                use_shared_data=USE_SHARED_DATA,
                shared_data_base_dir=SHARED_DATA_BASE_DIR
            )
            
            if results:
                all_results.append(results)
            else:
                print(f"WARNING: Experiment failed for context={context_length}, pred={prediction_length}")
    
    # Save summary
    with open(summary_file, 'w') as f:
        f.write("MOIRAI2 SHAP Analysis - Experiments Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {DATASET_NAME}/{YEAR}\n")
        f.write(f"Frequency: {FREQ}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"Total experiments: {len(all_results)}/{total_experiments}\n")
        f.write(f"Samples per experiment: {NUM_SAMPLES}\n")
        f.write(f"SHAP samples per experiment: {SHAP_SAMPLES}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(all_results, 1):
            f.write(f"Experiment {i}:\n")
            f.write(f"  Context length: {result['context_length']}\n")
            f.write(f"  Prediction length: {result['prediction_length']}\n")
            f.write(f"  Samples: {result['num_samples']}\n")
            f.write(f"  Data range: [{result['X_min']:.2f}, {result['X_max']:.2f}]\n")
            f.write(f"  Output: {DATASET_NAME}_{YEAR}_{FREQ}_ctx{result['context_length']}_pred{result['prediction_length']}/\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"All experiments complete!")
    print(f"Total successful: {len(all_results)}/{total_experiments}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print(f"Summary: {summary_file}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
