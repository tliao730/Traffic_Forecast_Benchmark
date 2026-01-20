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


def analyze_shap_moirai2(wrapper, X, max_samples=20, context_length=1000):
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
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
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
    
    return shap_values, X_sample


def plot_temporal_importance(shap_values, context_length=1000,
                             output_dir='./moirai2_shap_plots',
                             model_name='moirai2'):
    """
    Plot temporal importance: which time lags are most important.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate mean absolute SHAP value for each time point
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Full temporal plot
    plt.figure(figsize=(16, 6))
    time_points = np.arange(context_length)
    plt.plot(time_points, mean_abs_shap, linewidth=1.5, color='steelblue', alpha=0.7)
    plt.fill_between(time_points, 0, mean_abs_shap, alpha=0.3, color='steelblue')
    
    plt.xlabel('Time Steps Back (from most recent)', fontsize=12)
    plt.ylabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'{model_name.upper()} - Temporal Feature Importance\n'
              f'Which historical time points matter most for prediction', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight top 10 most important time points
    top_10_indices = np.argsort(mean_abs_shap)[-10:]
    plt.scatter(top_10_indices, mean_abs_shap[top_10_indices], 
               color='red', s=100, zorder=5, label='Top 10 important points')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_temporal_importance_full.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_temporal_importance_full.png")
    
    # Zoomed plot - most recent 200 points
    zoom_length = min(200, context_length)
    plt.figure(figsize=(14, 6))
    plt.bar(range(zoom_length), mean_abs_shap[:zoom_length], 
           color='steelblue', alpha=0.7)
    plt.xlabel(f'Most Recent {zoom_length} Time Steps', fontsize=12)
    plt.ylabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'{model_name.upper()} - Recent History Importance (Zoomed)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_temporal_importance_recent.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_temporal_importance_recent.png")
    
    # Print top important time points
    print(f"\nTop 20 most important time points (steps back from current):")
    top_20_indices = np.argsort(mean_abs_shap)[-20:][::-1]
    for i, idx in enumerate(top_20_indices, 1):
        print(f"  {i}. Position {idx} (t-{context_length-idx}): {mean_abs_shap[idx]:.6f}")


def plot_time_windows(shap_values, context_length=1000,
                     output_dir='./moirai2_shap_plots',
                     model_name='moirai2'):
    """
    Analyze importance by time windows.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Define time windows - same as Ridge for fair comparison
    # Divide into three equal parts
    n_lags = context_length
    recent = mean_abs_shap[:n_lags//3].sum()  # First 1/3 (most recent)
    medium = mean_abs_shap[n_lags//3:2*n_lags//3].sum()  # Middle 1/3
    distant = mean_abs_shap[2*n_lags//3:].sum()  # Last 1/3 (most distant)
    
    # Create bar plot
    plt.figure(figsize=(12, 7))
    third = n_lags // 3
    windows = [
        f'Recent\n(lag 1-{third})',
        f'Medium\n(lag {third+1}-{third*2})',
        f'Distant\n(lag {third*2+1}-{n_lags})'
    ]
    values = [recent, medium, distant]
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    
    bars = plt.bar(windows, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    plt.ylabel('Total |SHAP value|', fontsize=13)
    plt.title(f'{model_name.upper()} - Importance by Time Window', fontsize=15, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}\n({100*value/sum(values):.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_time_windows.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_time_windows.png")
    
    # Print statistics
    total = sum(values)
    print(f"\nImportance by time window:")
    print(f"  Recent (lag 1-{third}): {recent:.4f} ({100*recent/total:.1f}%)")
    print(f"  Medium (lag {third+1}-{third*2}): {medium:.4f} ({100*medium/total:.1f}%)")
    print(f"  Distant (lag {third*2+1}-{n_lags}): {distant:.4f} ({100*distant/total:.1f}%)")


def plot_shap_heatmap(shap_values, X_sample, 
                     output_dir='./moirai2_shap_plots',
                     model_name='moirai2'):
    """
    Create heatmap of SHAP values across samples and time.
    """
    plt.figure(figsize=(16, 8))
    
    # Downsample for visualization if needed
    if X_sample.shape[1] > 200:
        # Show every nth point to make it readable
        step = X_sample.shape[1] // 200
        shap_plot = shap_values[:, ::step]
        time_points = np.arange(0, X_sample.shape[1], step)
    else:
        shap_plot = shap_values
        time_points = np.arange(X_sample.shape[1])
    
    plt.imshow(shap_plot, aspect='auto', cmap='RdBu_r', 
              interpolation='nearest')
    plt.colorbar(label='SHAP value')
    plt.xlabel('Time Steps Back', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.title(f'{model_name.upper()} - SHAP Values Heatmap\n'
              f'Red = Positive impact, Blue = Negative impact', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_heatmap.png")


def plot_sample_with_shap_and_prediction(wrapper, X_sample, y_sample, shap_values,
                                        num_examples=3,
                                        output_dir='./moirai2_shap_plots',
                                        model_name='moirai2'):
    """
    Plot individual examples showing:
    1. Historical data (context)
    2. SHAP values for each time point
    3. Model predictions vs actual future values
    
    Args:
        wrapper: Model wrapper for predictions
        X_sample: Historical context data (n_samples, context_length)
        y_sample: Actual future values (n_samples, prediction_length)
        shap_values: SHAP values (n_samples, context_length)
        num_examples: Number of examples to plot
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to available samples
    num_examples = min(num_examples, len(X_sample))
    
    for idx in range(num_examples):
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        context_length = X_sample.shape[1]
        prediction_length = y_sample.shape[1]
        
        # Prepare time axis
        historical_time = np.arange(-context_length, 0)
        future_time = np.arange(0, prediction_length)
        
        # Get data for this sample
        historical_data = X_sample[idx]
        actual_future = y_sample[idx]
        shap_vals = shap_values[idx]
        
        # Get model prediction
        predicted_future = wrapper.model.predict([historical_data])
        # Extract median forecast (quantile 0.5)
        # predicted_future is a list with one element (batch size 1)
        # Shape: (num_quantiles, prediction_length)
        median_idx = len(predicted_future[0]) // 2
        predicted_future = np.array(predicted_future[0][median_idx])  # Get all time points for median quantile
        
        # Plot 1: Historical data with SHAP values (color-coded)
        ax1 = axes[0]
        scatter = ax1.scatter(historical_time, historical_data, 
                            c=shap_vals, cmap='RdYlGn', 
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.plot(historical_time, historical_data, 'b-', alpha=0.3, linewidth=1)
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('SHAP Value', fontsize=10)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Present')
        ax1.set_xlabel('Time Steps', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'Sample {idx+1}: Historical Data (colored by SHAP importance)', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: SHAP values bar chart
        ax2 = axes[1]
        colors = ['red' if x > 0 else 'blue' for x in shap_vals]
        ax2.bar(historical_time, shap_vals, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Time Steps Back', fontsize=12)
        ax2.set_ylabel('SHAP Value', fontsize=12)
        ax2.set_title('SHAP Values: Contribution of Each Historical Point\n'
                     'Red = Increases prediction, Blue = Decreases prediction', 
                     fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Full time series with prediction
        ax3 = axes[2]
        # Historical
        ax3.plot(historical_time, historical_data, 'b-', linewidth=2, 
                label='Historical Data', alpha=0.7)
        # Actual future
        ax3.plot(future_time, actual_future, 'g-', linewidth=2, 
                marker='o', markersize=6, label='Actual Future', alpha=0.8)
        # Predicted future
        ax3.plot(future_time, predicted_future, 'r--', linewidth=2, 
                marker='s', markersize=6, label='Predicted Future', alpha=0.8)
        
        # Shade future region
        ax3.axvspan(0, prediction_length, alpha=0.1, color='yellow', label='Forecast Horizon')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Present')
        
        ax3.set_xlabel('Time Steps', fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Full Time Series: History + Actual vs Predicted Future', 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(alpha=0.3)
        
        # Calculate error metrics
        mae = np.mean(np.abs(actual_future - predicted_future))
        mse = np.mean((actual_future - predicted_future) ** 2)
        ax3.text(0.02, 0.98, f'MAE: {mae:.2f}\nMSE: {mse:.2f}', 
                transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_sample_{idx+1}_detailed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {model_name}_sample_{idx+1}_detailed.png")
    
    print(f"\nGenerated {num_examples} detailed sample visualizations!")


def run_single_experiment(dataset_name, year, freq, context_length, prediction_length,
                         model_path, device, num_samples, shap_samples, base_output_dir):
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
    
    # SHAP analysis
    try:
        shap_values, X_sample = analyze_shap_moirai2(
            wrapper, X, 
            max_samples=shap_samples,
            context_length=context_length
        )
    except Exception as e:
        print(f"ERROR in SHAP analysis: {e}")
        return None
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_temporal_importance(shap_values, context_length, output_dir, exp_name)
    plot_time_windows(shap_values, context_length, output_dir, exp_name)
    plot_shap_heatmap(shap_values, X_sample, output_dir, exp_name)
    
    # Plot detailed examples with predictions
    y_sample = y[:shap_samples] if len(y) >= shap_samples else y
    plot_sample_with_shap_and_prediction(wrapper, X_sample, y_sample, shap_values,
                                        num_examples=min(3, len(X_sample)), 
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
        for i, item_id in enumerate(item_ids[:shap_samples]):
            f.write(f"{i+1}. {item_id}\n")
    
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
                base_output_dir=BASE_OUTPUT_DIR
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
