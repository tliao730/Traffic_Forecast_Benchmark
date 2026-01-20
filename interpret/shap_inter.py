"""
SHAP Interpretability Analysis for Time Series Models

This script performs SHAP (SHapley Additive exPlanations) analysis on time series
forecasting models to understand the contribution of historical time points to predictions.

Supported models:
- Ridge Regression
- XGBoost
- LightGBM
- Random Forest

The script analyzes how each of the 48 historical time points contributes to the forecast.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Add parent directory to path to import from benchmark
sys.path.append(str(Path(__file__).parent.parent / 'benchmark'))
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# Try to import from benchmark
try:
    from benchmark.common import setup_dataset, get_prediction_length
    from gift_eval.data import Dataset
except ImportError as e:
    print(f"Warning: Could not import from benchmark: {e}")
    print("Running in standalone mode...")


def create_lag_features(data, n_lags=48):
    """
    Create lag features for supervised learning.
    
    Args:
        data: Time series data
        n_lags: Number of lag features (default 48)
    
    Returns:
        X: Feature matrix (n_samples, n_lags)
        y: Target values (n_samples,)
    """
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def load_data_from_dataset(dataset_name='ca/2019/short', split='test'):
    """
    Load data from gift-eval dataset.
    
    Args:
        dataset_name: Dataset name (e.g., 'ca/2019/short')
        split: 'train' or 'test'
    
    Returns:
        X: Feature matrix
        y: Target values
        full_data: Full time series
    """
    print(f"Loading dataset: {dataset_name}, split: {split}")
    
    try:
        # Setup environment
        os.environ["GIFT_EVAL"] = str(Path(__file__).parent.parent / 
                                      'gift-eval' / 'datasets')
        
        # Load dataset
        dataset = Dataset.load(dataset_name)
        
        # Get data based on split
        if split == 'test':
            data_iter = dataset.test
        else:
            data_iter = dataset.train
        
        # Collect all time series
        all_series = []
        for item in data_iter:
            target = item['target']
            if len(target.shape) > 1:
                # Multivariate: take first dimension
                target = target[:, 0]
            all_series.append(target)
        
        # Use the first time series for analysis
        full_data = all_series[0]
        print(f"Loaded time series with {len(full_data)} time points")
        
        # Create lag features
        X, y = create_lag_features(full_data, n_lags=48)
        print(f"Created {len(X)} samples with 48 lag features")
        
        return X, y, full_data
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to synthetic data...")
        return load_synthetic_data()


def load_synthetic_data(n_samples=1000, n_lags=48):
    """
    Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_lags: Number of lag features
    
    Returns:
        X: Feature matrix
        y: Target values
        full_data: Full time series
    """
    print(f"Generating synthetic data: {n_samples} samples, {n_lags} lags")
    
    # Generate synthetic time series with trend + seasonality + noise
    t = np.arange(n_samples + n_lags)
    trend = 0.1 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    noise = np.random.normal(0, 1, len(t))
    full_data = trend + seasonal + noise
    
    # Create lag features
    X, y = create_lag_features(full_data, n_lags=n_lags)
    
    return X, y, full_data


def train_model(X, y, model_type='ridge'):
    """
    Train a model on the data.
    
    Args:
        X: Feature matrix
        y: Target values
        model_type: Type of model ('ridge', 'xgboost', 'lightgbm', 'random_forest')
    
    Returns:
        model: Trained model
    """
    print(f"\nTraining {model_type} model...")
    
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X, y)
    print(f"Model trained on {len(X)} samples")
    
    return model


def analyze_shap(model, X, y, model_type='ridge', max_samples=100):
    """
    Perform SHAP analysis on the model.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        model_type: Type of model
        max_samples: Maximum number of samples to analyze (for speed)
    
    Returns:
        explainer: SHAP explainer
        shap_values: SHAP values
        X_sample: Sample data used for analysis
        y_sample: Corresponding target values
    """
    print(f"\nPerforming SHAP analysis...")
    
    # Sample data for faster computation
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    print(f"Analyzing {len(X_sample)} samples...")
    
    # Create SHAP explainer based on model type
    if model_type == 'ridge':
        # Use Linear explainer for linear models
        explainer = shap.LinearExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    elif model_type in ['xgboost', 'lightgbm']:
        # Use Tree explainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type == 'random_forest':
        # Use Tree explainer for random forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Use Kernel explainer as fallback (slower)
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 50))
        shap_values = explainer.shap_values(X_sample)
    
    print(f"SHAP analysis complete!")
    
    return explainer, shap_values, X_sample, y_sample


def plot_shap_summary(shap_values, X_sample, explainer, output_dir='./shap_plots', model_name='ridge'):
    """
    Generate SHAP summary plots.
    
    Args:
        shap_values: SHAP values
        X_sample: Sample data
        explainer: SHAP explainer object
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature names: lag-1, lag-2, ..., lag-48
    feature_names = [f'lag-{i+1}' for i in range(X_sample.shape[1])]
    
    print(f"\nGenerating SHAP plots...")
    
    # 1. Summary plot (bar) - mean absolute SHAP value
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                     plot_type='bar', show=False)
    plt.title(f'{model_name.upper()} - Feature Importance (Mean |SHAP|)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_summary_bar.png")
    
    # 2. Summary plot (beeswarm) - detailed view
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title(f'{model_name.upper()} - SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_summary_beeswarm.png")
    
    # 3. Heatmap of SHAP values
    plt.figure(figsize=(14, 8))
    shap.plots.heatmap(shap.Explanation(values=shap_values, 
                                        data=X_sample,
                                        feature_names=feature_names),
                       show=False)
    plt.title(f'{model_name.upper()} - SHAP Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_heatmap.png")
    
    # 4. Individual force plot for first prediction
    plt.figure(figsize=(20, 3))
    shap.force_plot(explainer.expected_value, shap_values[0], X_sample[0],
                   feature_names=feature_names, matplotlib=True, show=False)
    plt.title(f'{model_name.upper()} - SHAP Force Plot (First Sample)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_shap_force_first.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_shap_force_first.png")


def plot_temporal_importance(shap_values, output_dir='./shap_plots', model_name='ridge'):
    """
    Plot temporal importance: which time lags are most important.
    
    Args:
        shap_values: SHAP values
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    # Calculate mean absolute SHAP value for each lag
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create temporal plot
    plt.figure(figsize=(14, 6))
    lags = np.arange(1, len(mean_abs_shap) + 1)
    plt.bar(lags, mean_abs_shap, color='steelblue', alpha=0.7)
    plt.xlabel('Time Lag (steps back)', fontsize=12)
    plt.ylabel('Mean |SHAP value|', fontsize=12)
    plt.title(f'{model_name.upper()} - Temporal Feature Importance\n'
              f'How much each historical time point contributes to prediction', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight top 5 most important lags
    top_5_indices = np.argsort(mean_abs_shap)[-5:]
    plt.bar(top_5_indices + 1, mean_abs_shap[top_5_indices], 
            color='coral', alpha=0.8, label='Top 5 important lags')
    
    # Add value labels for top 5
    for idx in top_5_indices:
        plt.text(idx + 1, mean_abs_shap[idx], f'lag-{idx+1}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_temporal_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_temporal_importance.png")
    
    # Print statistics
    print(f"\nTop 10 most important time lags:")
    top_10_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    for i, idx in enumerate(top_10_indices, 1):
        print(f"  {i}. lag-{idx+1}: {mean_abs_shap[idx]:.4f}")


def plot_lag_groups(shap_values, output_dir='./shap_plots', model_name='ridge'):
    """
    Plot importance grouped by time windows (recent, medium, distant past).
    
    Args:
        shap_values: SHAP values
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Group lags into windows
    n_lags = len(mean_abs_shap)
    recent = mean_abs_shap[:n_lags//3].sum()  # First 1/3 (most recent)
    medium = mean_abs_shap[n_lags//3:2*n_lags//3].sum()  # Middle 1/3
    distant = mean_abs_shap[2*n_lags//3:].sum()  # Last 1/3 (most distant)
    
    # Create grouped bar plot
    plt.figure(figsize=(10, 6))
    groups = ['Recent\n(lag 1-16)', 'Medium\n(lag 17-32)', 'Distant\n(lag 33-48)']
    values = [recent, medium, distant]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    bars = plt.bar(groups, values, color=colors, alpha=0.7)
    plt.ylabel('Total |SHAP value|', fontsize=12)
    plt.title(f'{model_name.upper()} - Importance by Time Window', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_lag_groups.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_lag_groups.png")
    
    # Print percentages
    total = recent + medium + distant
    print(f"\nImportance by time window:")
    print(f"  Recent (lag 1-16):  {recent:.4f} ({100*recent/total:.1f}%)")
    print(f"  Medium (lag 17-32): {medium:.4f} ({100*medium/total:.1f}%)")
    print(f"  Distant (lag 33-48): {distant:.4f} ({100*distant/total:.1f}%)")


def plot_sample_with_shap_and_prediction(model, X_sample, y_sample, shap_values,
                                        num_examples=3,
                                        output_dir='./shap_plots',
                                        model_name='ridge'):
    """
    Plot individual examples showing:
    1. Historical data (context)
    2. SHAP values for each time point
    3. Model predictions vs actual future values
    
    Args:
        model: Trained model
        X_sample: Historical context data (n_samples, n_lags)
        y_sample: Actual future values (n_samples,)
        shap_values: SHAP values (n_samples, n_lags)
        num_examples: Number of examples to plot
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to available samples
    num_examples = min(num_examples, len(X_sample))
    
    for idx in range(num_examples):
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        n_lags = X_sample.shape[1]
        
        # Prepare time axis
        historical_time = np.arange(-n_lags, 0)
        future_time = [0]  # Only 1 step ahead for Ridge in this setup
        
        # Get data for this sample
        historical_data = X_sample[idx]
        actual_future = y_sample[idx]
        shap_vals = shap_values[idx]
        
        # Get model prediction
        predicted_future = model.predict(X_sample[idx].reshape(1, -1))[0]
        
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
        ax3.plot(future_time, [actual_future], 'go', markersize=10, 
                label='Actual Next Value', alpha=0.8)
        # Predicted future
        ax3.plot(future_time, [predicted_future], 'rs', markersize=10, 
                label='Predicted Next Value', alpha=0.8)
        
        # Connect last historical point to predictions
        ax3.plot([historical_time[-1], 0], [historical_data[-1], actual_future], 
                'g--', alpha=0.4, linewidth=1)
        ax3.plot([historical_time[-1], 0], [historical_data[-1], predicted_future], 
                'r--', alpha=0.4, linewidth=1)
        
        # Shade future region
        ax3.axvspan(-0.5, 0.5, alpha=0.1, color='yellow', label='Forecast Point')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        ax3.set_xlabel('Time Steps', fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.set_title('Full Time Series: History + Actual vs Predicted Next Value', 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(alpha=0.3)
        
        # Calculate error metrics
        error = abs(actual_future - predicted_future)
        ax3.text(0.02, 0.98, 
                f'Actual: {actual_future:.2f}\n'
                f'Predicted: {predicted_future:.2f}\n'
                f'Error: {error:.2f}', 
                transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{model_name}_sample_{idx+1}_detailed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {model_name}_sample_{idx+1}_detailed.png")
    
    print(f"\nGenerated {num_examples} detailed sample visualizations!")


def main():
    """Main function to run SHAP analysis."""
    print("=" * 80)
    print("SHAP Interpretability Analysis for Time Series Forecasting")
    print("=" * 80)
    
    # Configuration
    MODEL_TYPE = 'ridge'  # Options: 'ridge', 'xgboost', 'lightgbm', 'random_forest'
    DATASET_NAME = 'ca/2019/short'  # Dataset to analyze
    USE_SYNTHETIC = False  # Set to True to use synthetic data
    USE_SHARED_DATA = True  # Set to True to use pre-generated shared data
    SHARED_DATA_DIR = './shap_data_experiments/sd_2019_15T_ctx48_pred3'  # Path to shared data
    OUTPUT_DIR = './ridge_shap_results'
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Dataset: {DATASET_NAME if not USE_SYNTHETIC else 'Synthetic'}")
    print(f"  Use shared data: {USE_SHARED_DATA}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # Load data
    if USE_SHARED_DATA and os.path.exists(SHARED_DATA_DIR):
        # Load pre-generated shared data
        print(f"\nLoading shared data from: {SHARED_DATA_DIR}")
        try:
            X_all = np.load(f'{SHARED_DATA_DIR}/X_context.npy')
            y_all = np.load(f'{SHARED_DATA_DIR}/y_future.npy')
            # For Ridge, we need to reshape y if it has multiple prediction steps
            if len(y_all.shape) > 1 and y_all.shape[1] > 1:
                # Use only first prediction step for Ridge
                y_all = y_all[:, 0]
            print(f"Loaded shared data: X shape {X_all.shape}, y shape {y_all.shape}")
            X, y, full_data = X_all, y_all, None
        except Exception as e:
            print(f"Failed to load shared data: {e}")
            print("Falling back to individual data loading...")
            USE_SHARED_DATA = False
    
    if not USE_SHARED_DATA:
        if USE_SYNTHETIC:
            X, y, full_data = load_synthetic_data(n_samples=1000, n_lags=48)
        else:
            try:
                X, y, full_data = load_data_from_dataset(DATASET_NAME, split='test')
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                print("Using synthetic data instead...")
                X, y, full_data = load_synthetic_data(n_samples=1000, n_lags=48)
    
    # Split data for training and testing
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train, model_type=MODEL_TYPE)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"\nModel performance on test set:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # SHAP analysis
    explainer, shap_values, X_sample, y_sample = analyze_shap(model, X_test, y_test,
                                                              model_type=MODEL_TYPE,
                                                              max_samples=100)
    
    # Generate plots
    plot_shap_summary(shap_values, X_sample, explainer, output_dir=OUTPUT_DIR, model_name=MODEL_TYPE)
    plot_temporal_importance(shap_values, output_dir=OUTPUT_DIR, model_name=MODEL_TYPE)
    plot_lag_groups(shap_values, output_dir=OUTPUT_DIR, model_name=MODEL_TYPE)
    
    # Plot detailed examples with predictions
    print("\nGenerating detailed sample visualizations...")
    plot_sample_with_shap_and_prediction(model, X_sample, y_sample, 
                                        shap_values, num_examples=3, 
                                        output_dir=OUTPUT_DIR, model_name=MODEL_TYPE)
    
    print(f"\n" + "=" * 80)
    print(f"Analysis complete! All plots saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Save SHAP values for further analysis
    np.save(f'{OUTPUT_DIR}/{MODEL_TYPE}_shap_values.npy', shap_values)
    np.save(f'{OUTPUT_DIR}/{MODEL_TYPE}_X_sample.npy', X_sample)
    print(f"\nSHAP values saved to: {OUTPUT_DIR}/{MODEL_TYPE}_shap_values.npy")


if __name__ == '__main__':
    main()
