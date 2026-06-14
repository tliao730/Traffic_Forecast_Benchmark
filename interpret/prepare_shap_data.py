"""
Prepare shared data for SHAP analysis across different models.

This script loads data directly from h5 files and saves it, ensuring Ridge and Moirai2
use exactly the same input data for fair comparison.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_base_dataset(dataset_name='ca', year='2019', freq='15T', random_seed=42):
    """
    Load and preprocess the base dataset once.
    
    Args:
        dataset_name: Dataset name (e.g., 'ca', 'sd')
        year: Year to load (e.g., '2019')
        freq: Resampling frequency (e.g., '15T')
        random_seed: Random seed for reproducibility
    
    Returns:
        test_df: Test split dataframe (without NaN columns)
        metadata: Basic metadata about the dataset
    """
    np.random.seed(random_seed)
    
    print(f"Loading base dataset: {dataset_name}/{year}")
    print(f"  Frequency: {freq}")
    print(f"  Random seed: {random_seed}")
    
    # Construct h5 file path
    data_dir = Path(__file__).parent.parent / 'data' / dataset_name
    h5_file = data_dir / f"{dataset_name}_his_raw_{year}.h5"
    
    if not h5_file.exists():
        h5_file = data_dir / f"{dataset_name}_his_{year}.h5"
    
    if not h5_file.exists():
        raise FileNotFoundError(f"Cannot find h5 file: {h5_file}")
    
    print(f"  Loading from: {h5_file}")
    
    # Load h5 file
    df = pd.read_hdf(h5_file)
    print(f"  Original shape: {df.shape}")
    
    # Resample to desired frequency
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
    
    if test_df.empty:
        raise ValueError('Test split is empty')
    
    # Remove columns with NaN values
    test_df = test_df.dropna(axis=1, how='any')
    print(f"  After removing NaN columns: {test_df.shape}")
    
    if test_df.shape[1] == 0:
        raise ValueError('No columns left after removing NaN values')
    
    metadata = {
        'dataset_name': dataset_name,
        'year': year,
        'freq': freq,
        'data_source': 'real_h5',
        'split': 'test',
        'num_time_series': test_df.shape[1],
        'num_time_points': test_df.shape[0],
        'random_seed': random_seed,
    }
    
    print(f"\nBase dataset loaded successfully!")
    print(f"  Available time series: {test_df.shape[1]}")
    print(f"  Time points in test set: {test_df.shape[0]}")
    
    return test_df, metadata


def extract_samples_from_dataframe(test_df, context_length, prediction_length, 
                                   num_samples=None, random_seed=42):
    """
    Extract samples from the loaded dataframe for a specific configuration.
    
    Args:
        test_df: Test dataframe
        context_length: Number of historical time points
        prediction_length: Number of future time points
        num_samples: Number of time series to sample (None = all)
        random_seed: Random seed for reproducibility
    
    Returns:
        X: Historical data (num_samples, context_length)
        y: Future data (num_samples, prediction_length)
        item_ids: List of item IDs
    """
    np.random.seed(random_seed)
    
    # Collect samples
    X_list = []
    y_list = []
    item_ids = []
    
    # Randomly sample from available time series
    available_columns = list(test_df.columns)
    np.random.shuffle(available_columns)
    
    # If num_samples is None, use all available columns
    if num_samples is None:
        num_samples = len(available_columns)
    
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
        raise ValueError(
            f"No valid time series found. Each series must have at least "
            f"{context_length + prediction_length} points without NaN."
        )
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    return X, y, item_ids


def save_data(X, y, item_ids, metadata, output_dir='./shared_shap_data'):
    """Save data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/X_context.npy', X)
    np.save(f'{output_dir}/y_future.npy', y)
    
    # Save item IDs
    with open(f'{output_dir}/item_ids.txt', 'w') as f:
        f.write(f"Total samples: {len(item_ids)}\n\n")
        for i, item_id in enumerate(item_ids, 1):
            f.write(f"{i}. {item_id}\n")
    
    # Save metadata as text
    with open(f'{output_dir}/metadata.txt', 'w') as f:
        f.write("Shared SHAP Analysis Data\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nData saved to: {output_dir}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Item IDs: {len(item_ids)}")


def load_data(data_dir='./shared_shap_data'):
    """Load saved data from disk."""
    X = np.load(f'{data_dir}/X_context.npy')
    y = np.load(f'{data_dir}/y_future.npy')
    
    # Load metadata
    metadata = {}
    with open(f'{data_dir}/metadata.txt', 'r') as f:
        for line in f:
            if ':' in line and not line.startswith('='):
                key, value = line.strip().split(':', 1)
                metadata[key.strip()] = value.strip()
    
    print(f"Loaded shared data from: {data_dir}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    return X, y, metadata


def main():
    """Prepare and save shared data for multiple configurations."""
    print("=" * 80)
    print("Preparing Shared Data for SHAP Analysis - Multi-Configuration")
    print("=" * 80)
    
    # ===== CONFIGURATION =====
    DATASET_NAME = 'sd'  # Dataset name
    YEAR = '2019'  # Year
    FREQ = '15T'  # Resampling frequency (15 minutes)
    RANDOM_SEED = 42
    
    # Experiment configurations (matching moirai2_shap_inter.py)
    CONTEXT_LENGTHS = [48, 98, 196, 336, 720]  # Historical lengths to test
    PREDICTION_LENGTHS = [3, 6, 12, 48]  # Prediction horizons to test
    
    # Number of samples (None = all available)
    NUM_SAMPLES = None  # Save all available samples
    
    BASE_OUTPUT_DIR = './shap_data_experiments'
    # ===== END CONFIGURATION =====
    
    print(f"\nGlobal Configuration:")
    print(f"  Dataset: {DATASET_NAME}/{YEAR}")
    print(f"  Frequency: {FREQ}")
    print(f"  Context lengths: {CONTEXT_LENGTHS}")
    print(f"  Prediction lengths: {PREDICTION_LENGTHS}")
    print(f"  Samples per config: {'ALL' if NUM_SAMPLES is None else NUM_SAMPLES}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Base output directory: {BASE_OUTPUT_DIR}")
    
    # Load base dataset ONCE
    print(f"\n{'='*80}")
    print("Step 1: Loading base dataset (once)")
    print(f"{'='*80}\n")
    
    try:
        test_df, base_metadata = load_base_dataset(
            dataset_name=DATASET_NAME,
            year=YEAR,
            freq=FREQ,
            random_seed=RANDOM_SEED
        )
    except Exception as e:
        print(f"ERROR: Failed to load base dataset: {e}")
        return
    
    # Generate data for all configurations
    total_configs = len(CONTEXT_LENGTHS) * len(PREDICTION_LENGTHS)
    print(f"\n{'='*80}")
    print(f"Step 2: Generating {total_configs} configurations from loaded data")
    print(f"{'='*80}\n")
    
    config_count = 0
    results_summary = []
    
    for context_length in CONTEXT_LENGTHS:
        for prediction_length in PREDICTION_LENGTHS:
            config_count += 1
            config_name = f"{DATASET_NAME}_{YEAR}_{FREQ}_ctx{context_length}_pred{prediction_length}"
            output_dir = os.path.join(BASE_OUTPUT_DIR, config_name)
            
            print(f"[Config {config_count}/{total_configs}] {config_name}")
            
            try:
                # Extract samples from loaded dataframe
                X, y, item_ids = extract_samples_from_dataframe(
                    test_df=test_df,
                    context_length=context_length,
                    prediction_length=prediction_length,
                    num_samples=NUM_SAMPLES,
                    random_seed=RANDOM_SEED
                )
                
                print(f"  Extracted {len(X)} samples")
                print(f"  X shape: {X.shape}, y shape: {y.shape}")
                print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
                
                # Create metadata
                metadata = {
                    **base_metadata,  # Include base metadata
                    'context_length': context_length,
                    'prediction_length': prediction_length,
                    'num_samples': len(X),
                    'X_shape': X.shape,
                    'y_shape': y.shape,
                    'X_mean': float(X.mean()),
                    'X_std': float(X.std()),
                    'y_mean': float(y.mean()),
                    'y_std': float(y.std()),
                    'X_min': float(X.min()),
                    'X_max': float(X.max()),
                    'y_min': float(y.min()),
                    'y_max': float(y.max()),
                }
                
                # Save data
                save_data(X, y, item_ids, metadata, output_dir)
                
                results_summary.append({
                    'config': config_name,
                    'context_length': context_length,
                    'prediction_length': prediction_length,
                    'num_samples': len(X),
                    'status': 'SUCCESS'
                })
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results_summary.append({
                    'config': config_name,
                    'context_length': context_length,
                    'prediction_length': prediction_length,
                    'num_samples': 0,
                    'status': f'FAILED: {str(e)}'
                })
            
            print()  # Empty line for readability
    
    # Save summary
    summary_file = os.path.join(BASE_OUTPUT_DIR, 'data_generation_summary.txt')
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("SHAP Data Generation Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {DATASET_NAME}/{YEAR}\n")
        f.write(f"Frequency: {FREQ}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        f.write(f"Total configurations: {total_configs}\n")
        f.write(f"Successful: {sum(1 for r in results_summary if r['status'] == 'SUCCESS')}\n")
        f.write(f"Failed: {sum(1 for r in results_summary if r['status'] != 'SUCCESS')}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results_summary, 1):
            f.write(f"{i}. {result['config']}\n")
            f.write(f"   Context: {result['context_length']}, Prediction: {result['prediction_length']}\n")
            f.write(f"   Samples: {result['num_samples']}\n")
            f.write(f"   Status: {result['status']}\n\n")
    
    print("=" * 80)
    print("Data generation complete!")
    print(f"Successful: {sum(1 for r in results_summary if r['status'] == 'SUCCESS')}/{total_configs}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print(f"Summary: {summary_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
