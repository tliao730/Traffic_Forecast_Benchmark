# SHAP Interpretability Analysis for Time Series Forecasting

This directory contains tools for performing SHAP (SHapley Additive exPlanations) analysis on time series forecasting models to understand which historical time points contribute most to predictions.

## Overview

Three main scripts provide comprehensive interpretability analysis:

1. **`prepare_shap_data.py`** - Data preparation (run this first)
2. **`shap_inter.py`** - SHAP analysis for Ridge regression (traditional ML)
3. **`moirai2_shap_inter.py`** - SHAP analysis for Moirai2 (deep learning transformer)

## Quick Start

### Step 1: Prepare Data

Generate datasets for all experimental configurations:

```bash
cd interpret
python prepare_shap_data.py
```

**Configuration (edit in script):**
- `DATASET_NAME`: Dataset to analyze (default: `'sd'`)
- `YEAR`: Year to use (default: `'2019'`)
- `FREQ`: Sampling frequency (default: `'15T'` for 15 minutes)
- `CONTEXT_LENGTHS`: Historical window sizes (default: `[48, 98, 196, 336, 720]`)
- `PREDICTION_LENGTHS`: Forecast horizons (default: `[3, 6, 12, 48]`)
- `NUM_SAMPLES`: Number of time series (default: `None` for all available)

**Output:**
```
shap_data_experiments/
├── sd_2019_15T_ctx48_pred3/
│   ├── X_context.npy          # Historical data (n_samples, context_length)
│   ├── y_future.npy           # Future values (n_samples, prediction_length)
│   ├── item_ids.txt           # Time series IDs
│   └── metadata.txt           # Dataset metadata
├── sd_2019_15T_ctx48_pred6/
├── ...
└── data_generation_summary.txt
```

**Total configurations generated:** 20 (5 context lengths × 4 prediction lengths)

### Step 2: Ridge SHAP Analysis

Analyze traditional machine learning model:

```bash
python shap_inter.py
```

**Key Configuration (edit in script):**
- `MODEL_TYPE`: Model to analyze (default: `'ridge'`, options: `'xgboost'`, `'lightgbm'`, `'random_forest'`)
- `USE_SHARED_DATA`: Use prepared data (default: `True`)
- `SHARED_DATA_DIR`: Path to prepared data (default: `'./shap_data_experiments/sd_2019_15T_ctx48_pred3'`)
- `OUTPUT_DIR`: Results directory (default: `'./ridge_shap_results'`)

**Output:**
```
ridge_shap_results/
├── ridge_shap_summary_bar.png        # Feature importance ranking
├── ridge_shap_summary_beeswarm.png   # Detailed SHAP values
├── ridge_shap_heatmap.png            # SHAP values across samples
├── ridge_temporal_importance.png     # Time-based importance
├── ridge_lag_groups.png              # Grouped by time windows
├── ridge_sample_1_detailed.png       # Individual prediction examples
├── ridge_sample_2_detailed.png
├── ridge_sample_3_detailed.png
├── ridge_shap_values.npy             # Raw SHAP values
└── ridge_X_sample.npy                # Analyzed samples
```

### Step 3: Moirai2 SHAP Analysis

Analyze deep learning transformer model:

```bash
python moirai2_shap_inter.py
```

**Key Configuration (edit in script):**
- `DATASET_NAME`: Dataset name (default: `'sd'`)
- `YEAR`: Year (default: `'2019'`)
- `FREQ`: Frequency (default: `'15T'`)
- `CONTEXT_LENGTHS`: Historical lengths to test (default: `[48, 98, 196, 336, 720]`)
- `PREDICTION_LENGTHS`: Forecast horizons to test (default: `[3, 6, 12, 48]`)
- `MODEL_PATH`: Moirai2 model (default: `'Salesforce/moirai-2.0-R-small'`)
- `NUM_SAMPLES`: Samples to load (default: `100`)
- `SHAP_SAMPLES`: Samples for SHAP (default: `10` - small due to computational cost)
- `BASE_OUTPUT_DIR`: Results directory (default: `'./moirai2_shap_experiments'`)

**Output:**
```
moirai2_shap_experiments/
├── sd_2019_15T_ctx48_pred3/
│   ├── sd_2019_15T_ctx48_pred3_temporal_importance_full.png
│   ├── sd_2019_15T_ctx48_pred3_temporal_importance_recent.png
│   ├── sd_2019_15T_ctx48_pred3_time_windows.png
│   ├── sd_2019_15T_ctx48_pred3_shap_heatmap.png
│   ├── sd_2019_15T_ctx48_pred3_sample_1_detailed.png
│   ├── sd_2019_15T_ctx48_pred3_sample_2_detailed.png
│   ├── sd_2019_15T_ctx48_pred3_sample_3_detailed.png
│   ├── shap_values.npy
│   ├── X_sample.npy
│   ├── y_sample.npy
│   ├── item_ids.txt
│   └── metadata.txt
├── sd_2019_15T_ctx48_pred6/
├── ...
└── experiments_summary.txt
```

**Total experiments:** 20 (5 context lengths × 4 prediction lengths)

## Input Data

### Required Format

Data should be in HDF5 format located at: `../data/{dataset_name}/{dataset_name}_his_raw_{year}.h5`

Example structure:
- Path: `../data/sd/sd_his_raw_2019.h5`
- Format: Pandas DataFrame with DatetimeIndex
- Columns: Individual time series (e.g., sensor IDs)
- Values: Traffic flow or other time series measurements

### Data Processing Pipeline

1. **Load** from HDF5 file
2. **Resample** to target frequency (e.g., 15 minutes)
3. **Split** into train/val/test (60%/20%/20%)
4. **Filter** to test set (last 20% of data)
5. **Remove** columns with any NaN values
6. **Extract** sliding windows (context + future)
7. **Verify** no NaN values in samples

## Output Visualizations

### 1. Temporal Importance Plots
Shows which historical time points are most important for predictions.
- **Full view**: All time lags
- **Zoomed view**: Most recent time points
- **Insights**: Identify critical time windows

### 2. Time Window Analysis
Groups importance by temporal distance:
- **Recent** (first 1/3): Most recent history
- **Medium** (middle 1/3): Mid-range history
- **Distant** (last 1/3): Distant past

**Key Finding**: Moirai2 focuses more on distant past (59%), while Ridge is more uniform.

### 3. SHAP Summary Plots
- **Bar plot**: Mean absolute SHAP values (feature ranking)
- **Beeswarm plot**: SHAP value distribution across samples
- **Heatmap**: SHAP values for all samples and features

### 4. Individual Predictions
Detailed 3-panel visualizations:
1. **Top**: Historical data colored by SHAP importance
2. **Middle**: SHAP values bar chart (red = increases, blue = decreases)
3. **Bottom**: Full timeline with actual vs predicted values

## Configuration Details

### Context Lengths
- **48**: 12 hours (at 15min intervals)
- **98**: ~24.5 hours
- **196**: ~49 hours
- **336**: ~84 hours (3.5 days)
- **720**: ~180 hours (7.5 days)

### Prediction Lengths
- **3**: 45 minutes ahead
- **6**: 1.5 hours ahead
- **12**: 3 hours ahead
- **48**: 12 hours ahead

## Requirements

```python
# Core dependencies
numpy
pandas
matplotlib
shap

# For Ridge analysis
scikit-learn
xgboost  # optional
lightgbm  # optional

# For Moirai2 analysis
torch
uni2ts  # Salesforce Moirai2 library
```

Install with:
```bash
pip install numpy pandas matplotlib shap scikit-learn torch
pip install uni2ts  # For Moirai2
```

## Computational Notes

### Ridge SHAP Analysis
- **Speed**: Fast (uses LinearExplainer)
- **Samples**: Analyzes 100 samples by default
- **Time**: ~1-2 minutes per configuration

### Moirai2 SHAP Analysis
- **Speed**: Slow (uses KernelExplainer, model-agnostic)
- **Samples**: Analyzes 10 samples by default (computational constraint)
- **Time**: ~10-20 minutes per configuration
- **GPU**: Recommended (set `DEVICE='cuda'`)

**Note**: Moirai2 uses fewer samples (10 vs 100) due to computational cost of SHAP on deep learning models.

## Customization

### Analyze Different Configurations

Edit `prepare_shap_data.py`:
```python
CONTEXT_LENGTHS = [48, 96]  # Only 2 context lengths
PREDICTION_LENGTHS = [3, 12]  # Only 2 prediction lengths
# Generates 4 configurations instead of 20
```

### Analyze Specific Dataset

Edit the configuration section in any script:
```python
DATASET_NAME = 'ca'  # Change from 'sd' to 'ca'
YEAR = '2020'        # Change year
FREQ = '30T'         # Change to 30-minute intervals
```

### Use Different Model

Edit `shap_inter.py`:
```python
MODEL_TYPE = 'xgboost'  # Or 'lightgbm', 'random_forest'
```

## Troubleshooting

### Out of Memory
- Reduce `NUM_SAMPLES` in configuration
- Reduce `SHAP_SAMPLES` for Moirai2 (e.g., 5 instead of 10)
- Use smaller context lengths

### Missing Data Files
Ensure data files exist at: `../data/{dataset}/sd_his_raw_{year}.h5`

### CUDA Errors
Set device to CPU in Moirai2:
```python
DEVICE = 'cpu'
```

## Interpretation Guide

### SHAP Values
- **Positive**: Feature increases prediction
- **Red color**: High feature value with positive impact
- **Blue color**: Low feature value with negative impact
- **Magnitude**: Importance of the time point

### Comparing Models
1. Run Ridge and Moirai2 on same configuration
2. Compare time window distributions
3. Identify differences in temporal attention patterns
4. Example finding: Moirai2 uses 59% distant past, Ridge more uniform

## Citation

If you use this interpretability analysis, please cite:
- SHAP: Lundberg & Lee (2017)
- Moirai2: Salesforce AI Research
- LargeST: [Your paper/repository]

## Contact

For questions or issues, please open an issue in the repository.
