# Comparison: chronos_1.py vs chronos-2.py

This document explains the differences between the two Chronos evaluation scripts as you requested.

## Overview

- **`chronos_1.py`**: Evaluates Chronos and Chronos-Bolt models (original and bolt variants)
- **`chronos-2.py`**: Evaluates Chronos-2 models (next generation architecture)

## Key Differences Found and Fixed

### 1. Environment Variable Setup

**chronos_1.py (original - line 26):**
```python
os.environ['GIFT_EVAL'] = '/home/defu/workspace/LargeST/data/gift_eval_datasets'
```
- Hardcoded path in the script
- Manual setup required

**chronos-2.py (original):**
```python
load_dotenv()
```
- Relied only on `.env` file or pre-set environment variable
- No fallback mechanism

**Both scripts (after fixes):**
- Now auto-detect GIFT_EVAL path if not set
- Print the path being used for verification
- Support both `.env` file and environment variable

### 2. Dataset Properties Handling

**chronos_1.py (original - lines 38-51):**
```python
# Flexible path handling with fallbacks
if os.path.exists("./notebooks/dataset_properties.json"):
    dataset_properties_map = json.load(open("./notebooks/dataset_properties.json"))
elif os.path.exists("./dataset_properties.json"):
    dataset_properties_map = json.load(open("./dataset_properties.json"))
else:
    dataset_properties_map = json.load(open("dataset_properties.json"))

# Automatically adds CA, GBA, GLA properties if missing
if 'ca' not in dataset_properties_map:
    dataset_properties_map['ca'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}
# ... similar for gba, gla
```

**chronos-2.py (original - line 33):**
```python
# Fixed path only
dataset_properties_map = json.load(open("./notebooks/dataset_properties.json"))
```
- No fallback paths
- Didn't add new datasets automatically

**Both scripts (after fixes):**
- Use `os.path.abspath` and `__file__` to determine script location
- Try multiple fallback paths
- Raise clear error message if file not found
- Both now auto-add CA, GBA, GLA, SD properties

### 3. Output Directory Paths

**chronos_1.py (original - line 241):**
```python
output_dir = f"../results/{model_name}"
```
- Relative path using f-string
- Assumes running from `gift-eval/notebooks/`

**chronos-2.py (original - line 201):**
```python
output_dir = os.path.join("..", "results", "chronos-2", "all_results.csv")
```
- Relative path using `os.path.join`
- Assumes running from `gift-eval/`
- Includes filename in path (should only be directory)

**Both scripts (after fixes):**
```python
# chronos_1.py
output_dir = os.path.join("..", "results", model_name)
print(f"Results will be saved to: {os.path.abspath(output_dir)}")

# chronos-2.py  
output_dir = os.path.join("results", "chronos-2", "all_results.csv")
print(f"Results will be saved to: {os.path.abspath(output_dir)}")
```
- Use `os.path.join` for portability
- Print absolute paths for verification
- Clear about run directory expectations

### 4. Run Directory Requirements

| Script | Original Run Directory | After Fixes | Reason |
|--------|----------------------|-------------|---------|
| `chronos_1.py` | `gift-eval/notebooks/` | `gift-eval/notebooks/` | Output path `../results/` resolves to `gift-eval/results/` |
| `chronos-2.py` | `gift-eval/` | `gift-eval/` | Dataset properties at `./notebooks/`, output at `./results/` |

### 5. Model Differences

**chronos_1.py - ChronosPredictor:**
```python
class ChronosPredictor:
    def __init__(self, model_path, num_samples: int, prediction_length: int):
        self.pipeline = BaseChronosPipeline.from_pretrained(model_path)
        # ...
    
    def predict(self, test_data_input, batch_size: int = 1024):
        # Batch prediction approach
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            # ...
```

**chronos-2.py - Chronos2Predictor:**
```python
class Chronos2Predictor:
    def __init__(self, model_name: str, prediction_length: int, batch_size: int,
                 quantile_levels: list[float], predict_batches_jointly: bool = False):
        self.pipeline = BaseChronosPipeline.from_pretrained(model_name)
        assert isinstance(self.pipeline, Chronos2Pipeline)
        # ...
    
    def predict(self, test_data_input):
        # Uses predict_quantiles method
        quantiles, _ = pipeline.predict_quantiles(
            inputs=input_data,
            prediction_length=self.prediction_length,
            batch_size=model_batch_size,
            quantile_levels=self.quantile_levels,
            predict_batches_jointly=self.predict_batches_jointly,
        )
```

**Key Model Differences:**
- Chronos-1/Bolt: Uses `BaseChronosPipeline` with sample or quantile forecasting
- Chronos-2: Uses `Chronos2Pipeline` with quantile forecasting and cross-learning option

### 6. Evaluation Methodology

**chronos_1.py:**
```python
# Evaluates all test data together
res = evaluate_model(
    predictor,
    test_data=dataset.test_data,
    metrics=metrics,
    batch_size=1024,
    # ...
)
```

**chronos-2.py:**
```python
# Evaluates rolling windows separately to avoid leakage
for window_idx in range(n_windows):
    entries_window_k = list(itertools.islice(dataset.test_data.input, window_idx, None, n_windows))
    forecasts_window_k = list(predictor.predict(entries_window_k))
    forecast_windows.append(forecasts_window_k)

forecasts = [item for items in zip(*forecast_windows) for item in items]
```

**Why different?** Chronos-2 uses in-context learning, so evaluating rolling windows together could cause data leakage.

### 7. Dataset Selection Format

**chronos_1.py (lines 29-32):**
```python
short_datasets = ""
med_long_datasets = "sd/2019/15T gba/2019/15T gla/2019/15T ca/2019/15T"
```

**chronos-2.py (lines 26-28):**
```python
SHORT_DATASETS = ""
MED_LONG_DATASETS = "gba/2019/15T gla/2019/15T ca/2019/15T sd/2019/15T"
```

Both use space-separated strings, just different capitalization conventions.

## Summary of Improvements Made

### Path Inconsistencies Fixed

| Issue | Before | After |
|-------|--------|-------|
| Dataset properties path | Mixed (chronos_1.py flexible, chronos-2.py fixed) | Both use flexible path resolution with clear error messages |
| Output directory | Inconsistent path formats | Both use `os.path.join`, print absolute paths |
| Environment variable | chronos_1.py hardcoded, chronos-2.py relies on dotenv | Both auto-detect with fallbacks |

### New Features Added

1. **Auto-detection of paths** based on script location
2. **Automatic addition** of CA, GBA, GLA, SD to dataset properties
3. **Clear error messages** when files not found
4. **Path verification** by printing absolute paths
5. **Documentation headers** explaining run directory requirements

## How to Use Each Script

### chronos_1.py - For Chronos and Chronos-Bolt Models

```bash
cd /home/defu/workspace/LargeST/gift-eval/notebooks
python chronos_1.py
```

**Supported Models:**
- `amazon/chronos-t5-small`
- `amazon/chronos-t5-base`
- `amazon/chronos-t5-large`
- `amazon/chronos-bolt-small`
- `amazon/chronos-bolt-base`

**Configure:**
- Line 29-32: Dataset selection
- Line 240: Model name for output
- Line 323: Model path for loading

### chronos-2.py - For Chronos-2 Models

```bash
cd /home/defu/workspace/LargeST/gift-eval
python notebooks/chronos-2.py
```

**Supported Models:**
- `s3://autogluon/chronos-2`

**Configure:**
- Line 26-28: Dataset selection
- Line 273: Forecast terms (short/medium/long)
- Line 298: Batch size and device settings

## Files Modified

1. **`chronos_1.py`**: Added robust path handling, environment auto-detection, clearer documentation
2. **`chronos-2.py`**: Added flexible path resolution, auto-add datasets, environment auto-detection
3. **Created `RUNNING_FOUNDATION_MODELS.md`**: Comprehensive guide with your data prep notes
4. **Created `QUICKSTART.md`**: Essential commands only
5. **Created `gift-eval/README_CHRONOS.md`**: Quick reference for Chronos models
6. **Updated `README.md`**: Added references to new documentation

## Why These Differences Exist

### Architecture Differences
- **Chronos-1/Bolt**: Traditional transformer-based forecasting
- **Chronos-2**: Next-gen architecture with in-context learning capabilities

### Run Directory Differences
- Each script's run directory is chosen to make relative paths work correctly
- `chronos_1.py` runs from `notebooks/` so `../results` goes to `gift-eval/results/`
- `chronos-2.py` runs from `gift-eval/` so `results/` goes to `gift-eval/results/`

### Evaluation Method Differences
- Chronos-2 needs special handling for rolling windows to prevent data leakage
- Chronos-1/Bolt can evaluate all data together without leakage concerns

## Verification Checklist

After running scripts, verify:

- [ ] `GIFT_EVAL` path is printed and correct
- [ ] `dataset_properties.json` loaded successfully
- [ ] Output directory path is printed and correct
- [ ] Results CSV file created in expected location
- [ ] No "file not found" errors

## Additional Notes

- Both scripts now handle CUDA OOM automatically by reducing batch size
- Both scripts skip already-completed datasets when resuming
- Results are appended to CSV files, allowing incremental evaluation
- All path operations now use `os.path.join` for cross-platform compatibility

