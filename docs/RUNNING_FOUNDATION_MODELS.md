# Running Foundation Models on LargeST

This guide provides step-by-step instructions for preparing the LargeST dataset and running foundation models (Chronos-1, Chronos-Bolt, and Chronos-2) for evaluation.

## What's New / Fixed

This document addresses the following issues:

✅ **Path Inconsistencies Fixed:**
- Both `chronos_1.py` and `chronos-2.py` now have robust path handling
- Scripts automatically detect and use correct paths for `dataset_properties.json`
- Clear instructions on which directory to run each script from
- Absolute paths printed for output directories

✅ **Environment Setup Clarified:**
- `GIFT_EVAL` environment variable setup explained
- Both scripts now auto-detect the GIFT_EVAL path if not set
- Added support for `.env` file in `gift-eval/` directory

✅ **Dataset Properties Updated:**
- Scripts automatically add CA, GBA, GLA, and SD dataset properties
- No need to manually edit `dataset_properties.json`

✅ **Documentation Added:**
- Step-by-step data preparation instructions
- Clear command execution locations (`cd` commands)
- File structure visualization
- Troubleshooting guide

## Prerequisites

- Python 3.8+
- Kaggle API credentials configured
- CUDA-capable GPU (recommended)
- Required Python packages (see installation steps below)

## Part 1: Data Preparation

### Step 1: Download Raw Dataset

```bash
# From workspace root: /home/defu/workspace/LargeST
mkdir -p data/raw
cd data/raw
kaggle datasets download liuxu77/largest
unzip largest.zip
rm largest.zip
cd ../..  # Return to workspace root
```

**Note:** The raw data files will be downloaded into `data/raw/`.

### Step 2: Process Dataset (Downsample to 15min Interval)

The processing notebook needs to be run to clean and downsample the raw CA data.

**Note:** This step uses a Jupyter notebook (`process_ca_his.ipynb`), not a Python script.

```bash
# From workspace root: /home/defu/workspace/LargeST
cd data_analysis/ca

# Open and run the notebook:
jupyter notebook process_ca_his.ipynb
# OR if using jupyter lab:
# jupyter lab process_ca_his.ipynb
```

**Action required:** Run all cells in the notebook to process the CA historical data. The notebook will:
- Load raw data from `data/ca/ca_his_raw_YYYY.h5` files
- Clean and filter the data
- Downsample to 15-minute intervals
- Save processed data back to the `data/ca/` directory

After processing, return to workspace root:
```bash
cd ../..  # Return to workspace root
```

### Step 3: Create Dataset for Baseline Model Training

```bash
# From workspace root: /home/defu/workspace/LargeST
cd data_analysis
python generate_data_for_training.py --dataset ca --years 2019
cd ..  # Return to workspace root
```

**Output:** Processed data will be saved to `data/ca/2019/`.

**For other datasets:**
```bash
# From data_analysis directory
python generate_data_for_training.py --dataset gba --years 2019
python generate_data_for_training.py --dataset gla --years 2019
python generate_data_for_training.py --dataset sd --years 2019
```

### Step 4: Create Dataset for GIFT Evaluation

```bash
# From workspace root: /home/defu/workspace/LargeST
cd data_analysis
python generate_data_for_gift_eval.py --dataset ca --years 2019 --freq 15T --tod 1 --dow 1 --overwrite
cd ..  # Return to workspace root
```

**Output:** GIFT evaluation data will be saved to `data/gift_eval_datasets/ca/2019/15T/`.

**For all datasets:**
```bash
# From data_analysis directory
python generate_data_for_gift_eval.py --dataset ca --years 2019 --freq 15T --tod 1 --dow 1 --overwrite
python generate_data_for_gift_eval.py --dataset gba --years 2019 --freq 15T --tod 1 --dow 1 --overwrite
python generate_data_for_gift_eval.py --dataset gla --years 2019 --freq 15T --tod 1 --dow 1 --overwrite
python generate_data_for_gift_eval.py --dataset sd --years 2019 --freq 15T --tod 1 --dow 1 --overwrite
cd ..  # Return to workspace root
```

## Part 2: Environment Setup for Foundation Models

### Step 1: Install Required Packages

```bash
# From workspace root: /home/defu/workspace/LargeST
cd gift-eval

# Install chronos-forecasting for Chronos-1 and Chronos-Bolt
pip install chronos-forecasting

# For Chronos-2, you need version 2.0+
pip install "chronos-forecasting>=2.0"

# Install other dependencies
pip install gluonts pandas numpy torch python-dotenv datasets
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the `gift-eval` directory to set the GIFT_EVAL path:

```bash
# From workspace root: /home/defu/workspace/LargeST
cd gift-eval
echo "GIFT_EVAL=/home/defu/workspace/LargeST/data/gift_eval_datasets" > .env
```

**Alternative:** You can also set it directly in your shell:
```bash
export GIFT_EVAL=/home/defu/workspace/LargeST/data/gift_eval_datasets
```

## Part 3: Running Foundation Models

### Running Chronos-1 and Chronos-Bolt Models

The `chronos_1.py` script supports both the original Chronos models and the newer Chronos-Bolt models.

**Important:** Run from the `gift-eval/notebooks` directory.

```bash
# From workspace root: /home/defu/workspace/LargeST
cd gift-eval/notebooks

# Run the script
python chronos_1.py
```

#### What This Script Does:

1. **Sets up environment**: Loads `GIFT_EVAL` environment variable (either from `.env` or from the script itself)
2. **Configures datasets**: You can modify lines 29 and 32 to select which datasets to evaluate:
   ```python
   short_datasets = ""  # For short-term forecasts
   med_long_datasets = "sd/2019/15T gba/2019/15T gla/2019/15T ca/2019/15T"
   ```
3. **Updates dataset properties**: Automatically adds CA, GBA, GLA, and SD to the dataset properties if not present
4. **Runs evaluation**: Evaluates the model on specified datasets for short, medium, and long-term forecasts
5. **Saves results**: Results are saved to `gift-eval/results/{model_name}/all_results.csv`

#### Configuration Options:

**Model Selection** (line 240):
```python
model_name = "chronos_bolt_base"  # Can be changed to:
# - "chronos_base" for Chronos-base
# - "chronos_small" for Chronos-small  
# - "chronos_large" for Chronos-large
# - "chronos_bolt_small" for Chronos-Bolt-small
# - "chronos_bolt_base" for Chronos-Bolt-base
```

**Model Path** (line 323):
```python
model_path="amazon/chronos-bolt-base"  # Can be changed to:
# - "amazon/chronos-t5-small" for original Chronos-small
# - "amazon/chronos-t5-base" for original Chronos-base
# - "amazon/chronos-t5-large" for original Chronos-large
# - "amazon/chronos-bolt-small" for Chronos-Bolt-small
# - "amazon/chronos-bolt-base" for Chronos-Bolt-base
```

**Output Directory** (line 241):
```python
output_dir = f"../results/{model_name}"
```
Results will be saved to `gift-eval/results/{model_name}/all_results.csv`

### Running Chronos-2 Model

The `chronos-2.py` script is specifically for the Chronos-2 model, which has a different architecture.

**Important:** Run from the `gift-eval` directory (one level up from notebooks).

```bash
# From workspace root: /home/defu/workspace/LargeST
cd gift-eval

# Run the script
python notebooks/chronos-2.py
```

**Why run from gift-eval directory?** The script expects to find `dataset_properties.json` at `./notebooks/dataset_properties.json` (line 33).

#### What This Script Does:

1. **Loads environment**: Uses `dotenv` to load environment variables (including `GIFT_EVAL`)
2. **Configures datasets**: Modify lines 26 and 28 to select datasets:
   ```python
   SHORT_DATASETS = ""
   MED_LONG_DATASETS = "gba/2019/15T gla/2019/15T ca/2019/15T sd/2019/15T"
   ```
3. **Evaluates by windows**: Uses rolling window evaluation to avoid cross-batch leakage
4. **Saves results**: Results are saved to `gift-eval/results/chronos-2/all_results.csv`

#### Configuration Options:

**Model Name** (line 199):
```python
model_name = "s3://autogluon/chronos-2"  # Chronos-2 model from S3
```

**Batch Size and Prediction Settings** (line 298):
```python
batch_size=100,
use_multivariate_data=True,
predict_batches_jointly=True,
device_map="cuda",
torch_dtype="float32",
```

**Output Directory** (line 201):
```python
output_dir = os.path.join("..", "results", "chronos-2", "all_results.csv")
```
Results will be saved to `gift-eval/results/chronos-2/all_results.csv`

## Part 4: Key Differences Between Scripts

📖 **For detailed comparison:** See [`CHRONOS_COMPARISON.md`](CHRONOS_COMPARISON.md)

| Feature | chronos_1.py | chronos-2.py |
|---------|--------------|--------------|
| **Run Directory** | `gift-eval/notebooks/` | `gift-eval/` |
| **Models Supported** | Chronos, Chronos-Bolt | Chronos-2 only |
| **Environment Setup** | Auto-detects (after fixes) | Auto-detects (after fixes) |
| **Dataset Properties** | Flexible path handling, auto-adds new datasets | Flexible path handling, auto-adds new datasets |
| **Output Path** | `../results/{model_name}/` | `results/chronos-2/` |
| **Evaluation Method** | Batch prediction | Rolling window to avoid leakage |
| **Predictor Class** | `ChronosPredictor` | `Chronos2Predictor` |

## Part 5: File Paths Summary

### Important Paths to Understand:

```
/home/defu/workspace/LargeST/                    # Workspace root
├── data/
│   ├── raw/                                      # Raw downloaded data
│   ├── ca/                                       # Processed CA data
│   │   └── 2019/                                 # Training data for 2019
│   ├── gift_eval_datasets/                       # GIFT evaluation datasets
│   │   ├── ca/2019/15T/
│   │   ├── gba/2019/15T/
│   │   ├── gla/2019/15T/
│   │   └── sd/2019/15T/
├── data_analysis/                                # Data processing scripts
│   ├── generate_data_for_training.py
│   ├── generate_data_for_gift_eval.py
│   └── ca/
│       └── process_ca_his.ipynb
├── gift-eval/                                    # GIFT evaluation framework
│   ├── .env                                      # Environment variables (create this)
│   ├── notebooks/
│   │   ├── chronos_1.py                         # Run from notebooks/ directory
│   │   ├── chronos-2.py                         # Run from gift-eval/ directory
│   │   └── dataset_properties.json              # Dataset metadata
│   └── results/                                  # Evaluation results
│       ├── chronos_bolt_base/
│       │   └── all_results.csv
│       └── chronos-2/
│           └── all_results.csv
```

## Part 6: Troubleshooting

### Issue 1: "Could not find dataset_properties.json"

**For chronos_1.py:** The script has flexible path handling and will try multiple locations. Make sure you're running from `gift-eval/notebooks/`.

**For chronos-2.py:** The script expects the file at `./notebooks/dataset_properties.json`. Make sure you're running from `gift-eval/` directory.

### Issue 2: "GIFT_EVAL environment variable not set"

Make sure you either:
1. Created a `.env` file in `gift-eval/` with the path, OR
2. Set the environment variable in your shell, OR
3. The path is hardcoded in the script (line 26 in chronos_1.py)

### Issue 3: "Dataset not found"

Verify that:
1. You ran `generate_data_for_gift_eval.py` for all datasets
2. The output directory is `data/gift_eval_datasets/`
3. The `GIFT_EVAL` environment variable points to the correct location

### Issue 4: CUDA Out of Memory

Both scripts have automatic batch size reduction. If you still encounter issues:
- Reduce the initial `batch_size` parameter in the script
- Use a smaller model variant (e.g., chronos-bolt-small instead of chronos-bolt-base)
- Use CPU instead of GPU by setting `device_map="cpu"`

## Part 7: Viewing Results

After running the evaluation scripts, results are saved as CSV files:

```bash
# View Chronos-1/Bolt results
cd /home/defu/workspace/LargeST/gift-eval/results/chronos_bolt_base
cat all_results.csv

# View Chronos-2 results
cd /home/defu/workspace/LargeST/gift-eval/results/chronos-2
cat all_results.csv
```

You can also load and analyze results using pandas:

```python
import pandas as pd

# Load results
df = pd.read_csv("gift-eval/results/chronos_bolt_base/all_results.csv")
print(df)

# Calculate average metrics
print(df.describe())
```

## Part 8: Quick Reference Commands

### Complete Pipeline for One Dataset (CA):

```bash
# Start from workspace root
cd /home/defu/workspace/LargeST

# 1. Download data
mkdir -p data/raw && cd data/raw
kaggle datasets download liuxu77/largest
unzip largest.zip && rm largest.zip
cd ../..

# 2. Process data (run notebook manually)
cd data_analysis/ca
# jupyter notebook process_ca_his.ipynb
cd ../..

# 3. Generate training data
cd data_analysis
python generate_data_for_training.py --dataset ca --years 2019
cd ..

# 4. Generate GIFT eval data
cd data_analysis
python generate_data_for_gift_eval.py --dataset ca --years 2019 --freq 15T --tod 1 --dow 1 --overwrite
cd ..

# 5. Set up environment
cd gift-eval
echo "GIFT_EVAL=/home/defu/workspace/LargeST/data/gift_eval_datasets" > .env

# 6. Run Chronos-1/Bolt
cd notebooks
python chronos_1.py
cd ..

# 7. Run Chronos-2  
python notebooks/chronos-2.py
```

### Run All Datasets:

```bash
# From workspace root
cd /home/defu/workspace/LargeST/data_analysis

# Generate training data for all datasets
for dataset in ca gba gla sd; do
    python generate_data_for_training.py --dataset $dataset --years 2019
done

# Generate GIFT eval data for all datasets
for dataset in ca gba gla sd; do
    python generate_data_for_gift_eval.py --dataset $dataset --years 2019 --freq 15T --tod 1 --dow 1 --overwrite
done

cd ..
```

## Notes

- Always pay attention to which directory you're running commands from
- The inconsistent paths (`./results` vs `../results`) are intentional based on the run directory
- Make sure all datasets are generated before running evaluation scripts
- Evaluation can take several hours depending on the number of datasets and model size

