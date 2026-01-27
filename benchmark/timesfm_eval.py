import csv
import json
import logging
import os
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import common functions (need to add this early for get_prediction_length)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
from common import get_prediction_length

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Import Dataset from the main gift-eval package
# Ensure we import from the main package
gift_eval_src_path = os.path.join(os.path.dirname(script_dir), "src")
if gift_eval_src_path not in sys.path:
    sys.path.insert(0, gift_eval_src_path)
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
from gluonts.itertools import batcher
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Check if timesfm is installed
# Note: This script is named timesfm_eval.py (not timesfm.py) to avoid
# conflicting with the timesfm package name.
try:
    import timesfm
except ImportError as e:
    raise ImportError(
        "timesfm package is not installed. Please install it by running:\n"
        "  pip install 'timesfm[pax]'  # for JAX backend (python 3.10.x)\n"
        "  or\n"
        "  pip install 'timesfm[torch]'  # for PyTorch backend (python 3.11.x)\n"
        "\n"
        "Original error: " + str(e)
    ) from e

# Dataset configuration
# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
short_datasets = "sd/2019/15T"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = ""

# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

# Determine the correct path for dataset_properties.json
dataset_properties_path = os.path.join(script_dir, "dataset_properties.json")
if not os.path.exists(dataset_properties_path):
    # Fallback to trying relative paths
    if os.path.exists("./dataset_properties.json"):
        dataset_properties_path = "./dataset_properties.json"
    elif os.path.exists("./notebooks/dataset_properties.json"):
        dataset_properties_path = "./notebooks/dataset_properties.json"
    else:
        raise FileNotFoundError(
            "dataset_properties.json not found. Make sure you're running this script from notebooks/ directory."
        )

dataset_properties_map = json.load(open(dataset_properties_path))

# Instantiate the metrics
metrics = [
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


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


class TimesFmPredictor:
    def __init__(
        self,
        tfm,
        prediction_length: int,
        ds_freq: str,
        *args,
        **kwargs,
    ):
        self.tfm = tfm
        self.prediction_length = prediction_length
        if self.prediction_length > self.tfm.horizon_len:
            self.tfm.horizon_len = (
                (self.prediction_length + self.tfm.output_patch_len - 1) //
                self.tfm.output_patch_len) * self.tfm.output_patch_len
            print('Jitting for new prediction length.')
        self.freq = timesfm.freq_map(ds_freq)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = []
            for entry in batch:
                arr = np.array(entry["target"])
                context.append(arr)
            freqs = [self.freq] * len(context)
            _, full_preds = self.tfm.forecast(context, freqs, normalize=True)
            full_preds = full_preds[:, 0:self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
        forecast_outputs = np.concatenate(forecast_outputs)

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.tfm.quantiles)),
                    start_date=forecast_start_date,
                )
            )

        return forecasts


pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Model configuration
model_name = "timesfm_2_0_500m"

# Load the TimesFM model
print("Loading TimesFM model...")
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        num_layers=50,
        horizon_len=128,
        context_len=2048,
        use_positional_embedding=False,
        output_patch_len=128,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-jax"),
)

# If you are using the pytorch version, uncomment the following:
# tfm = timesfm.TimesFm(
#     hparams=timesfm.TimesFmHparams(
#         backend="gpu",
#         per_core_batch_size=32,
#         num_layers=50,
#         horizon_len=128,
#         context_len=2048,
#         use_positional_embedding=False,
#         output_patch_len=128,
#     ),
#     checkpoint=timesfm.TimesFmCheckpoint(
#         huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
# )

print("Model loaded successfully.")

# Output directory relative to notebooks/
output_dir = os.path.join("..", "results", model_name)
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(output_dir)}")
csv_file_path = os.path.join(output_dir, "all_results.csv")

# Order all dataset settings from lowest to highest prediction length
# to minimize the number of jittings (not necessary for pytorch version)
all_ds_tuples = []

for ds_num, ds_name in enumerate(all_datasets):
    ds_key = ds_name.split("/")[0]
    print(f"Preparing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
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
        # Initialize the dataset to get prediction_length
        to_univariate = (
            False
            if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
        
        # Override dataset's prediction_length with our custom values
        prediction_length = get_prediction_length(term)
        dataset.prediction_length = prediction_length
        
        all_ds_tuples.append(
            (dataset.prediction_length, ds_config, ds_name, term, to_univariate)
        )

# Sort by prediction_length to minimize jitting
all_ds_tuples = sorted(all_ds_tuples)

# Check if file exists and read completed datasets
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

# Evaluate on all settings
for entry in all_ds_tuples:
    prediction_length = entry[0]
    ds_config = entry[1]
    ds_name = entry[2]
    term = entry[3]
    to_univariate = entry[4]
    ds_key, ds_freq, _ = ds_config.split("/")

    if ds_config in done_datasets:
        print(f"Skipping already completed dataset: {ds_config}")
        continue

    dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
    
    # Override dataset's prediction_length with our custom values
    dataset.prediction_length = prediction_length
    print(f"Using custom prediction length: {prediction_length} (term: {term})")
    
    season_length = get_seasonality(dataset.freq)
    print(f"Processing entry: {entry}")
    print(f"Dataset size: {len(dataset.test_data)}")
    predictor = TimesFmPredictor(
        tfm=tfm,
        prediction_length=dataset.prediction_length,
        ds_freq=ds_freq,
    )
    # Measure the time taken for evaluation
    res = evaluate_model(
        predictor,
        test_data=dataset.test_data,
        metrics=metrics,
        batch_size=1024,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    # Append the results to the CSV file
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

    print(f"Results for {ds_config} have been written to {csv_file_path}")

# Load and display final results
results_file = os.path.join("..", "results", model_name, "all_results.csv")
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    print("\nFinal aggregated results:")
    print(df)
else:
    print(f"\nNo results file found at {results_file}")

