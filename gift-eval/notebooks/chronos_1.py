# %% [markdown]
# # Quick Start: Running Chronos and Chronos-Bolt models on gift-eval benchmark
# 
# This notebook shows how to run Chronos and Chronos-Bolt models on the gift-eval benchmark.
# 
# **IMPORTANT: Run this script from the gift-eval/notebooks/ directory:**
# ```bash
# cd gift-eval/notebooks
# python chronos_1.py
# ```
# 
# Make sure you download the gift-eval benchmark and set the `GIFT-EVAL` environment variable correctly before running this notebook.
# 
# We will use the `Dataset` class to load the data and run the model. If you have not already please check out the [dataset.ipynb](./dataset.ipynb) notebook to learn more about the `Dataset` class. We are going to just run the model on two datasets for brevity. But feel free to run on any dataset by changing the `short_datasets` and `med_long_datasets` variables below.

# %% [markdown]
# Install Chronos package:
# ``
# pip install chronos-forecasting
# ``

# %%
import json
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set GIFT_EVAL environment variable
os.environ['GIFT_EVAL'] = '/home/defu/workspace/LargeST/data/gift_eval_datasets'

# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
short_datasets = "" #"sd/2019/15T gba/2019/15T gla/2019/15T ca/2019/15T"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "sd/2019/15T gba/2019/15T gla/2019/15T ca/2019/15T"

# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

# Determine the correct path for dataset_properties.json
# This script should be run from gift-eval/notebooks/ directory
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_properties_path = os.path.join(script_dir, "dataset_properties.json")

if not os.path.exists(dataset_properties_path):
    # Fallback to trying relative paths
    if os.path.exists("./dataset_properties.json"):
        dataset_properties_path = "./dataset_properties.json"
    elif os.path.exists("./notebooks/dataset_properties.json"):
        dataset_properties_path = "./notebooks/dataset_properties.json"
    else:
        raise FileNotFoundError(
            "dataset_properties.json not found. Make sure you're running this script from gift-eval/notebooks/ directory."
        )

dataset_properties_map = json.load(open(dataset_properties_path))

# Add properties for new datasets (ca, gba, gla)
if 'ca' not in dataset_properties_map:
    dataset_properties_map['ca'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}
if 'gba' not in dataset_properties_map:
    dataset_properties_map['gba'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}
if 'gla' not in dataset_properties_map:
    dataset_properties_map['gla'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}

# %%
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

# %% [markdown]
# ## Chronos Predictor
# 
# For foundation models, we need to implement a wrapper containing the model and use the wrapper to generate predicitons.
# 
# This is just meant to be a simple wrapper to get you started, feel free to use your own custom implementation to wrap any model.

# %%
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from chronos import BaseChronosPipeline, ForecastType
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm


@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


class ChronosPredictor:
    def __init__(
        self,
        model_path,
        num_samples: int,
        prediction_length: int,
        *args,
        **kwargs,
    ):
        print("prediction_length:", prediction_length)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            *args,
            **kwargs,
        )
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        pipeline = self.pipeline
        predict_kwargs = (
            {"num_samples": self.num_samples}
            if pipeline.forecast_type == ForecastType.SAMPLES
            else {}
        )
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    forecast_outputs.append(
                        pipeline.predict(
                            context,
                            prediction_length=self.prediction_length,
                            **predict_kwargs,
                        ).numpy()
                    )
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        # Convert forecast samples into gluonts Forecast objects
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])

            if pipeline.forecast_type == ForecastType.SAMPLES:
                forecasts.append(
                    SampleForecast(samples=item, start_date=forecast_start_date)
                )
            elif pipeline.forecast_type == ForecastType.QUANTILES:
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=list(map(str, pipeline.quantiles)),
                        start_date=forecast_start_date,
                    )
                )

        return forecasts

# %% [markdown]
# ## Evaluation
# 
# Now that we have our predictor class, we can use it to predict on the gift-eval benchmark datasets. We will use the `evaluate_model` function to evaluate the model. This function is a helper function to evaluate the model on the test data and return the results in a dictionary. We are going to follow the naming conventions explained in the [README](../README.md) file to store the results in a csv file called `all_results.csv` under the `results/chronos` folder.
# 
# The first column in the csv file is the dataset config name which is a combination of the dataset name, frequency and the term:
# 
# ```python
# f"{dataset_name}/{freq}/{term}"
# ```
# 

# %%
all_datasets

# %%
import logging


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

# %%
import csv
import os

from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset

# Iterate over all available datasets

model_name = "chronos_bolt_base"
# Output directory relative to gift-eval/notebooks/
output_dir = os.path.join("..", "results", model_name)
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(output_dir)}")

# Define the path for the CSV file
csv_file_path = os.path.join(output_dir, "all_results.csv")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Check if file exists and read completed datasets
import pandas as pd
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
        print(f"Dataset size: {len(dataset.test_data)}")
        predictor = ChronosPredictor(
            # use "amazon/chronos-t5-base" for the corresponding original Chronos model
            model_path="amazon/chronos-bolt-base",
            num_samples=20,
            prediction_length=dataset.prediction_length,
            device_map="cuda:0",
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

        print(f"Results for {ds_name} have been written to {csv_file_path}")

# %% [markdown]
# ## Results
# 
# Running the above cell will generate a csv file called `all_results.csv` under the `results/chronos` folder containing the results for the Chronos model on the gift-eval benchmark. We can display the csv file using the follow code:

# %%
import pandas as pd

results_file = os.path.join("..", "results", model_name, "all_results.csv")
df = pd.read_csv(results_file)
df

# %%



