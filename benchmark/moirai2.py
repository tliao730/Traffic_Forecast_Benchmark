import csv
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from config import device, med_long_datasets, short_datasets
from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE,
                                SMAPE, MeanWeightedSumQuantileLoss)
from gluonts.itertools import batcher
from gluonts.model import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# Load environment variables
load_dotenv()

# skip the no mean value warning from QuantileForecast
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)


def get_device(device_str="auto"):
    if device_str == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device_str)
    return device_obj


class MoiraiQuantilePredictor:
    def __init__(
            self,
            model_path: str,
            prediction_length: int = 100,
            context_length: int = 4000,
            target_dim: int = 1,
            feat_dynamic_real_dim: int = 0,
            past_feat_dynamic_real_dim: int = 0,
            device_str: str = 'auto',
            batch_size: int = 2048,
            quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        ):
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.device = get_device(device_str)
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.model_path),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
        ).to(self.device)

    def predict(self, test_data_input):
        while True:
            try:
                print("Model - MoiraiQuantile loaded with batch_size:", self.batch_size)
                # Generate forecast samples
                forecast_quantiles = []
                for batch in (batcher(test_data_input, batch_size=self.batch_size)):
                    past_target = [entry["target"] for entry in batch]
                    forecasts = self.model.predict(past_target)
                    forecast_quantiles.append(forecasts)
                forecast_quantiles = np.concatenate(forecast_quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {self.batch_size}, reducing to {self.batch_size // 2}"
                )
                self.batch_size //= 2

        # Convert forecast samples into gluonts QuantileForecast objects
        quantile_forecasts = []
        for item, ts in zip(forecast_quantiles, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            quantile_forecasts.append(
                QuantileForecast(
                item_id = ts["item_id"],
                forecast_arrays=item,
                start_date=forecast_start_date,
                forecast_keys=list(map(str, self.quantile_levels)))
            )
        return quantile_forecasts


# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

CUSTOM_CONTEXT_LENGTH = {
    "sd/2019/15T": 4000,
}

# Determine the correct path for dataset_properties.json
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_properties_path = os.path.join(script_dir, "dataset_properties.json")

if not os.path.exists(dataset_properties_path):
    # Fallback to trying relative paths
    if os.path.exists("./dataset_properties.json"):
        dataset_properties_path = "./dataset_properties.json"
    elif os.path.exists("./benchmark/dataset_properties.json"):
        dataset_properties_path = "./benchmark/dataset_properties.json"
    else:
        raise FileNotFoundError(
            "dataset_properties.json not found. Make sure you're running this script from benchmark/ directory."
        )

dataset_properties_map = json.load(open(dataset_properties_path))

# Add properties for new datasets (ca, gba, gla, sd)
if 'ca' not in dataset_properties_map:
    dataset_properties_map['ca'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}
if 'gba' not in dataset_properties_map:
    dataset_properties_map['gba'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}
if 'gla' not in dataset_properties_map:
    dataset_properties_map['gla'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}
if 'sd' not in dataset_properties_map:
    dataset_properties_map['sd'] = {"frequency": "15T", "domain": "Transport", "num_variates": 1}

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

model_name = "Moirai2"
# Output directory relative to benchmark/
output_dir = os.path.join("..", "results", model_name)
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(output_dir)}")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Define the path for the CSV file
csv_file_path = os.path.join(output_dir, "all_results.csv")

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
            parts = ds_name.split("/")
            ds_key = parts[0]
            ds_freq = parts[-1]
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
        to_univariate = False if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1 else True
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)

        print(f"Prediction length: {dataset.prediction_length}")
        print(f"Dataset size: {len(dataset.test_data)}")

        context_length = CUSTOM_CONTEXT_LENGTH.get(ds_name, 4000)

        predictor = MoiraiQuantilePredictor(
            model_path=f"Salesforce/moirai-2.0-R-small",
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            target_dim=1,
            past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
            batch_size=64,
        )

        season_length = get_seasonality(dataset.freq)

        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=64,
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
                    "moirai_small",
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


results_file = os.path.join("..", "results", model_name, "all_results.csv")
df = pd.read_csv(results_file)
print("\nFinal aggregated results:")
print(df)

