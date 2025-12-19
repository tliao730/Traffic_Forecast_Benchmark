import csv
import json
import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from config import device, med_long_datasets, short_datasets
from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE,
                                SMAPE, MeanWeightedSumQuantileLoss)
from gluonts.itertools import batcher
from gluonts.model import Forecast, evaluate_model
from gluonts.model.forecast import SampleForecast
from gluonts.time_feature import get_seasonality
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

# Add Kairos to Python path - update this to your Kairos path
kairos_path = '/home/defu/workspace/Kairos'
if kairos_path not in sys.path:
    sys.path.insert(0, kairos_path)

from tsfm.model.kairos import AutoModel

# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

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


def pad_or_truncate(sequence, max_length=2048, pad_value=np.nan):
    """
    Pads or truncates a sequence on the left to a specified max_length.

    Args:
        sequence (list or np.ndarray): The input sequence.
        max_length (int): The target length.
        pad_value (int or float): The value to use for padding, defaults to np.nan.

    Returns:
        np.ndarray: A NumPy array of length max_length.
    """
    seq_np = np.array(sequence)
    current_length = len(seq_np)

    if current_length < max_length:
        padding_size = max_length - current_length
        return np.pad(seq_np, (padding_size, 0), 'constant', constant_values=pad_value)
    else:
        return seq_np[-max_length:]


class KairosPredictor:
    def __init__(
        self,
        model_path,
        prediction_length: int,
        *args,
        **kwargs,
    ):
        print("prediction_length:", prediction_length)
        self.prediction_length = prediction_length
        # Check for CUDA availability and set the primary device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the model
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        # Move the model to the primary device
        self.model.to(self.device)

    def predict(self, test_data_input, batch_size: int = 256) -> List[Forecast]:
        self.model.eval()
        model = self.model
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                with torch.no_grad():
                    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                        context = [torch.tensor(pad_or_truncate(entry["target"], max_length=2048)) for entry in batch]
                        forecast_outputs.append(
                            model(
                                past_target=torch.stack(context).to(self.device),
                                prediction_length=self.prediction_length,
                                generation=True,
                                infer_is_positive=True,
                                force_flip_invariance=True,
                            )["prediction_outputs"].detach().cpu().numpy()
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
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return forecasts


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

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Model Configuration
# model_name = "Kairos_10m"
# model_path = "mldi-lab/Kairos_10m"
# model_name = "Kairos_23m"
# model_path = "mldi-lab/Kairos_23m"
model_name = "Kairos_50m"
model_path = "mldi-lab/Kairos_50m"

# Output directory relative to benchmark/
output_dir = os.path.join("..", "results", model_name)
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {os.path.abspath(output_dir)}")
csv_file_path = os.path.join(output_dir, "all_results.csv")

# Check if file exists and read completed datasets
completed_datasets = set()
if os.path.exists(csv_file_path):
    print(f"'{csv_file_path}' exists. Reading completed datasets...")
    with open(csv_file_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if row:
                completed_datasets.add(row[0])
    print(f"Found {len(completed_datasets)} completed datasets.")
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

        if ds_config in completed_datasets:
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
        predictor = KairosPredictor(
            model_path=model_path,
            prediction_length=dataset.prediction_length,
        )
        # Measure the time taken for evaluation
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
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


results_file = os.path.join("..", "results", model_name, "all_results.csv")
df = pd.read_csv(results_file)
print("\nFinal aggregated results:")
print(df)

