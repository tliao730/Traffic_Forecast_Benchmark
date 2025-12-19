import csv
import json
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from config import device, med_long_datasets, short_datasets
from dotenv import load_dotenv
from gift_eval.data import Dataset
from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE,
                                SMAPE, MeanWeightedSumQuantileLoss)
from gluonts.model import evaluate_model

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Set GIFT_EVAL environment variable
# os.environ['GIFT_EVAL'] = '/home/defu/workspace/LargeST/data/gift_eval_datasets'

# FlowState path - update this to your granite-tsfm path
sys.path.append(os.path.realpath("/home/defu/workspace/LargeST/granite-tsfm/"))

from tsfm_public import FlowStateForPrediction
from notebooks.hfdemo.flowstate.gift_wrapper import FlowState_Gift_Wrapper

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
    MAE(forecast_type="mean"),
    MAE(forecast_type=0.5),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]

# Auxiliary functions
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def LoadFlowState(pred_length, n_ch, freq, device='cpu', domain=None, nd=False, batch_size=16):
    model_name = 'ibm-research/FlowState'
    # model_name = 'ibm-granite/granite-timeseries-flowstate-r1'  # alternative model
    flowstate = FlowStateForPrediction.from_pretrained(model_name).to(device)

    config = flowstate.config
    config.min_context = 0
    config.device = device
    flowstate = FlowState_Gift_Wrapper(flowstate, pred_length, n_ch=n_ch, batch_size=batch_size,
                                 f=freq, device=device, domain=domain, no_daily=nd)
    return flowstate

# Model configuration
model_name = "FlowState-9.1M"
seed = 0
batch_size = 16

# Output directory relative to benchmark/
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
                "eval_metrics/MAE[mean]",
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

set_seed(seed)

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

        set_seed(seed)
        
        # Initialize the dataset
        to_univariate = (
            False
            if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
            else True
        )
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
        print(f"Dataset size: {len(dataset.test_data)}")

        all_lengths = []
        for x in dataset.test_data:
            if len(x[0]["target"].shape) == 1:
                all_lengths.append(len(x[0]["target"]))
                num_channels = 1
            else:
                all_lengths.append(x[0]["target"].shape[1])
                num_channels = x[0]["target"].shape[0]

        no_daily = 'l2c' in ds_name  # necessary to get correct seasonality for bizitobs_l2c datasets
        flowstate = LoadFlowState(pred_length=dataset.prediction_length,
                                 n_ch=num_channels,
                                 freq=dataset.freq,
                                 device=device,
                                 domain=dataset_properties_map[ds_key]["domain"],
                                 nd=no_daily,
                                 batch_size=batch_size,
                                 )

        with torch.no_grad():
            # Evaluate
            res = evaluate_model(
                flowstate,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
            )

        print(f'MASE: {res["MASE[0.5]"][0]}')

        # Append the results to the CSV file
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    ds_config,
                    model_name,
                    res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[mean]"][0],
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

