# %% [markdown]
# # Quick Start: Running Foundation Model Moirai 2 on gift-eval benchmark
# 
# This notebook shows how to run the Foundation Model Moirai on the gift-eval benchmark.
# 
# Make sure you download the gift-eval benchmark and set the `GIFT-EVAL` environment variable correctly before running this notebook.
# 
# We will use the `Dataset` class to load the data and run the model. If you have not already please check out the [dataset.ipynb](./dataset.ipynb) notebook to learn more about the `Dataset` class. We are going to just run the model on two datasets for brevity. But feel free to run on any dataset by changing the `short_datasets` and `med_long_datasets` variables below.

# %% [markdown]
# ## Moirai Predictor
# We first install moirai via  
# 1. `pip install uni2ts`. 
# 
# Its predictor has already inherited `gluonts.torch.PyTorchPredictor` which is compatible with the `gluonts` evaluation pipeline. Note that there is a dependency conflict that `uni2ts` uses `gluonts~=0.14.3` for its evaluation while `gift_eval` requires `gluonts~=0.15.1`. 
# Here, we need to reinstall it via 
# 
# 2. `pip install gluonts==0.15.1`
# 
# We then load the model with initial hyperparameters, like `prediction_length`, `context_length`, `patch_size`, `num_samples`, etc. You can find more details about these hyperparameters from <https://github.com/SalesforceAIResearch/uni2ts/blob/main/example/moirai_forecast_pandas.ipynb>. 

# %%
from typing import Optional
import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast

from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
import logging
# skip the no mean value warning from QuantileForecast
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)

def get_device(device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    return device

class MoiraiQuantilePredictor:
    def __init__(
            self,
            model_path: str,
            prediction_length: int = 100,
            context_length: int = 4000,
            target_dim: int =1,
            feat_dynamic_real_dim: int =0,
            past_feat_dynamic_real_dim: int =0,
            device: str = 'auto',
            batch_size: int = 2048,
            quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        ):
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.device = get_device(device)
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
                    forecasts = self.model.predict(past_target) # full_forecasts shape: (batch num_quantiles future_time #tgt)
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

# %%
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
short_datasets = "sd/2019/15T"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = ""

# Get union of short and med_long datasets
all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

CUSTOM_CONTEXT_LENGTH = {
    "sd/2019/15T": 4000,
}

dataset_properties_map = json.load(open("./notebooks/dataset_properties.json"))

# %%
from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
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
# ## Evaluation
# 
# Now that we have our predictor class, we can use it to predict on the gift-eval benchmark datasets. We will use the `evaluate_model` function to evaluate the model. This function is a helper function to evaluate the model on the test data and return the results in a dictionary. We are going to follow the naming conventions explained in the [README](../README.md) file to store the results in a csv file called `all_results.csv` under the `results/moirai_small` folder.
# 
# The first column in the csv file is the dataset config name which is a combination of the dataset name, frequency and the term:
# 
# ```python
# f"{dataset_name}/{freq}/{term}"
# ```
# 

# %%
from gluonts.model import evaluate_model
import csv
import os
import time
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset

# Iterate over all available datasets

output_dir = "../results/Moirai2"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# Define the path for the CSV file
csv_file_path = os.path.join(output_dir, "all_results.csv")

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

for ds_name in all_datasets:
    ds_key = ds_name.split("/")[0]
    print(f"Processing dataset: {ds_name}")
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

        # Initialize the dataset, since Moirai support multivariate time series forecast, it does not require
        # to convert the original data into univariate
        to_univariate = False if Dataset(name=ds_name, term=term,to_univariate=False).target_dim == 1 else True
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)

        context_length = CUSTOM_CONTEXT_LENGTH.get(ds_name, 4000)

        predictor = MoiraiQuantilePredictor(
            model_path=f"Salesforce/moirai-2.0-R-small",
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            target_dim=1,
            past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
            batch_size=512,
        )

        season_length = get_seasonality(dataset.freq)

        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=512,
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

# %% [markdown]
# ## Results
# 
# Running the above cell will generate a csv file called `all_results.csv` under the `results/moirai_small` folder containing the results for the Moirai model on the gift-eval benchmark. The csv file will look like this:
# 

# %%
import pandas as pd

df = pd.read_csv("./results/moirai_small/all_results.csv")
df


