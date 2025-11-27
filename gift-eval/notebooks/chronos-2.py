# %% [markdown]
# # Running Chronos 2 on gift-eval benchmark
# 
# **The following notebook is only intended to reproduce GIFT-Eval results using a GluonTS-style predictor interface. For practical usage, we recommend using the simpler interface of Chronos-2 as described in the [Github repo](https://github.com/amazon-science/chronos-forecasting).**
# 
# Make sure you download the gift-eval benchmark and set the `GIFT-EVAL` environment variable correctly before running this notebook.
# 
# We will use the `Dataset` class to load the data and run the model. If you have not already please check out the [dataset.ipynb](./dataset.ipynb) notebook to learn more about the `Dataset` class. We are going to just run the model on two datasets for brevity. But feel free to run on any dataset by changing the `SHORT_DATASETS` and `MED_LONG_DATASETS` variables below.

# %% [markdown]
# Install Chronos package:
# ``
# pip install chronos-forecasting>=2.0
# ``

# %%
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
SHORT_DATASETS = "sd/2019/15T"

#MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
MED_LONG_DATASETS = ""

# Get union of short and med_long datasets
all_datasets = list(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))

dataset_properties_map = json.load(open("./notebooks/dataset_properties.json"))

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

# %%
import logging
from typing import List

import numpy as np
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast


logger = logging.getLogger("Chronos-2 Predictor")
logger.setLevel(logging.INFO)


class Chronos2Predictor:
    def __init__(
        self,
        model_name: str,
        prediction_length: int,
        batch_size: int,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        predict_batches_jointly: bool = False,
        **kwargs
    ):
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            **kwargs,
        )
        assert isinstance(self.pipeline, Chronos2Pipeline), "This is Predictor is for Chronos-2, see other notebook for Chronos and Chronos-Bolt"
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.predict_batches_jointly = predict_batches_jointly


    def _pack_model_items(self, items):
        for item in items:
            model_input = {
                "target": item["target"],
            }
            yield model_input


    def predict(self, test_data_input) -> List[Forecast]:
        pipeline = self.pipeline
        model_batch_size = self.batch_size
        if self.predict_batches_jointly:
            logger.info("Note: Using cross learning mode. Please ensure that different rolling windows of the same time series are not in `test_data_input` to avoid any potential leakage due to in-context learning.")

        # Generate forecasts
        forecast_outputs = []
        input_data = list(self._pack_model_items(test_data_input))
        is_univariate_data = input_data[0]["target"].ndim == 1  # homogenous across all intputs
        while True:
            try:
                quantiles, _ = pipeline.predict_quantiles(
                        inputs=input_data,
                        prediction_length=self.prediction_length,
                        batch_size=model_batch_size,
                        quantile_levels=self.quantile_levels,
                        predict_batches_jointly=self.predict_batches_jointly,
                )
                quantiles = torch.stack(quantiles)
                # quantiles [batch, variates, seq_len, quantiles]
                quantiles = quantiles.permute(0, 3, 2, 1).cpu().numpy()
                # forecast_outputs [batch, quantiles, seq_len, variates]
                if is_univariate_data:
                    quantiles = quantiles.squeeze(-1) # squeeze variate to avoid error in eval due to broadcasting
                assert quantiles.shape[1] == len(self.quantile_levels)
                assert quantiles.shape[2] == self.prediction_length
                forecast_outputs.append(quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OutOfMemoryError at model_batch_size {model_batch_size}, reducing to {model_batch_size // 2}")
                model_batch_size //= 2

        # Convert forecasts into gluonts Forecast objects
        forecast_outputs = np.concatenate(forecast_outputs, axis=0)
        assert len(forecast_outputs) == len(input_data)
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecast = QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, self.quantile_levels)),
                start_date=forecast_start_date,
            )
            forecasts.append(forecast)
        return forecasts


# %% [markdown]
# ## Evaluation
# 
# Now that we have our predictor class, we can use it to predict on the gift-eval benchmark datasets. We will use the `evaluate_model` function to evaluate the model. This function is a helper function to evaluate the model on the test data and return the results in a dictionary. We are going to follow the naming conventions explained in the [README](../README.md) file to store the results in a csv file called `all_results.csv` under the `results/chronos-2` folder.
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
import itertools
import pandas as pd

from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset

model_name = "s3://autogluon/chronos-2"
output_dir = f"./results/chronos-2/all_results.csv"
pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
pretty_model_name = {
    "s3://autogluon/chronos-2": "Chronos-2",
}

def evaluate_on_dataset(
    model_name: str,
    ds_name: str,
    ds_term: str,
    batch_size: int,
    use_multivariate_data: bool = True,
    **predictor_kwargs,
):
    is_multivariate_source = (
        Dataset(
            name=ds_name,
            term=ds_term,
            to_univariate=False,
        ).target_dim
        > 1
    )

    dataset = Dataset(
        name=ds_name,
        term=ds_term,
        to_univariate=is_multivariate_source and not use_multivariate_data,
    )

    logger.info(f"Dataset size: {len(dataset.test_data)}")

    predictor = Chronos2Predictor(
        model_name=model_name,
        prediction_length=dataset.prediction_length,
        batch_size=batch_size,
        **predictor_kwargs,
    )
    
    # Avoid cross batch leakage of rolling evalution by prediction of windows individually.
    forecast_windows = []
    n_windows = dataset.test_data.windows
    for window_idx in range(n_windows):
        entries_window_k = list(itertools.islice(dataset.test_data.input, window_idx, None, n_windows))
        forecasts_window_k = list(predictor.predict(entries_window_k))
        forecast_windows.append(forecasts_window_k)        

    forecasts = [item for items in zip(*forecast_windows) for item in items] # interleave results again
    season_length = get_seasonality(dataset.freq)
    return evaluate_forecasts(
            forecasts,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        ) \
    .reset_index(drop=True) \
    .to_dict(orient="records")


all_results = []
for ds_num, ds_name in enumerate(all_datasets):
    ds_key = ds_name.split("/")[0]
    logger.info(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
    term = "short"

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

    logger.info(f"Generating forecasts for {ds_config}")
    all_results.append(
        (
            evaluate_on_dataset(
                model_name=model_name,
                ds_name=ds_name,
                ds_term=term,
                batch_size=100,
                use_multivariate_data=True,
                predict_batches_jointly=True,
                device_map="cuda",
                torch_dtype="float32",
            ),
            ds_config,
            dataset_properties_map[ds_key]["domain"],
            dataset_properties_map[ds_key]["num_variates"],
        )
    )


result_df_rows = []
for result_metrics, ds_config, domain, num_variates in all_results:
    result_metrics = {f"eval_metrics/{k}": v for k, v in result_metrics[0].items()}

    result_df_rows.append(
        {
            "dataset": ds_config,
            "model": pretty_model_name.get(model_name, model_name),
            **result_metrics,
            "domain": domain,
            "num_variates": num_variates,
        }
    )
results_df = pd.DataFrame(result_df_rows).sort_values(by="dataset")
results_df.to_csv(output_dir, index=False)
logger.info(f"Results have been written to {output_dir}.")



