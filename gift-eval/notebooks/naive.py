# %% [markdown]
# # Quick Start: Running Naive model on gift-eval benchmark
# 
# This notebook shows how to run the Naive model on the gift-eval benchmark.
# 
# Make sure you download the gift-eval benchmark and set the `GIFT-EVAL` environment variable correctly before running this notebook.
# 
# We will use the `Dataset` class to load the data and run the model. If you have not already please check out the [dataset.ipynb](./dataset.ipynb) notebook to learn more about the `Dataset` class. We are going to just run the model on two datasets for brevity. But feel free to run on any dataset by changing the `short_datasets` and `med_long_datasets` variables below.

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
# ## StatsForecast Predictor
# 
# We will use the `StatsForecastPredictor` class to wrap the Naive model. This class is a wrapper around the `StatsForecast` library and is used to make it compatible with the `gluonts` interface. Note that `StatsForecastPredictor` is compatible with any model from the `statsforecast` library, however for brevity we will just use the `Naive` model in this notebook.
# 
# This is just meant to be a simple wrapper to get you started, feel free to use your own custom implementation to wrap any model.

# %%
import inspect
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Type
import logging

import numpy as np
import pandas as pd
from gluonts.core.component import validated
from gluonts.dataset import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation, MissingValueImputation
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
)


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


class StatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast`_ package.

    This class is used via subclassing and setting the ``ModelType`` class
    attribute to specify the ``statsforecast`` model type to use.

    .. _statsforecast: https://github.com/Nixtla/statsforecast

    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    quantile_levels
        Optional list of quantile levels that we want predictions for.
        Note: this is only supported by specific types of models, such as
        ``AutoARIMA``. By default this is ``None``, giving only the mean
        prediction.
    **model_params
        Keyword arguments to be passed to the model type for construction.
        The specific arguments accepted or required depend on the
        ``ModelType``; please refer to the documentation of ``statsforecast``
        for details.
    """

    ModelType: Type

    @validated()
    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        freq: str,
        quantile_levels: Optional[List[float]] = None,
        imputation_method: MissingValueImputation = LastValueImputation(),
        max_length: Optional[int] = None,
        batch_size: int = 1,
        parallel: bool = False,
        **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        if "season_length" in inspect.signature(self.ModelType.__init__).parameters:
            model_params["season_length"] = season_length

        self.freq = freq
        self.model = StatsForecast(
            models=[self.ModelType(**model_params)],
            freq=freq,
            fallback_model=SeasonalNaive(season_length=season_length),
            n_jobs=-1 if parallel else 1,
        )
        self.fallback_model = StatsForecast(
            # Fallback model when main model returns NaNs
            models=[SeasonalNaive(season_length=season_length)],
            freq=freq,
            n_jobs=-1 if parallel else 1,
        )
        self.config = ModelConfig(quantile_levels=quantile_levels)
        self.imputation_method = imputation_method
        self.batch_size = batch_size
        self.max_length = max_length

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        batch = {}
        total_series = len(dataset)
        self.logger.info(f"Starting prediction on {total_series} series.")

        for idx, entry in enumerate(dataset):
            assert entry["target"].ndim == 1, "only for univariate time series"
            assert (
                len(entry["target"]) >= 1
            ), "all time series should have at least one data point"

            if self.max_length is not None:
                entry["start"] += len(entry["target"][: -self.max_length])
                entry["target"] = entry["target"][-self.max_length :]

            target = np.asarray(entry["target"], np.float32)
            if np.isnan(target).any():
                target = target.copy()
                target = self.imputation_method(target)

            unique_id = (
                f"{entry['item_id']}_{str(forecast_start(entry))}_{str(len(batch))}"
            )
            start = entry["start"]
            batch[unique_id] = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        start=start.to_timestamp(),
                        periods=len(target),
                        freq=start.freq,
                    ).to_numpy(),
                    "y": target,
                }
            )

            if len(batch) == self.batch_size:
                self.logger.info(f"Processing batch {idx // self.batch_size + 1}.")
                results = self.sf_predict(pd.concat(batch.values()))
                yield from self.yield_forecast(batch.keys(), results)
                batch = {}

        if len(batch) > 0:
            self.logger.info(f"Processing final batch.")
            results = self.sf_predict(pd.concat(batch.values()))
            yield from self.yield_forecast(batch.keys(), results)

        self.logger.info("Prediction completed.")

    def sf_predict(self, Y_df: pd.DataFrame) -> pd.DataFrame:
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals
        results = self.model.forecast(
            df=Y_df,
            h=self.prediction_length,
            **kwargs,
        )
        # replace nan results with fallback
        row_nan = results.isnull().values.any(axis=-1)
        if row_nan.any():
            nan_ids = results[row_nan].index.values
            nan_df = Y_df[Y_df["unique_id"].isin(nan_ids)]
            fallback_results = self.fallback_model.forecast(
                df=nan_df,
                h=self.prediction_length,
                **kwargs,
            )
            results = pd.concat(
                [
                    results[~results.index.isin(nan_ids)],
                    fallback_results,
                ]
            )

        return results

    def yield_forecast(
        self, item_ids, results: pd.DataFrame
    ) -> Iterator[QuantileForecast]:
        results.set_index('unique_id',inplace=True)
        for idx in item_ids:
            prediction = results.loc[idx]
            forecast_arrays = []
            model_name = self.ModelType.__name__
            for key in self.config.statsforecast_keys:
                if key == "mean":
                    forecast_arrays.append(prediction.loc[:, model_name].to_numpy())
                else:
                    forecast_arrays.append(
                        prediction.loc[:, f"{model_name}-{key}"].to_numpy()
                    )

            yield QuantileForecast(
                forecast_arrays=np.stack(forecast_arrays, axis=0),
                forecast_keys=self.config.forecast_keys,
                start_date=prediction.ds.iloc[0].to_period(freq=self.freq),
                item_id=idx,
            )


class NaivePredictor(StatsForecastPredictor):
    """
    A predictor wrapping the ``Naive`` model from `statsforecast`_.

    See :class:`StatsForecastPredictor` for the list of arguments.

    .. _statsforecast: https://github.com/Nixtla/statsforecast
    """

    ModelType = Naive

# %% [markdown]
# ## Evaluation
# 
# Now that we have our predictor class, we can use it to predict on the gift-eval benchmark datasets. We will use the `evaluate_model` function to evaluate the model. This function is a helper function to evaluate the model on the test data and return the results in a dictionary. We are going to follow the naming conventions explained in the [README](../README.md) file to store the results in a csv file called `all_results.csv` under the `results/naive` folder.
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

output_dir = "../results/naive"
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the path for the CSV file
csv_file_path = os.path.join(output_dir, "all_results.csv")

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

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
    print(f"Processing dataset: {ds_name}")

    # We only need the short term for sd/2019/15T
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

    # Initialize the dataset
    to_univariate = (
        False
        if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1
        else True
    )
    dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
    season_length = get_seasonality(dataset.freq)

    # Initialize the predictor
    predictor = NaivePredictor(
        dataset.prediction_length,
        season_length=season_length,
        freq=dataset.freq,
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        batch_size=512,
    )

    # Evaluate
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
                "naive",
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
# Running the above cell will generate a csv file called `all_results.csv` under the `results/naive` folder containing the results for the Naive model on the gift-eval benchmark. The csv file will look like this:
# 

# %%
import pandas as pd

df = pd.read_csv("../results/naive/all_results.csv")
df


