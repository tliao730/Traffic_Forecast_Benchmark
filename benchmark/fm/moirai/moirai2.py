import argparse
import os

import numpy as np
import torch
from common import eval, eval_time
from config import config as benchmark_config
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


class MoiraiQuantilePredictor:
    def __init__(
        self,
        model_path: str,
        prediction_length: int = 100,
        context_length: int = 4000,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
        device_str: str = "auto",
        batch_size: int = 2048,
        quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.device = device_str
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        cache_dir = benchmark_config.hf_home
        try:
            module = Moirai2Module.from_pretrained(self.model_path, cache_dir=cache_dir)
        except TypeError:
            module = Moirai2Module.from_pretrained(self.model_path)
        self.model = Moirai2Forecast(
            module=module,
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
                for batch in batcher(test_data_input, batch_size=self.batch_size):
                    # Handle both dict and tuple formats
                    past_target = []
                    for entry in batch:
                        if isinstance(entry, tuple):
                            # Entry is (input_dict, label) tuple
                            past_target.append(entry[0]["target"])
                        else:
                            # Entry is just a dict
                            past_target.append(entry["target"])

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
            # Handle both dict and tuple formats
            if isinstance(ts, tuple):
                ts_dict = ts[0]  # Extract input dict from tuple
            else:
                ts_dict = ts

            forecast_start_date = ts_dict["start"] + len(ts_dict["target"])
            quantile_forecasts.append(
                QuantileForecast(
                    item_id=ts_dict.get("item_id", "unknown"),
                    forecast_arrays=item,
                    start_date=forecast_start_date,
                    forecast_keys=list(map(str, self.quantile_levels)),
                )
            )
        return quantile_forecasts


def main():
    model_name = "Moirai2"
    model_path = "Salesforce/moirai-2.0-R-small"
    device = "cuda"

    def predictor_factory(dataset):
        return MoiraiQuantilePredictor(
            model_path=model_path,
            prediction_length=dataset.prediction_length,
            context_length=4000,
            target_dim=1,
            feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
            batch_size=64,
            device_str=device,
        )

    parser = argparse.ArgumentParser(
        description="Moirai2 evaluation or time estimation"
    )
    parser.add_argument(
        "--eval-time", action="store_true", help="Run eval_time (time estimation) only"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save all predictions and ground truth to CSV (in addition to metrics)",
    )
    parser.add_argument(
        "--pred-out-dir",
        type=str,
        default=None,
        help="Directory for prediction CSVs (default: result_root/Moirai2/predictions)",
    )
    args = parser.parse_args()

    if args.eval_time:
        eval_time(model_name, model_path, predictor_factory)
    else:
        save_dir = args.pred_out_dir
        if args.save_predictions and save_dir is None:
            from config import config

            save_dir = os.path.join(config.result_root, model_name, "predictions")
        eval(
            model_name,
            model_path,
            predictor_factory,
            save_predictions_dir=save_dir if args.save_predictions else None,
        )


if __name__ == "__main__":
    main()
