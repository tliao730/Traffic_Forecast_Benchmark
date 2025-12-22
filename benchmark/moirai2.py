import numpy as np
import torch
from common import eval
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
                for batch in batcher(test_data_input, batch_size=self.batch_size):
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
                    item_id=ts["item_id"],
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

    eval(model_name, model_path, predictor_factory)


if __name__ == "__main__":
    main()
