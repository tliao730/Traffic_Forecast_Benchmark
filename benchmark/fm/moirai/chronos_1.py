import argparse
import numpy as np
import torch
from config import config as benchmark_config
from chronos import BaseChronosPipeline, ForecastType
from common import eval, eval_time
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm import tqdm


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
        kwargs.pop("cache_dir", None)  # Ensure YAML cache dir is used
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            *args,
            cache_dir=benchmark_config.hf_home,
            **kwargs,
        )
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(self, test_data_input, batch_size: int = 1024):
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


def main():
    parser = argparse.ArgumentParser(description="Chronos evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    model_name = "chronos_bolt_base"
    model_path = "amazon/chronos-bolt-base"
    device = "cuda:0"

    def predictor_factory(dataset):
        return ChronosPredictor(
            model_path=model_path,
            num_samples=20,
            prediction_length=dataset.prediction_length,
            device_map=device,
        )

    if args.eval_time:
        eval_time(model_name, model_path, predictor_factory)
    else:
        eval(model_name, model_path, predictor_factory)


if __name__ == "__main__":
    main()
