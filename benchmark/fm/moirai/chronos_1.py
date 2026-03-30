import argparse

import numpy as np
import torch
from chronos import BaseChronosPipeline, ForecastType
from config import config as benchmark_config
from fm.fm_utils import build_basic_parser, load_pretrained_with_cache, run_benchmark
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm import tqdm

MODEL_NAME = "chronos_bolt_base"
MODEL_PATH = "amazon/chronos-bolt-base"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_NUM_SAMPLES = 20
DEFAULT_BATCH_SIZE = 1024


def _build_parser() -> argparse.ArgumentParser:
    parser = build_basic_parser("Chronos evaluation or time estimation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of generated samples (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device map passed to Chronos pipeline (default: {DEFAULT_DEVICE})",
    )
    return parser


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
        kwargs.pop("cache_dir", None)  # Ensure YAML cache dir is sourced centrally.
        self.pipeline = load_pretrained_with_cache(
            BaseChronosPipeline,
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
    args = _build_parser().parse_args()

    def predictor_factory(dataset):
        return ChronosPredictor(
            model_path=MODEL_PATH,
            num_samples=args.num_samples,
            prediction_length=dataset.prediction_length,
            device_map=args.device,
        )

    run_benchmark(
        eval_time_only=args.eval_time,
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        predictor_factory=predictor_factory,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
