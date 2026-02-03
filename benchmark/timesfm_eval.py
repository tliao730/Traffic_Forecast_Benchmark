import argparse
import logging
import os
import sys
import warnings
from typing import List

import numpy as np
from dotenv import load_dotenv
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm

from common import eval, eval_time

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Ensure we can import timesfm
try:
    import timesfm
except ImportError as e:
    raise ImportError(
        "timesfm package is not installed. Please install it by running:\n"
        "  pip install 'timesfm[pax]'  # for JAX backend (python 3.10.x)\n"
        "  or\n"
        "  pip install 'timesfm[torch]'  # for PyTorch backend (python 3.11.x)\n"
        "\n"
        "Original error: " + str(e)
    ) from e


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))


class TimesFmPredictor:
    """
    Wrapper around a TimesFm model to be used with common.eval().
    This is functionally the same as the predictor in timesfm_eval.py,
    but without the dataset/evaluation loop.
    """

    def __init__(
        self,
        tfm,
        prediction_length: int,
        ds_freq: str,
        *args,
        **kwargs,
    ):
        self.tfm = tfm
        self.prediction_length = prediction_length
        if self.prediction_length > self.tfm.horizon_len:
            self.tfm.horizon_len = (
                (self.prediction_length + self.tfm.output_patch_len - 1)
                // self.tfm.output_patch_len
            ) * self.tfm.output_patch_len
            print("Jitting for new prediction length.")
        self.freq = timesfm.freq_map(ds_freq)

    def predict(self, test_data_input, batch_size: int = 1024) -> List[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = []
            for entry in batch:
                arr = np.array(entry["target"])
                context.append(arr)
            freqs = [self.freq] * len(context)
            _, full_preds = self.tfm.forecast(context, freqs, normalize=True)
            full_preds = full_preds[:, 0 : self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
        forecast_outputs = np.concatenate(forecast_outputs)

        # Convert forecast samples into gluonts Forecast objects
        forecasts: List[Forecast] = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.tfm.quantiles)),
                    start_date=forecast_start_date,
                )
            )

        return forecasts


def main():
    model_name = "timesfm_2_0_500m"
    model_path = "google/timesfm-2.0-500m-jax"

    # Load the TimesFM model once, as in the original script
    print("Loading TimesFM model...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            num_layers=50,
            horizon_len=128,
            context_len=2048,
            use_positional_embedding=False,
            output_patch_len=128,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-jax"
        ),
    )
    print("Model loaded successfully.")

    def predictor_factory(dataset):
        """
        Given a Dataset from common.eval, construct a TimesFmPredictor.
        """
        return TimesFmPredictor(
            tfm=tfm,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
        )

    parser = argparse.ArgumentParser(description="TimesFM evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    if args.eval_time:
        eval_time(model_name, model_path, predictor_factory)
    else:
        eval(model_name, model_path, predictor_factory, batch_size=1024)


if __name__ == "__main__":
    main()

