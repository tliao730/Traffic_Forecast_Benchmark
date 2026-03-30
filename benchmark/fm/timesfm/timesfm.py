import argparse
import warnings
from typing import List

import numpy as np
from dotenv import load_dotenv
from fm.fm_utils import (
    build_basic_parser,
    get_entry_target,
    run_benchmark,
    to_quantile_forecasts,
)
from gluonts.itertools import batcher
from gluonts.model import Forecast
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

MODEL_NAME = "timesfm_2_0_500m"
MODEL_PATH = "google/timesfm-2.0-500m-jax"
DEFAULT_BATCH_SIZE = 1024

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

def _build_parser() -> argparse.ArgumentParser:
    parser = build_basic_parser("TimesFM evaluation or time estimation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    return parser


def _load_timesfm_model():
    print("Loading TimesFM model...")
    model = timesfm.TimesFm(
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
            huggingface_repo_id=MODEL_PATH
        ),
    )
    print("Model loaded successfully.")
    return model


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

    def predict(self, test_data_input, batch_size: int = DEFAULT_BATCH_SIZE) -> List[Forecast]:
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
            context = [np.array(get_entry_target(entry)) for entry in batch]
            freqs = [self.freq] * len(context)
            _, full_preds = self.tfm.forecast(context, freqs, normalize=True)
            full_preds = full_preds[:, 0 : self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))
        forecast_outputs = np.concatenate(forecast_outputs)

        return to_quantile_forecasts(
            forecast_outputs,
            test_data_input,
            self.tfm.quantiles,
        )


def main():
    args = _build_parser().parse_args()
    tfm = _load_timesfm_model()

    def predictor_factory(dataset):
        return TimesFmPredictor(
            tfm=tfm,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
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
