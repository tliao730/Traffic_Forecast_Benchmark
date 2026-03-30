import os
from typing import List

import numpy as np
import torch
from fm.fm_utils import (
    build_basic_parser,
    get_entry_target,
    load_pretrained_with_cache,
    run_benchmark,
    to_sample_forecasts,
)
from fm.torch_utils import run_with_batch_size_backoff
from gluonts.itertools import batcher
from gluonts.model import Forecast
from tqdm.auto import tqdm

import argparse

from load_model import setup_model_runtime

MODEL_NAME = "Kairos_50m"
MODEL_PATH = "mldi-lab/Kairos_50m"
DEFAULT_BATCH_SIZE = 256
DEFAULT_CONTEXT_MAX_LENGTH = 2048

MODEL_RUNTIME = setup_model_runtime("kairos", __file__)
benchmark_config = MODEL_RUNTIME.config

from tsfm.model.kairos import AutoModel  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = build_basic_parser("Kairos evaluation or time estimation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_get_env_int("KAIROS_EVAL_BATCH_SIZE", DEFAULT_BATCH_SIZE),
        help=(
            f"Evaluation batch size (default: KAIROS_EVAL_BATCH_SIZE or {DEFAULT_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--context-max-length",
        type=int,
        default=_get_env_int("KAIROS_CONTEXT_MAX_LENGTH", DEFAULT_CONTEXT_MAX_LENGTH),
        help=(
            "Context max length used for pad/truncate "
            f"(default: KAIROS_CONTEXT_MAX_LENGTH or {DEFAULT_CONTEXT_MAX_LENGTH})"
        ),
    )
    return parser


def pad_or_truncate(sequence, max_length: int = 2048, pad_value: float = np.nan) -> np.ndarray:
    """
    Pads or truncates a sequence on the left to a specified max_length.
    Logic copied from the original kairos.py.
    """
    seq_np = np.array(sequence)
    current_length = len(seq_np)

    if current_length < max_length:
        padding_size = max_length - current_length
        return np.pad(seq_np, (padding_size, 0), "constant", constant_values=pad_value)
    else:
        return seq_np[-max_length:]


def _get_env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return int(val)


class KairosPredictor:
    """
    Wrapper around tsfm.model.kairos.AutoModel to work with common.eval().
    This is the same predictor as in the old kairos.py, just without the
    dataset/evaluation loop.
    """

    def __init__(
        self,
        model_path: str,
        prediction_length: int,
        *args,
        **kwargs,
    ):
        print("prediction_length:", prediction_length)
        self.prediction_length = prediction_length
        # Check for CUDA availability and set the primary device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the model
        self.model = load_pretrained_with_cache(
            AutoModel,
            model_path,
            trust_remote_code=True,
            cache_dir=benchmark_config.hf_home,
        )

        # Move the model to the primary device
        self.model.to(self.device)

    def predict(self, test_data_input, batch_size: int = 256) -> List[Forecast]:
        context_max_len = _get_env_int(
            "KAIROS_CONTEXT_MAX_LENGTH", DEFAULT_CONTEXT_MAX_LENGTH
        )
        self.model.eval()
        model = self.model

        def _run_with_batch_size(current_batch_size: int) -> np.ndarray:
            forecast_outputs = []
            with torch.no_grad():
                for batch in tqdm(batcher(test_data_input, batch_size=current_batch_size)):
                    context = [
                        torch.tensor(
                            pad_or_truncate(
                                get_entry_target(entry),
                                max_length=context_max_len,
                            )
                        )
                        for entry in batch
                    ]
                    forecast_outputs.append(
                        model(
                            past_target=torch.stack(context).to(self.device),
                            prediction_length=self.prediction_length,
                            generation=True,
                            infer_is_positive=True,
                            force_flip_invariance=True,
                        )["prediction_outputs"].detach().cpu().numpy()
                    )
            return np.concatenate(forecast_outputs)

        forecast_outputs, _ = run_with_batch_size_backoff(
            _run_with_batch_size,
            batch_size,
        )

        # Convert forecast samples into gluonts Forecast objects
        return to_sample_forecasts(forecast_outputs, test_data_input)


def main():
    args = _build_parser().parse_args()

    def predictor_factory(dataset):
        return KairosPredictor(
            model_path=MODEL_PATH,
            prediction_length=dataset.prediction_length,
        )

    # Keep env var behavior for backwards compatibility while allowing CLI override.
    os.environ["KAIROS_CONTEXT_MAX_LENGTH"] = str(args.context_max_length)

    run_benchmark(
        eval_time_only=args.eval_time,
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        predictor_factory=predictor_factory,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
