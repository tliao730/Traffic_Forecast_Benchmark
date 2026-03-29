import os
from typing import Iterator

import argparse
from gluonts.model.forecast import Forecast

from load_model import setup_model_environment

MODEL_NAME = "tabpfn_ts"
MODEL_PATH = "tabpfn-time-series"  # Label used in benchmark outputs.
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_BATCH_SIZE = 1024

MODEL_ENV = setup_model_environment("tabpfn_ts", __file__)
from common import eval, eval_time

tabpfn_path = MODEL_ENV.require_repo_path()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TabPFN-TS evaluation or time estimation")
    parser.add_argument(
        "--eval-time",
        action="store_true",
        help="Run eval_time (time estimation) only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=f"TabPFN context length (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    return parser


print(f"Added tabpfn-time-series to Python path: {tabpfn_path}")

# Import the TabPFN time series predictor class
try:
    from tabpfn_ts_wrapper import TabPFNTSPredictor, TabPFNMode
except ImportError as e:
    requirements_path = MODEL_ENV.repo_file("requirements.txt")
    raise ImportError(
        f"Failed to import tabpfn_ts_wrapper: {e}\n"
        "\n"
        "Please ensure all dependencies are installed:\n"
        f"  pip install -r {requirements_path}"
    ) from e


class TabPFNPredictor:
    """
    Wrapper class for TabPFNTSPredictor to work with common.eval() function.
    """

    def __init__(
        self,
        prediction_length: int,
        ds_freq: str,
        tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL,
        context_length: int = 4096,
    ):
        print("prediction_length:", prediction_length)
        self.prediction_length = prediction_length
        self.ds_freq = ds_freq
        self.tabpfn_mode = tabpfn_mode
        self.context_length = context_length
        self.predictor = TabPFNTSPredictor(
            ds_prediction_length=prediction_length,
            ds_freq=ds_freq,
            tabpfn_mode=tabpfn_mode,
            context_length=context_length,
        )

    def predict(self, test_data_input) -> Iterator[Forecast]:
        """
        Predict method that returns an iterator of Forecast objects.
        This method delegates to the underlying TabPFNTSPredictor.
        """
        return self.predictor.predict(test_data_input)


def main():
    args = _build_parser().parse_args()
    tabpfn_mode = TabPFNMode.LOCAL

    def predictor_factory(dataset):
        return TabPFNPredictor(
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            tabpfn_mode=tabpfn_mode,
            context_length=args.context_length,
        )

    if args.eval_time:
        eval_time(MODEL_NAME, MODEL_PATH, predictor_factory)
    else:
        eval(MODEL_NAME, MODEL_PATH, predictor_factory, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
