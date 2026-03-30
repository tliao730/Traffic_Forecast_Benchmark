import os
from typing import Iterator

import argparse
from fm.fm_utils import build_basic_parser, run_benchmark
from gluonts.model.forecast import Forecast

from load_model import setup_model_runtime

MODEL_NAME = "tabpfn_ts"
MODEL_PATH = "tabpfn-time-series"  # Label used in benchmark outputs.
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_BATCH_SIZE = 1024

MODEL_RUNTIME = setup_model_runtime("tabpfn_ts", __file__)
tabpfn_path = MODEL_RUNTIME.require_repo_path()


def _build_parser() -> argparse.ArgumentParser:
    parser = build_basic_parser("TabPFN-TS evaluation or time estimation")
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
    requirements_path = MODEL_RUNTIME.repo_file("requirements.txt")
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

    run_benchmark(
        eval_time_only=args.eval_time,
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        predictor_factory=predictor_factory,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
