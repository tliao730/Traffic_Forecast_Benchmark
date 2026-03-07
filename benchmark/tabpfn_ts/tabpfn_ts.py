import os
import sys
from typing import Iterator

import argparse
from common import eval, eval_time
from gluonts.model.forecast import Forecast

# Try multiple possible locations for tabpfn-time-series
_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
_trafficfm_root = os.path.dirname(_benchmark_dir)
possible_paths = [
    os.path.join(_trafficfm_root, "envs", "tabpfn_ts", "tabpfn-time-series"),  # env-specific install
    os.path.join(os.path.expanduser("~"), "tabpfn-time-series"),
    os.path.join(_trafficfm_root, "tabpfn-time-series"),
    "tabpfn-time-series",  # relative to current working directory
]

tabpfn_path = None
for path in possible_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        tabpfn_path = abs_path
        break

if tabpfn_path is None:
    raise FileNotFoundError(
        "tabpfn-time-series repository not found. Please clone it first:\n"
        "  git clone https://github.com/PriorLabs/tabpfn-time-series.git\n"
        "  cd tabpfn-time-series && git checkout v1.0.0\n"
        f"  Searched in: {', '.join(possible_paths)}"
    )

# Add both the main repository and the gift_eval subdirectory to the path
sys.path.append(tabpfn_path)
gift_eval_path = os.path.join(tabpfn_path, "gift_eval")
if os.path.exists(gift_eval_path):
    sys.path.append(gift_eval_path)

print(f"Added tabpfn-time-series to Python path: {tabpfn_path}")

# Import the TabPFN time series predictor class
try:
    from tabpfn_ts_wrapper import TabPFNTSPredictor, TabPFNMode
except ImportError as e:
    requirements_path = os.path.join(tabpfn_path, "requirements.txt")
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
    model_name = "tabpfn_ts"
    model_path = "tabpfn-time-series"  # This is just a label, not an actual model path
    tabpfn_mode = TabPFNMode.LOCAL

    def predictor_factory(dataset):
        return TabPFNPredictor(
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            tabpfn_mode=tabpfn_mode,
            context_length=4096,
        )

    parser = argparse.ArgumentParser(description="TabPFN-TS evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    if args.eval_time:
        eval_time(model_name, model_path, predictor_factory)
    else:
        eval(model_name, model_path, predictor_factory, batch_size=1024)


if __name__ == "__main__":
    main()

