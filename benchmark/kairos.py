import os
import sys
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm

import argparse
from common import eval, eval_time

# Load environment variables
load_dotenv()

# Add Kairos repo to Python path (tsfm is from the cloned Kairos repo, not pip).
# Set KAIROS_PATH to your clone, e.g. export KAIROS_PATH=/path/to/Kairos
kairos_path = os.environ.get("KAIROS_PATH")
if not kairos_path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(script_dir, "..", "Kairos"),
        os.path.join(script_dir, "..", "envs", "kairos", "Kairos"),
    ]:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "tsfm")):
            kairos_path = os.path.abspath(candidate)
            break
if not kairos_path or not os.path.isdir(kairos_path):
    raise SystemExit(
        "Kairos repo not found. Clone it and set KAIROS_PATH:\n"
        "  git clone https://github.com/foundation-model-research/Kairos.git\n"
        "  export KAIROS_PATH=/path/to/Kairos\n"
        "  uv run python benchmark/kairos_new.py ..."
    )
if kairos_path not in sys.path:
    sys.path.insert(0, kairos_path)

from tsfm.model.kairos import AutoModel  # noqa: E402


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
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        # Move the model to the primary device
        self.model.to(self.device)

    def predict(self, test_data_input, batch_size: int = 256) -> List[Forecast]:
        self.model.eval()
        model = self.model
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                with torch.no_grad():
                    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                        context = [
                            torch.tensor(pad_or_truncate(entry["target"], max_length=2048))
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
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2

        # Convert forecast samples into gluonts Forecast objects
        forecasts: List[Forecast] = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(SampleForecast(samples=item, start_date=forecast_start_date))

        return forecasts


def main():
    # Model configuration (same defaults as old kairos.py)
    model_name = "Kairos_50m"
    model_path = "mldi-lab/Kairos_50m"

    def predictor_factory(dataset):
        """
        Given a Dataset object from common.eval, construct the Kairos predictor.
        """
        return KairosPredictor(
            model_path=model_path,
            prediction_length=dataset.prediction_length,
        )

    parser = argparse.ArgumentParser(description="Kairos evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    if args.eval_time:
        eval_time(model_name, model_path, predictor_factory)
    else:
        eval(model_name, model_path, predictor_factory, batch_size=256)


if __name__ == "__main__":
    main()

