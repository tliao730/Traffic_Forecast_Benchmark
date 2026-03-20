import os
import sys
import gc
import math
from typing import Any

import numpy as np
import torch

# Set environment variable for CUDA
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Toto path: set TOTO_PATH to your clone, or clone into envs/toto/toto
_script_dir = os.path.dirname(os.path.abspath(__file__))
_trafficfm_root = os.path.dirname(_script_dir)
_default_toto = os.path.join(_trafficfm_root, "envs", "toto", "toto")
toto_path = os.environ.get("TOTO_PATH", _default_toto)
if os.path.isdir(toto_path):
    if toto_path not in sys.path:
        sys.path.insert(0, os.path.realpath(toto_path))
else:
    raise FileNotFoundError(
        f"toto repository not found at {toto_path}. "
        "Clone it: git clone https://github.com/DataDog/toto.git "
        "or set TOTO_PATH to your clone path."
    )

from gluonts.dataset.split import split
from gluonts.time_feature import get_seasonality
#from inference.gluonts_predictor import Multivariate, TotoPredictor
from toto.inference.gluonts_predictor import Multivariate, TotoPredictor
from toto.model.toto import Toto

import argparse
from common import eval, eval_time
from config import device
from config import config as benchmark_config

DEFAULT_CONTEXT_LENGTH = 4096


def get_maximal_context_length(dataset):
    """
    Calculates the maximal context length that can be used for the given dataset,
    based on the shortest time series in the dataset and the number of
    prediction windows used for validation and testing.
    """
    shortest_series_in_dataset = dataset._min_series_length
    total_prediction_windows = dataset.windows + 1
    max_context_length = (
        shortest_series_in_dataset
        - total_prediction_windows * dataset.prediction_length
    )
    return max_context_length


def get_total_gpu_memory():
    """Get total GPU VRAM capacity in MB."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)


def calculate_optimal_batch_size(
    model,
    target_dim,
    prediction_length,
    context_length,
    use_kv_cache,
    num_samples,
    safety_factor=0.01,
):
    """
    Calculate the optimal batch size based on available GPU memory and model requirements.
    """
    try:
        model_width = model.model.embed_dim
        model_depth = model.model.num_layers

        model_param_memory_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        base_memory_per_sample = (model_width * model_depth * 4) / (1024 * 1024)
        io_memory = (target_dim * (context_length + prediction_length) * 4) / (
            1024 * 1024
        )

        kv_memory = 0
        if use_kv_cache:
            kv_memory = (model_depth * model_width * 2 * context_length * 4) / (
                1024 * 1024
            )

        mem_per_sample_mb = base_memory_per_sample + io_memory + kv_memory
        mem_per_batch_mb = mem_per_sample_mb * target_dim * num_samples

        gpu_mem = get_total_gpu_memory()
        cuda_reserved_mb = 1024

        available_memory = (
            gpu_mem - model_param_memory_mb - cuda_reserved_mb
        ) * safety_factor

        max_batch_size = max(
            1, int(available_memory / (mem_per_batch_mb / num_samples))
        )
        max_batch_size = min(16, max_batch_size)
        return max_batch_size
    except RuntimeError as e:
        print(f"Error calculating optimal batch size: {e}")
        return 1


class TOTOModelPredictorWrapper:
    """Wrapper for TOTOPredictor that handles OOM errors by adjusting batch size."""

    def __init__(
        self,
        model,
        prediction_length,
        context_length,
        mode,
        num_samples=128,
        use_kv_cache=True,
    ):
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.mode = mode
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.samples_per_batch = num_samples
        self.model = model
        self._adjusted = False

        self._initialize_predictor()

    def _initialize_predictor(self):
        """Initialize the TOTOPredictor with the current samples_per_batch."""
        self.predictor = TotoPredictor.create_for_eval(
            model=self.model,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            mode=self.mode,
            samples_per_batch=self.samples_per_batch,
        )

    def predict(self, gluonts_test_data: tuple):
        """Perform prediction while adjusting samples_per_batch if OOM errors occur."""
        if not self._adjusted:
            print(
                "Initializing predictor with samples_per_batch =",
                self.samples_per_batch,
            )
            while self.samples_per_batch >= 1:
                try:
                    print(
                        f"Attempting prediction with samples_per_batch = {self.samples_per_batch} and context_length = {self.context_length}"
                    )
                    predictions = list(
                        self.predictor.predict(
                            gluonts_test_data,
                            use_kv_cache=self.use_kv_cache,
                            num_samples=self.num_samples,
                        )
                    )
                    self._adjusted = True
                    return predictions
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        if self.samples_per_batch > 1:
                            print(
                                f"Out of memory with samples_per_batch = {self.samples_per_batch}. Reducing batch size."
                            )
                            self.samples_per_batch = self.samples_per_batch // 2
                            torch.cuda.empty_cache()
                        else:
                            print(
                                f"OOM at minimal batch size. Cannot proceed with this context length and sample count."
                            )
                            raise e
                        self._initialize_predictor()
                    else:
                        raise e

        return self.predictor.predict(
            gluonts_test_data,
            use_kv_cache=self.use_kv_cache,
            num_samples=self.num_samples,
        )


class TotoPredictorWrapper:
    """
    Wrapper class for Toto model to work with common.eval() function.
    """

    def __init__(
        self,
        model,
        prediction_length: int,
        context_length: int = None,
        num_samples: int = 256,
        use_kv_cache: bool = True,
        pad_short_series: bool = False,
    ):
        print("prediction_length:", prediction_length)
        self.model = model
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.pad_short_series = pad_short_series
        self.context_length = context_length or DEFAULT_CONTEXT_LENGTH
        self._current_dataset = None

    def _get_context_length(self, dataset):
        """Calculate context length for the current dataset."""
        if not self.pad_short_series:
            max_context = get_maximal_context_length(dataset)
            context_length = min(DEFAULT_CONTEXT_LENGTH, max_context)
        else:
            context_length = DEFAULT_CONTEXT_LENGTH
        return max(context_length, self.model.model.patch_embed.stride)

    def predict(self, test_data_input):
        """
        Predict method that creates a TOTOModelPredictorWrapper for each dataset.
        """
        # Get context length for current dataset
        context_length = self.context_length
        if self._current_dataset is not None:
            context_length = self._get_context_length(self._current_dataset)

        # Calculate optimal batch size
        if self._current_dataset is not None:
            suggested_batch_size = calculate_optimal_batch_size(
                model=self.model,
                target_dim=self._current_dataset.target_dim,
                prediction_length=self.prediction_length,
                context_length=context_length,
                use_kv_cache=self.use_kv_cache,
                num_samples=self.num_samples,
            )
        else:
            suggested_batch_size = 16

        # Create predictor wrapper
        predictor_wrapper = TOTOModelPredictorWrapper(
            model=self.model,
            prediction_length=self.prediction_length,
            context_length=context_length,
            mode=Multivariate(batch_size=suggested_batch_size),
            num_samples=self.num_samples,
            use_kv_cache=self.use_kv_cache,
        )

        return predictor_wrapper.predict(test_data_input)


def main():
    model_name = "Toto-Open-Base-1.0"
    model_path = "Datadog/Toto-Open-Base-1.0"
    num_samples = 256
    use_kv_cache = True
    pad_short_series = False

    # Load model once
    print("Loading Toto model...")
    try:
        model = Toto.from_pretrained(model_path, cache_dir=benchmark_config.hf_home)
    except TypeError:
        model = Toto.from_pretrained(model_path)
    model = model.to(device if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model = torch.compile(model)

    def predictor_factory(dataset):
        # Store current dataset for context length calculation
        predictor = TotoPredictorWrapper(
            model=model,
            prediction_length=dataset.prediction_length,
            num_samples=num_samples,
            use_kv_cache=use_kv_cache,
            pad_short_series=pad_short_series,
        )
        predictor._current_dataset = dataset
        return predictor

    parser = argparse.ArgumentParser(description="Toto evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    try:
        if args.eval_time:
            eval_time(model_name, model_path, predictor_factory)
        else:
            eval(model_name, model_path, predictor_factory, batch_size=num_samples)
    finally:
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

