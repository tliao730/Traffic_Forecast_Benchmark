import os
import gc

import numpy as np
import torch
from fm.fm_utils import build_basic_parser, load_pretrained_with_cache, run_benchmark

MODEL_NAME = "Toto-Open-Base-1.0"
MODEL_PATH = "Datadog/Toto-Open-Base-1.0"
DEFAULT_NUM_SAMPLES = 256
DEFAULT_USE_KV_CACHE = True
DEFAULT_PAD_SHORT_SERIES = False

# Set environment variable for CUDA
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from load_model import setup_model_environment

setup_model_environment("toto", __file__)

from config import device
from config import config as benchmark_config

#from inference.gluonts_predictor import Multivariate, TotoPredictor
from toto.inference.gluonts_predictor import Multivariate, TotoPredictor
from toto.model.toto import Toto

import argparse

DEFAULT_CONTEXT_LENGTH = 4096


def _build_parser() -> argparse.ArgumentParser:
    parser = build_basic_parser("Toto evaluation or time estimation")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of generated samples (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--pad-short-series",
        action="store_true",
        default=DEFAULT_PAD_SHORT_SERIES,
        help="Pad short series instead of shrinking context length dynamically",
    )
    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="Disable KV cache during prediction",
    )
    return parser


def _load_toto_model(model_path: str):
    print("Loading Toto model...")
    model = load_pretrained_with_cache(
        Toto,
        model_path,
        cache_dir=benchmark_config.hf_home,
    )
    model = model.to(device if torch.cuda.is_available() else "cpu")
    model = model.eval()
    return torch.compile(model)


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
    args = _build_parser().parse_args()
    use_kv_cache = DEFAULT_USE_KV_CACHE and not args.disable_kv_cache
    model = _load_toto_model(MODEL_PATH)

    def predictor_factory(dataset):
        predictor = TotoPredictorWrapper(
            model=model,
            prediction_length=dataset.prediction_length,
            num_samples=args.num_samples,
            use_kv_cache=use_kv_cache,
            pad_short_series=args.pad_short_series,
        )
        predictor._current_dataset = dataset
        return predictor

    try:
        run_benchmark(
            eval_time_only=args.eval_time,
            model_name=MODEL_NAME,
            model_path=MODEL_PATH,
            predictor_factory=predictor_factory,
            batch_size=args.num_samples,
        )
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
