import os
import gc

import torch
from fm.fm_utils import (
    build_basic_parser,
    load_pretrained_with_cache,
    run_benchmark,
)

MODEL_NAME = "Toto-Open-Base-1.0"
MODEL_PATH = "Datadog/Toto-Open-Base-1.0"
DEFAULT_NUM_SAMPLES = 256
DEFAULT_USE_KV_CACHE = True
DEFAULT_PAD_SHORT_SERIES = False

# Set environment variable for CUDA
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from load_model import setup_model_runtime

MODEL_RUNTIME = setup_model_runtime("toto", __file__)
benchmark_config = MODEL_RUNTIME.config
device = MODEL_RUNTIME.device

from toto.model.toto import Toto
from .toto_runtime import TotoPredictorWrapper

import argparse


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


def main():
    args = _build_parser().parse_args()
    use_kv_cache = DEFAULT_USE_KV_CACHE and not args.disable_kv_cache
    model = _load_toto_model(MODEL_PATH)

    def predictor_factory(dataset):
        return TotoPredictorWrapper(
            model=model,
            dataset=dataset,
            prediction_length=dataset.prediction_length,
            num_samples=args.num_samples,
            use_kv_cache=use_kv_cache,
            pad_short_series=args.pad_short_series,
        )

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
