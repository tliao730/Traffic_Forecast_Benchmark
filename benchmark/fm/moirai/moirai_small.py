import argparse

from fm.fm_utils import build_basic_parser, load_pretrained_with_cache, run_benchmark
from load_model import setup_model_runtime
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

MODEL_NAME = "moirai_small"
MODEL_PATH = "Salesforce/moirai-1.0-R-small"
DEFAULT_CONTEXT_LENGTH = 4000
DEFAULT_PATCH_SIZE = 32
DEFAULT_NUM_SAMPLES = 20
DEFAULT_BATCH_SIZE = 64

MODEL_RUNTIME = setup_model_runtime("moirai_small", __file__)
benchmark_config = MODEL_RUNTIME.config


def _build_parser() -> argparse.ArgumentParser:
    parser = build_basic_parser("Moirai evaluation or time estimation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples for MoiraiForecast (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=f"Model context length (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size used by MoiraiForecast (default: {DEFAULT_PATCH_SIZE})",
    )
    return parser


def _load_moirai_module(model_path: str) -> MoiraiModule:
    return load_pretrained_with_cache(
        MoiraiModule,
        model_path,
        cache_dir=benchmark_config.hf_home,
    )


def main():
    args = _build_parser().parse_args()
    moirai_module = _load_moirai_module(MODEL_PATH)

    def predictor_factory(dataset):
        model = MoiraiForecast(
            module=moirai_module,
            prediction_length=dataset.prediction_length,
            context_length=args.context_length,
            patch_size=args.patch_size,
            num_samples=args.num_samples,
            target_dim=dataset.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
        )
        return model.create_predictor(batch_size=args.batch_size)

    run_benchmark(
        eval_time_only=args.eval_time,
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        predictor_factory=predictor_factory,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
