import argparse
from dotenv import load_dotenv

from common import eval, eval_time
from config import config as benchmark_config
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

MODEL_NAME = "moirai_small"
MODEL_PATH = "Salesforce/moirai-1.0-R-small"
DEFAULT_CONTEXT_LENGTH = 4000
DEFAULT_PATCH_SIZE = 32
DEFAULT_NUM_SAMPLES = 20
DEFAULT_BATCH_SIZE = 64

# Load environment variables (for model caches, etc.)
load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Moirai evaluation or time estimation")
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
    try:
        return MoiraiModule.from_pretrained(
            model_path,
            cache_dir=benchmark_config.hf_home,
        )
    except TypeError:
        # Some implementations do not accept cache_dir.
        return MoiraiModule.from_pretrained(model_path)


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

    if args.eval_time:
        eval_time(MODEL_NAME, MODEL_PATH, predictor_factory)
    else:
        eval(MODEL_NAME, MODEL_PATH, predictor_factory, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

