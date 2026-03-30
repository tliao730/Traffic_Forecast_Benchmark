import argparse
from typing import Any, Callable, Optional


def build_basic_parser(description: str) -> argparse.ArgumentParser:
    """Create the shared FM CLI parser with the standard eval-time toggle."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--eval-time",
        action="store_true",
        help="Run eval_time (time estimation) only",
    )
    return parser


def load_pretrained_with_cache(
    loader_cls: Any,
    model_path: str,
    *args: Any,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Call ``from_pretrained`` with a cache dir when supported, and gracefully
    fall back for loaders that do not accept the ``cache_dir`` keyword.
    """
    if cache_dir is None:
        return loader_cls.from_pretrained(model_path, *args, **kwargs)

    try:
        return loader_cls.from_pretrained(
            model_path,
            *args,
            cache_dir=cache_dir,
            **kwargs,
        )
    except TypeError:
        return loader_cls.from_pretrained(model_path, *args, **kwargs)


def run_benchmark(
    *,
    eval_time_only: bool,
    model_name: str,
    model_path: str,
    predictor_factory: Callable[..., Any],
    batch_size: int = 1024,
    save_predictions_dir: Optional[str] = None,
) -> Any:
    """Dispatch to ``eval_time`` or ``eval`` using the shared FM entrypoint shape."""
    from common import eval, eval_time

    if eval_time_only:
        return eval_time(model_name, model_path, predictor_factory)

    return eval(
        model_name,
        model_path,
        predictor_factory,
        batch_size=batch_size,
        save_predictions_dir=save_predictions_dir,
    )
