import argparse
import os
from types import ModuleType
from typing import Any, Callable, Iterable, Optional

import numpy as np


def build_basic_parser(description: str) -> argparse.ArgumentParser:
    """Create the shared FM CLI parser: the eval-time toggle and the dump flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--eval-time",
        action="store_true",
        help="Run eval_time (time estimation) only",
    )
    add_prediction_dump_args(parser)
    return parser


def add_prediction_dump_args(parser: argparse.ArgumentParser, *, dashed: bool = True) -> None:
    """
    Register the ``--save-predictions`` / ``--pred-out-dir`` pair.

    The FM, ML and simple entrypoints spell flags with dashes, the ssm/linear
    trainers with underscores; ``dashed`` picks the spelling while the dest
    stays the same, so ``resolve_prediction_dir`` works for both. (The GNN
    entrypoints have their own ``--save_predictions`` in gnn/src/utils/args.py.)
    """
    save_flag = "--save-predictions" if dashed else "--save_predictions"
    dir_flag = "--pred-out-dir" if dashed else "--pred_out_dir"
    parser.add_argument(
        save_flag,
        dest="save_predictions",
        action="store_true",
        help="Also write per-timestamp predictions and ground truth to CSV",
    )
    parser.add_argument(
        dir_flag,
        dest="pred_out_dir",
        type=str,
        default=None,
        help="Directory for prediction CSVs (default: result_root/<model>/predictions)",
    )


def resolve_prediction_dir(args: Any, model_name: str) -> Optional[str]:
    """Return the prediction-dump directory, or None when the flag is unset."""
    if not getattr(args, "save_predictions", False):
        return None
    out_dir = getattr(args, "pred_out_dir", None)
    if out_dir:
        return out_dir
    config = load_benchmark_config_module().config
    # Same "<model>_prof{N}" directory eval() writes the metrics to, or a
    # profiling dump would silently overwrite the benchmark dump of that model.
    return os.path.join(
        config.result_root, config.result_model_name(model_name), "predictions"
    )


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


def load_benchmark_config_module() -> ModuleType:
    """
    Load ``.env`` first, then import the shared benchmark config module.

    This keeps ``BENCHMARK_CONFIG`` selection and Hugging Face cache env vars
    consistent across FM entrypoints.
    """
    from dotenv import load_dotenv

    load_dotenv()

    import config as benchmark_config_module

    return benchmark_config_module


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


def unwrap_entry(entry: Any) -> dict[str, Any]:
    """Normalize dataset entries that may be either dicts or (dict, label) tuples."""
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def get_entry_target(entry: Any) -> Any:
    """Return the target array/sequence from a normalized dataset entry."""
    return unwrap_entry(entry)["target"]


def get_forecast_start_date(entry: Any) -> Any:
    """Compute the forecast start date for a dataset entry."""
    entry_dict = unwrap_entry(entry)
    return entry_dict["start"] + len(entry_dict["target"])


def infer_num_channels(entries: Iterable[Any], default: int = 1) -> int:
    """Infer channel count from the first available dataset entry."""
    for entry in entries:
        target_arr = np.asarray(get_entry_target(entry))
        return default if target_arr.ndim == 1 else int(target_arr.shape[0])
    return default


def to_sample_forecasts(forecast_outputs: Any, test_data_input: Iterable[Any]) -> list[Any]:
    """Convert model sample outputs into GluonTS SampleForecast objects."""
    from gluonts.model.forecast import SampleForecast

    forecasts = []
    for item, entry in zip(forecast_outputs, test_data_input):
        forecasts.append(
            SampleForecast(
                samples=item,
                start_date=get_forecast_start_date(entry),
            )
        )
    return forecasts


def to_quantile_forecasts(
    forecast_outputs: Any,
    test_data_input: Iterable[Any],
    forecast_keys: Iterable[Any],
    *,
    include_item_id: bool = False,
    default_item_id: str = "unknown",
) -> list[Any]:
    """Convert quantile arrays into GluonTS QuantileForecast objects."""
    from gluonts.model.forecast import QuantileForecast

    forecasts = []
    forecast_keys = list(map(str, forecast_keys))
    for item, entry in zip(forecast_outputs, test_data_input):
        entry_dict = unwrap_entry(entry)
        forecast_kwargs = {
            "forecast_arrays": item,
            "forecast_keys": forecast_keys,
            "start_date": get_forecast_start_date(entry_dict),
        }
        if include_item_id:
            forecast_kwargs["item_id"] = entry_dict.get("item_id", default_item_id)
        forecasts.append(QuantileForecast(**forecast_kwargs))
    return forecasts
