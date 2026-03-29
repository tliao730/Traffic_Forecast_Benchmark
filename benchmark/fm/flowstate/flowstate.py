import argparse
import os
import random
import warnings
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv

from load_model import setup_model_environment

# Load environment variables
load_dotenv()

MODEL_ENV = setup_model_environment("flowstate", __file__)

from common import eval, eval_time
from config import device

warnings.filterwarnings("ignore")

MODEL_NAME = "FlowState-9.1M"
MODEL_PATH = "ibm-research/FlowState"
DEFAULT_SEED = 0
DEFAULT_BATCH_SIZE = 8

from tsfm_public import FlowStateForPrediction  # noqa: E402

# `gift_wrapper.py` lives under `notebooks/` and is not an importable Python package,
# so we load it directly by file path (no sys.path tricks).
_gift_wrapper_mod = MODEL_ENV.load_repo_module(
    "flowstate_gift_wrapper",
    "notebooks",
    "hfdemo",
    "flowstate",
    "gift_wrapper.py",
)
FlowState_Gift_Wrapper = _gift_wrapper_mod.FlowState_Gift_Wrapper


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowState evaluation or time estimation")
    parser.add_argument(
        "--eval-time",
        action="store_true",
        help="Run eval_time (time estimation) only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("FLOWSTATE_EVAL_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        help=(
            "Evaluation batch size "
            f"(default: FLOWSTATE_EVAL_BATCH_SIZE or {DEFAULT_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_flowstate(
    pred_length: int,
    n_ch: int,
    freq: str,
    device_str: str = "cpu",
    domain: Optional[str] = None,
    nd: bool = False,
    batch_size: int = 16,
):
    """
    Construct the FlowState_Gift_Wrapper predictor for a given dataset.
    This is the same logic as the old script's LoadFlowState function,
    just with slightly cleaned argument names.
    """
    model_name = "ibm-research/FlowState"
    flowstate = FlowStateForPrediction.from_pretrained(model_name).to(device_str)

    config = flowstate.config
    config.min_context = 0
    config.device = device_str
    # Reduce context length to lower memory usage in SSM FFT kernels.
    # This value is consumed by FlowState_Gift_Wrapper via cfg["context_length"].
    context_length = os.environ.get("FLOWSTATE_CONTEXT_LENGTH")
    if context_length:
        config.context_length = int(context_length)
    flowstate = FlowState_Gift_Wrapper(
        flowstate,
        pred_length,
        n_ch=n_ch,
        batch_size=batch_size,
        f=freq,
        device=device_str,
        domain=domain,
        no_daily=nd,
    )
    return flowstate


def main():
    args = _build_parser().parse_args()

    def predictor_factory(dataset):
        set_seed(args.seed)

        # Determine number of channels from one test sample
        num_channels = 1
        for x in dataset.test_data:
            # Dataset entries may be (data, label) tuples or dicts
            if isinstance(x, tuple):
                target = x[0]["target"]
            else:
                target = x["target"]

            target_arr = np.asarray(target)
            if target_arr.ndim == 1:
                num_channels = 1
            else:
                # shape: (C, T)
                num_channels = target_arr.shape[0]
            break

        # Special handling for bizitobs_l2c datasets (same as old script)
        no_daily = "l2c" in dataset.name

        predictor = load_flowstate(
            pred_length=dataset.prediction_length,
            n_ch=num_channels,
            freq=dataset.freq,
            device_str=device,
            domain=getattr(dataset, "domain", None),
            nd=no_daily,
            batch_size=args.batch_size,
        )
        return predictor

    if args.eval_time:
        eval_time(MODEL_NAME, MODEL_PATH, predictor_factory)
    else:
        eval(MODEL_NAME, MODEL_PATH, predictor_factory, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
