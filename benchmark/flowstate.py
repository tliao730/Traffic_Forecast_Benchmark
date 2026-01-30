import os
import random
import sys
import warnings
from typing import Optional

import numpy as np
import torch
from config import device
from dotenv import load_dotenv

from common import eval

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# FlowState path: set GRANITE_TSFM_PATH to your clone, or clone into envs/flowstate/granite-tsfm
_script_dir = os.path.dirname(os.path.abspath(__file__))
_trafficfm_root = os.path.dirname(_script_dir)
_default_granite = os.path.join(_trafficfm_root, "envs", "flowstate", "granite-tsfm")
_granite_path = os.environ.get("GRANITE_TSFM_PATH", _default_granite)
if os.path.isdir(_granite_path):
    sys.path.insert(0, os.path.realpath(_granite_path))
else:
    raise FileNotFoundError(
        f"granite-tsfm repo not found at {_granite_path}. "
        "Clone it: git clone https://github.com/ibm-granite/granite-tsfm.git "
        "or set GRANITE_TSFM_PATH to your clone path."
    )

from tsfm_public import FlowStateForPrediction  # noqa: E402
from notebooks.hfdemo.flowstate.gift_wrapper import FlowState_Gift_Wrapper  # noqa: E402


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
    model_name = "FlowState-9.1M"
    model_path = "ibm-research/FlowState"
    seed = 0
    batch_size = 16

    def predictor_factory(dataset):
        """
        Given a Dataset object from common.eval, construct the FlowState predictor.
        """
        set_seed(seed)

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
            batch_size=batch_size,
        )
        return predictor

    # Delegate benchmark loop and metrics/CSV handling to common.eval
    eval(model_name, model_path, predictor_factory, batch_size=batch_size)


if __name__ == "__main__":
    main()

