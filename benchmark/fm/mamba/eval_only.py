"""
Load a saved checkpoint and run test evaluation only.
Use this when training was interrupted after saving a checkpoint.

Run from benchmark/:
  uv run --project fm/moirai python fm/mamba/eval_only.py --term long --year 2018
"""

import argparse
import os
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from fm.mamba.mamba_model import MambaForecastModel
from fm.mamba.mamba import load_gift_eval_entries, evaluate_on_test, set_seed

GIFT_EVAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "dataset", "LargeST", "gift_eval"
)
TERM_TO_PRED_LEN = {"short": 3, "medium": 6, "long": 12}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--term",           type=str, default="long",
                        choices=["short", "medium", "long"])
    parser.add_argument("--year",           type=str, default="2018")
    parser.add_argument("--context_length", type=int, default=48)
    parser.add_argument("--num_sensors",    type=int, default=0)
    parser.add_argument("--windows_per_sensor", type=int, default=50)
    parser.add_argument("--d_model",        type=int, default=64)
    parser.add_argument("--d_state",        type=int, default=16)
    parser.add_argument("--d_conv",         type=int, default=4)
    parser.add_argument("--expand",         type=int, default=2)
    parser.add_argument("--num_layers",     type=int, default=2)
    parser.add_argument("--seed",           type=int, default=2023)
    parser.add_argument("--device",        type=str, default="cuda")
    parser.add_argument("--log_dir",       type=str,
                        default="/u/tliao2/TrafficFM/benchmark/experiments/mamba_fm/SD/2018/")
    parser.add_argument("--wandb_project", type=str, default="TrafficFM")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pred_len = TERM_TO_PRED_LEN[args.term]

    print(f"term={args.term}, context={args.context_length}, pred={pred_len}, year={args.year}")

    print("Loading test data ...")
    os.environ["GIFT_EVAL"] = os.path.abspath(GIFT_EVAL_PATH)
    test_entries = load_gift_eval_entries(f"sd/{args.year}/15T")

    model = MambaForecastModel(
        prediction_length=pred_len,
        d_model=args.d_model, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand,
        num_layers=args.num_layers,
    ).to(device)

    ckpt_path = os.path.join(args.log_dir, f"best_model_{args.term}_s{args.seed}.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    wandb.init(
        project=args.wandb_project,
        name=f"mamba_fm_SD_{args.year}_{args.term}_s{args.seed}_eval",
        config=vars(args),
    )

    mae, rmse, mape = evaluate_on_test(
        model, test_entries, args.context_length, pred_len,
        args.windows_per_sensor, args.num_sensors, device
    )
    print(f"\n[{args.term.upper()}] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.4f}")
    wandb.log({"test/mae": mae, "test/rmse": rmse, "test/mape": mape})
    wandb.finish()


if __name__ == "__main__":
    main()
