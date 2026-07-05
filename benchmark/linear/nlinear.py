"""
NLinear (Normalization-Linear) for traffic forecasting using gift_eval data format.
Trains on sd_train/{year}/15T (or any dataset), validates on sd_val/{year}/15T,
evaluates via gift_eval standard run_benchmark() — same pipeline as SSM models.

Run from benchmark/:
  uv run --project fm/moirai python -m linear.nlinear --year 2018
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import wandb
from gluonts.itertools import batcher
from gluonts.model import Forecast
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fm.fm_utils import get_entry_target, run_benchmark, to_sample_forecasts

TERM_TO_PRED_LEN = {"short": 3, "medium": 6, "long": 12}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NLinearModel(nn.Module):
    """Normalization-Linear: subtract last value before linear projection, add back after."""

    def __init__(self, context_length: int, prediction_length: int):
        super().__init__()
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.linear = nn.Linear(context_length, prediction_length)

    def forward(self, x):
        # x: (B, L, 1)
        seq_last = x[:, -1:, 0].detach()        # (B, 1)
        x = x[:, :, 0] - seq_last               # (B, L)
        out = self.linear(x)                     # (B, pred_len)
        return out + seq_last                    # (B, pred_len)

    def param_num(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_windows(entries, context_len: int, pred_len: int,
                 windows_per_sensor: int, num_sensors: int):
    X, Y = [], []
    if num_sensors > 0:
        entries = entries[:num_sensors]
    for entry in entries:
        target = np.asarray(entry["target"], dtype=np.float32)
        T = len(target)
        span = context_len + pred_len
        if T < span:
            continue
        max_start = T - span
        if windows_per_sensor > 0 and max_start >= windows_per_sensor:
            indices = np.linspace(0, max_start, windows_per_sensor, dtype=int)
        else:
            indices = np.arange(0, max_start + 1)
        for s in indices:
            ctx = target[s : s + context_len]
            fut = target[s + context_len : s + span]
            if ctx.std() < 1.0:
                continue
            m, std = ctx.mean(), ctx.std() + 1e-8
            X.append((ctx - m) / std)
            Y.append((fut - m) / std)
    X = np.stack(X).astype(np.float32)[:, :, np.newaxis]
    Y = np.stack(Y).astype(np.float32)
    return X, Y


def train_one_term(args, term, device, train_entries, val_entries):
    pred_len  = TERM_TO_PRED_LEN[term]
    ckpt_path = os.path.join(args.log_dir, f"best_model_{term}_s{args.seed}.pt")

    if os.path.exists(ckpt_path) and not args.force_retrain:
        print(f"\n[{term}] Checkpoint found, skipping: {ckpt_path}")
        return ckpt_path

    X_tr, Y_tr   = make_windows(train_entries, args.context_length, pred_len,
                                 args.windows_per_sensor, args.num_sensors)
    X_val, Y_val = make_windows(val_entries,   args.context_length, pred_len,
                                 args.windows_per_sensor, args.num_sensors)
    print(f"\n[{term}] Train: {X_tr.shape[0]} | Val: {X_val.shape[0]}")
    wandb.log({f"{term}/train_windows": X_tr.shape[0], f"{term}/val_windows": X_val.shape[0]})

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
                              batch_size=args.bs, shuffle=True, num_workers=2)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
                              batch_size=args.bs, shuffle=False, num_workers=2)

    model = NLinearModel(
        context_length=args.context_length,
        prediction_length=pred_len,
    ).to(device)
    print(f"  params: {model.param_num():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    best_val, wait = np.inf, 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = (model(x_b) - y_b).abs().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_losses.append((model(x_b) - y_b).abs().mean().item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        print(f"  Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f}")
        wandb.log({"epoch": epoch, f"{term}/train_loss": train_loss,
                   f"{term}/val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"  Best val={best_val:.4f}, saved to {ckpt_path}")
    return ckpt_path


class NLinearPredictor:
    def __init__(self, checkpoints: dict, args, device):
        self.checkpoints = checkpoints
        self.args = args
        self.device = device
        self._models = {}
        self._current_pred_len = 3

    def _get_model(self, prediction_length: int):
        if prediction_length not in self._models:
            model = NLinearModel(
                context_length=self.args.context_length,
                prediction_length=prediction_length,
            ).to(self.device)
            for term, plen in TERM_TO_PRED_LEN.items():
                if plen == prediction_length and term in self.checkpoints:
                    model.load_state_dict(torch.load(self.checkpoints[term],
                                                     map_location=self.device))
                    break
            model.eval()
            self._models[prediction_length] = model
        return self._models[prediction_length]

    def predict(self, test_data_input, batch_size: int = 256) -> List[Forecast]:
        forecast_outputs = []
        test_data_input  = list(test_data_input)
        model = self._get_model(self._current_pred_len)

        with torch.no_grad():
            for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
                contexts, means, stds = [], [], []
                for entry in batch:
                    target = np.array(get_entry_target(entry), dtype=np.float32)
                    ctx = target[-self.args.context_length:]
                    if len(ctx) < self.args.context_length:
                        ctx = np.pad(ctx, (self.args.context_length - len(ctx), 0))
                    m, std = ctx.mean(), ctx.std() + 1e-8
                    means.append(m)
                    stds.append(std)
                    contexts.append((ctx - m) / std)

                x = torch.tensor(np.stack(contexts)[:, :, np.newaxis],
                                 dtype=torch.float32).to(self.device)
                pred_norm = model(x).cpu().numpy()
                for pn, m, s in zip(pred_norm, means, stds):
                    forecast_outputs.append((pn * s + m)[np.newaxis, :])

        return to_sample_forecasts(forecast_outputs, test_data_input)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",            type=str,   default="sd")
    parser.add_argument("--year",               type=str,   default="2019")
    parser.add_argument("--context_length",     type=int,   default=96)
    parser.add_argument("--num_sensors",        type=int,   default=0)
    parser.add_argument("--windows_per_sensor", type=int,   default=50)
    parser.add_argument("--bs",                 type=int,   default=256)
    parser.add_argument("--lrate",              type=float, default=1e-3)
    parser.add_argument("--max_epochs",         type=int,   default=50)
    parser.add_argument("--patience",           type=int,   default=15)
    parser.add_argument("--seed",               type=int,   default=2023)
    parser.add_argument("--device",             type=str,   default="cuda")
    parser.add_argument("--log_dir",            type=str,
                        default="/u/tliao2/TrafficFM/benchmark/experiments/linear/nlinear/SD/2019/")
    parser.add_argument("--wandb_project",      type=str,   default="TrafficFM")
    parser.add_argument("--force_retrain",      action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    print("Loading gift_eval data …")
    from config import config
    from gift_eval.data import Dataset
    os.environ["GIFT_EVAL"] = config.gift_eval_datasets_path

    train_entries = list(Dataset(name=f"{args.dataset}_train/{args.year}/15T",
                                 term="short").gluonts_dataset)
    val_entries   = list(Dataset(name=f"{args.dataset}_val/{args.year}/15T",
                                 term="short").gluonts_dataset)

    dataset_tag = args.dataset.upper()
    model_name  = f"nlinear_{dataset_tag}{args.year}_ctx{args.context_length}_w{args.windows_per_sensor}"
    wandb.init(
        project=args.wandb_project,
        name=f"{model_name}_s{args.seed}",
        config=vars(args),
    )

    checkpoints = {}
    for term in ["short", "medium", "long"]:
        checkpoints[term] = train_one_term(args, term, device, train_entries, val_entries)

    print("\nEvaluating via gift_eval …")
    predictor = NLinearPredictor(checkpoints=checkpoints, args=args, device=device)

    def predictor_factory(dataset):
        predictor._current_pred_len = dataset.prediction_length
        return predictor

    test_dataset = f"{args.dataset}/{args.year}/15T"
    config.short_datasets    = test_dataset
    config.med_long_datasets = test_dataset

    run_benchmark(
        eval_time_only=False,
        model_name=model_name,
        model_path=args.log_dir,
        predictor_factory=predictor_factory,
        batch_size=args.bs,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
