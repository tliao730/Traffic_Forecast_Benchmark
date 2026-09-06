"""
S4 (diagonal state space model) for traffic forecasting using gift_eval data format.
Trains on sd_train/{year}/15T, validates on sd_val/{year}/15T,
evaluates via gift_eval standard run_benchmark() — same pipeline as mamba.py.

Run from benchmark/:
  uv run --project fm/moirai python ssm/s4.py --year 2018
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
from ssm.s4_model import S4ForecastModel
from ssm.resume_utils import peek_resume, restore_resume, save_resume

TERM_TO_PRED_LEN = {"short": 3, "medium": 6, "long": 12}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    resume_state = None if args.force_retrain else peek_resume(args, term)
    if resume_state is not None and resume_state["finished"]:
        print(f"\n[{term}] Training already finished, skipping: {ckpt_path}")
        return ckpt_path
    if resume_state is None and os.path.exists(ckpt_path) and not args.force_retrain:
        # run completed before resume support existed (no resume checkpoint)
        print(f"\n[{term}] Checkpoint found, skipping: {ckpt_path}")
        return ckpt_path

    X_tr, Y_tr   = make_windows(train_entries, args.context_length, pred_len,
                                 args.windows_per_sensor, args.num_sensors)
    X_val, Y_val = make_windows(val_entries,   args.context_length, pred_len,
                                 args.windows_per_sensor, args.num_sensors)
    print(f"\n[{term}] Train: {X_tr.shape[0]} | Val: {X_val.shape[0]}")
    wandb.log({f"{term}/train_windows": X_tr.shape[0], f"{term}/val_windows": X_val.shape[0]})

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr)),
                              batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
                              batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    model = S4ForecastModel(
        prediction_length=pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        dt=args.dt,
        num_layers=args.num_layers,
    ).to(device)
    print(f"  params: {model.param_num():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    best_val, wait = np.inf, 0
    start_epoch = 1
    if resume_state is not None:
        start_epoch, best_val, wait = restore_resume(resume_state, model, optimizer)

    for epoch in range(start_epoch, args.max_epochs + 1):
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

        finished = wait >= args.patience or epoch == args.max_epochs
        save_resume(args, term, model, optimizer, epoch, best_val, wait, finished)
        if wait >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"  Best val={best_val:.4f}, saved to {ckpt_path}")
    return ckpt_path


class S4Predictor:
    def __init__(self, checkpoints: dict, args, device):
        self.checkpoints = checkpoints
        self.args = args
        self.device = device
        self._models = {}
        self._current_pred_len = 3

    def _get_model(self, prediction_length: int):
        if prediction_length not in self._models:
            model = S4ForecastModel(
                prediction_length=prediction_length,
                d_model=self.args.d_model,
                d_state=self.args.d_state,
                dt=self.args.dt,
                num_layers=self.args.num_layers,
            ).to(self.device)
            for term, plen in TERM_TO_PRED_LEN.items():
                if plen == prediction_length and term in self.checkpoints:
                    sd = torch.load(self.checkpoints[term], map_location=self.device)
                    for i, block in enumerate(model.blocks):
                        key = f"blocks.{i}.s4.Lambda_re"
                        if key in sd:
                            r = sd[key].shape[1]
                            if r != block.s4.d_state:
                                from ssm.s4_model import S4Layer
                                block.s4 = S4Layer(
                                    self.args.d_model, r, self.args.dt
                                ).to(self.device)
                    model.load_state_dict(sd)
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
    parser.add_argument("--dataset",             type=str,   default="sd")
    parser.add_argument("--year",                type=str,   default="2019")
    parser.add_argument("--context_length",      type=int,   default=48)
    parser.add_argument("--num_sensors",         type=int,   default=0)
    parser.add_argument("--windows_per_sensor",  type=int,   default=50)
    parser.add_argument("--d_model",             type=int,   default=64)
    parser.add_argument("--d_state",             type=int,   default=64)
    parser.add_argument("--num_layers",          type=int,   default=2)
    parser.add_argument("--dt",                  type=float, default=0.01)
    parser.add_argument("--bs",                  type=int,   default=256)
    parser.add_argument("--num_workers",         type=int,   default=2,
                        help="DataLoader worker processes. Keep <= (--cpus-per-task - 1); "
                             "SLURM allocates 1 CPU per task by default, and 2 workers on "
                             "1 core starve the loader.")
    parser.add_argument("--lrate",               type=float, default=1e-3)
    parser.add_argument("--max_epochs",          type=int,   default=50)
    parser.add_argument("--patience",            type=int,   default=15)
    parser.add_argument("--seed",                type=int,   default=2023)
    parser.add_argument("--device",              type=str,   default="cuda")
    parser.add_argument("--log_dir",             type=str,
                        default="/u/tliao2/TrafficFM/benchmark/experiments/ssm_bench/s4/SD/2019/")
    parser.add_argument("--wandb_project",       type=str,   default="TrafficFM")
    parser.add_argument("--force_retrain",       action="store_true")
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

    model_name = f"s4_{args.dataset.upper()}{args.year}_ctx{args.context_length}_w{args.windows_per_sensor}"
    wandb.init(
        project=args.wandb_project,
        name=model_name,
        config=vars(args),
    )

    checkpoints = {}
    for term in ["short", "medium", "long"]:
        checkpoints[term] = train_one_term(args, term, device, train_entries, val_entries)

    print("\nEvaluating via gift_eval …")
    predictor = S4Predictor(checkpoints=checkpoints, args=args, device=device)

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
