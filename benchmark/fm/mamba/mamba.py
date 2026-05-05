"""
Mamba for traffic forecasting using gift_eval data format.
Trains on sd_train/2019/15T, validates on sd_val/2019/15T,
evaluates on sd/2019/15T — same split as FM models.

Run from benchmark/:
  uv run --project fm/moirai python fm/mamba/mamba.py --term short
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from fm.mamba.mamba_model import MambaForecastModel

GIFT_EVAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "dataset", "LargeST", "gift_eval"
)
TERM_TO_PRED_LEN = {"short": 3, "medium": 6, "long": 12}


# ── helpers ────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_gift_eval_entries(dataset_name: str) -> list:
    os.environ["GIFT_EVAL"] = os.path.abspath(GIFT_EVAL_PATH)
    from gift_eval.data import Dataset
    ds = Dataset(name=dataset_name, term="short", to_univariate=False)
    return list(ds.gluonts_dataset)


def make_windows(entries, context_len: int, pred_len: int,
                 windows_per_sensor: int, num_sensors: int):
    """
    Slide (context_len + pred_len) windows over each sensor's series.
    Returns normalized (X, Y) numpy arrays and per-window (mean, std) for denorm.
    """
    X, Y, means, stds = [], [], [], []
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
            std_raw = ctx.std()
            if std_raw < 1.0:  # skip near-zero / constant traffic windows
                continue
            m, std = ctx.mean(), std_raw + 1e-8
            X.append((ctx - m) / std)
            Y.append((fut - m) / std)
            means.append(m)
            stds.append(std)

    X = np.stack(X).astype(np.float32)[:, :, np.newaxis]  # (N, T, 1)
    Y = np.stack(Y).astype(np.float32)                    # (N, H)
    return X, Y, np.array(means), np.array(stds)


def masked_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (pred - target).abs()[mask].mean()


# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate_on_test(model, entries, context_len, pred_len,
                     windows_per_sensor, num_sensors, device):
    """Compute MAE / RMSE / MAPE on original scale."""
    model.eval()
    maes, rmses, mapes = [], [], []

    if num_sensors > 0:
        entries = entries[:num_sensors]

    with torch.no_grad():
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
                m, std = ctx.mean(), ctx.std() + 1e-8

                x_norm = torch.tensor((ctx - m) / std).float()
                x_norm = x_norm.unsqueeze(0).unsqueeze(-1).to(device)  # (1, T, 1)
                pred_norm = model(x_norm).cpu().numpy()[0]
                pred_orig = pred_norm * std + m

                mask = fut != 0
                if mask.sum() == 0:
                    continue
                err = np.abs(pred_orig[mask] - fut[mask])
                maes.append(err.mean())
                rmses.append((err ** 2).mean())
                mapes.append((err / (np.abs(fut[mask]) + 1e-8)).mean())

    mae  = float(np.mean(maes))
    rmse = float(np.sqrt(np.mean(rmses)))
    mape = float(np.mean(mapes))
    return mae, rmse, mape


# ── main ───────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--term",            type=str,   default="short",
                        choices=["short", "medium", "long"])
    parser.add_argument("--context_length",  type=int,   default=48,
                        help="context steps (default 48 = 12 h at 15T)")
    parser.add_argument("--num_sensors",     type=int,   default=0,
                        help="0 = all 716 sensors")
    parser.add_argument("--windows_per_sensor", type=int, default=50)
    parser.add_argument("--d_model",         type=int,   default=64)
    parser.add_argument("--d_state",         type=int,   default=16)
    parser.add_argument("--d_conv",          type=int,   default=4)
    parser.add_argument("--expand",          type=int,   default=2)
    parser.add_argument("--num_layers",      type=int,   default=2)
    parser.add_argument("--bs",              type=int,   default=256)
    parser.add_argument("--lrate",           type=float, default=1e-3)
    parser.add_argument("--max_epochs",      type=int,   default=50)
    parser.add_argument("--patience",        type=int,   default=15)
    parser.add_argument("--seed",            type=int,   default=2023)
    parser.add_argument("--device",          type=str,   default="cuda")
    parser.add_argument("--log_dir",         type=str,
                        default="/scratch/bcqc/tliao2/TrafficFM/experiments/mamba_fm/SD/2019/")
    parser.add_argument("--wandb_project",   type=str,   default="TrafficFM")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pred_len = TERM_TO_PRED_LEN[args.term]
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"term={args.term}, context={args.context_length}, pred={pred_len}")
    print(f"num_sensors={args.num_sensors or 'all'}, windows_per_sensor={args.windows_per_sensor}")

    # ── load data ──────────────────────────────────────────────────────
    print("Loading gift_eval data …")
    train_entries = load_gift_eval_entries("sd_train/2019/15T")
    val_entries   = load_gift_eval_entries("sd_val/2019/15T")
    test_entries  = load_gift_eval_entries("sd/2019/15T")

    X_tr, Y_tr, _, _ = make_windows(train_entries, args.context_length, pred_len,
                                    args.windows_per_sensor, args.num_sensors)
    X_val, Y_val, _, _ = make_windows(val_entries, args.context_length, pred_len,
                                      args.windows_per_sensor, args.num_sensors)

    print(f"Train: {X_tr.shape[0]} samples | Val: {X_val.shape[0]} samples")
    print(f"X shape: {X_tr.shape}  (N, context_length, 1)")
    print(f"Y shape: {Y_tr.shape}  (N, pred_len)")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2)

    # ── model ──────────────────────────────────────────────────────────
    model = MambaForecastModel(
        prediction_length=pred_len,
        d_model=args.d_model, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand,
        num_layers=args.num_layers,
    ).to(device)
    print(f"Parameters: {model.param_num():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

    # ── wandb ──────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=f"mamba_fm_SD_2019_{args.term}_s{args.seed}",
        config=vars(args),
    )

    # ── training ───────────────────────────────────────────────────────
    best_val, wait = np.inf, 0
    ckpt_path = os.path.join(args.log_dir, f"best_model_{args.term}_s{args.seed}.pt")

    for epoch in range(1, args.max_epochs + 1):
        # train
        model.train()
        train_losses = []
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = (pred - y_b).abs().mean()  # plain MAE on normalized data
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                val_losses.append((model(x_b) - y_b).abs().mean().item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ val_loss improved to {best_val:.4f}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    # ── test evaluation ─────────────────────────────────────────────────
    print("\nEvaluating on test set …")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    mae, rmse, mape = evaluate_on_test(
        model, test_entries, args.context_length, pred_len,
        args.windows_per_sensor, args.num_sensors, device
    )
    print(f"\n[{args.term.upper()}] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.4f}")
    wandb.log({"test/mae": mae, "test/rmse": rmse, "test/mape": mape})
    wandb.finish()


if __name__ == "__main__":
    main()
