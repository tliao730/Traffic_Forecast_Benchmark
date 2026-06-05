"""
Compare GNN dataloader test windows against gift_eval test windows.
Checks that Y values (ground truth) are identical across all terms.

Usage:
    cd /u/tliao2/TrafficFM
    uv run --project benchmark/fm/moirai python compare_test_windows.py
    uv run --project benchmark/fm/moirai python compare_test_windows.py --term medium
    uv run --project benchmark/fm/moirai python compare_test_windows.py --term long
"""
import os
import sys
import logging
import types
import argparse
import numpy as np

TERM_TO_HORIZON = {"short": 3, "medium": 6, "long": 12}
DATASET  = "SD"
YEAR     = "2018"
GNN_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmark/gnn"))
DATA_PATH = os.path.join(GNN_DIR, "data/sd")
GIFT_EVAL_PATH = os.path.abspath("dataset/LargeST/gift_eval")

parser = argparse.ArgumentParser()
parser.add_argument("--term", default="short", choices=["short", "medium", "long"])
cli = parser.parse_args()
term    = cli.term
horizon = TERM_TO_HORIZON[term]

# ── 1. GNN DataLoader ────────────────────────────────────────────────────────
sys.path.insert(0, GNN_DIR)
from src.utils.dataloader import load_dataset

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

args = types.SimpleNamespace(
    input_dim=3,
    seq_len=48,
    horizon=horizon,  # dataloader infers gift_eval term from this
    bs=64,
    dataset=DATASET,
    years=YEAR,
)

dataloader, scaler = load_dataset(DATA_PATH, args, logger)
loader = dataloader["test_loader"]

gnn_windows = []
for _, Y in loader.get_iterator():
    # Y shape: (batch, horizon, 716, 1)
    Y_denorm = Y[..., 0] * scaler.std.numpy() + scaler.mean.numpy()
    for i in range(len(Y_denorm)):
        gnn_windows.append(Y_denorm[i])  # (horizon, 716)

print(f"[GNN DataLoader] term={term}, windows={len(gnn_windows)}, Y shape={gnn_windows[0].shape}")

# ── 2. gift_eval test windows ────────────────────────────────────────────────
os.environ["GIFT_EVAL"] = GIFT_EVAL_PATH
from gift_eval.data import Dataset

ds    = Dataset(name=f"sd/{YEAR}/15T", term=term, to_univariate=False)
pairs = list(ds.test_data)

print(f"[gift_eval]      term={term}, windows={ds.windows}, pred_len={ds.prediction_length}")
print()

# ── 3. Compare Y[:horizon, sensor=0] for each window ────────────────────────
print(f"{'win':>4}  {'gift_eval Y sensor0':>35}  {'GNN Y sensor0':>35}  {'match':>5}")
print("-" * 90)

all_match = True
for i in range(ds.windows):
    _, lab_entry = pairs[i]
    lab_arr = np.array(lab_entry["target"])             # (pred_len,) or (pred_len, 716)
    ge_y = lab_arr[:horizon, 0] if lab_arr.ndim > 1 else lab_arr[:horizon]

    gnn_y = gnn_windows[i][:, 0]                       # (horizon,), sensor 0

    match = np.allclose(ge_y, gnn_y, atol=0.5)
    all_match = all_match and match

    print(f"{i:>4}  {str(ge_y.round(0)):>35}  {str(gnn_y.round(0)):>35}  {'✓' if match else '✗':>5}")

print()
print("All windows match!" if all_match else "Mismatch found — check above.")
