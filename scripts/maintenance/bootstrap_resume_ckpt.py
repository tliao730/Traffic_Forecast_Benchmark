"""Bootstrap a last_checkpoint_s{seed}.pt from a run started with the old
(non-resumable) code, so the new segmented-training code can continue it.

Usage (AFTER scancel-ing the running job):
    python scripts/bootstrap_resume_ckpt.py <log_dir> [--seed 2023]

<log_dir> is the year-suffixed experiment dir, e.g.
    benchmark/gnn/experiments/STTN/GLA/2018

The script:
  1. picks the newest final_model_s{seed}.pt out of <log_dir> and its parent
     (old runs saved to the year-less parent dir) and moves it into <log_dir>
  2. parses record_s{seed}.log for the last completed epoch, best valid loss,
     and the early-stop wait counter
  3. writes last_checkpoint_s{seed}.pt (optimizer state is unavailable and
     left as None; Adam moments restart on resume)
"""
import argparse
import os
import re
import shutil
import sys

import torch


def parse_log(log_path):
    """Return (last_epoch, min_loss, wait) from the newest run in the log."""
    lines = open(log_path).read().splitlines()
    # a log file can hold several runs; only look at the newest one
    starts = [i for i, ln in enumerate(lines) if "Namespace(" in ln]
    if starts:
        lines = lines[starts[-1]:]

    epoch_re = re.compile(r"Epoch: (\d+),.*Valid Loss: ([0-9.]+)")
    last_epoch, min_loss, last_improve = 0, float("inf"), 0
    for ln in lines:
        m = epoch_re.search(ln)
        if m:
            last_epoch = int(m.group(1))
            vloss = float(m.group(2))
            if vloss < min_loss:
                min_loss = vloss
                last_improve = last_epoch
    if last_epoch == 0:
        sys.exit(f"No completed epoch found in {log_path}")
    return last_epoch, min_loss, last_epoch - last_improve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="year-suffixed experiment dir")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--out", default=None,
                        help="override checkpoint output path (for testing)")
    args = parser.parse_args()

    log_dir = args.log_dir.rstrip("/")
    model_name = f"final_model_s{args.seed}.pt"
    candidates = [
        os.path.join(os.path.dirname(log_dir), model_name),  # old year-less path
        os.path.join(log_dir, model_name),
    ]
    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates:
        sys.exit(f"No {model_name} found in {log_dir} or its parent")
    newest = max(candidates, key=os.path.getmtime)
    target = os.path.join(log_dir, model_name)
    if newest != target:
        print(f"moving newest best model {newest} -> {target}")
        shutil.move(newest, target)

    epoch, min_loss, wait = parse_log(
        os.path.join(log_dir, f"record_s{args.seed}.log")
    )
    print(f"parsed log: last completed epoch={epoch}, "
          f"min valid loss={min_loss:.4f}, wait={wait}")

    ckpt = {
        "epoch": epoch,
        "min_loss": min_loss,
        "wait": wait,
        "finished": False,
        "iter_cnt": 0,
        "model": torch.load(target, map_location="cpu", weights_only=True),
        "optimizer": None,
        "scheduler": None,
        "rng": {},
        "extra": {},
    }
    out = args.out or os.path.join(log_dir, f"last_checkpoint_s{args.seed}.pt")
    torch.save(ckpt, out)
    print(f"wrote {out} — resubmitting the run script will resume at epoch {epoch + 1}")


if __name__ == "__main__":
    main()
