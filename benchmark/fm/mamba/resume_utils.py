"""Epoch-level checkpoint/resume for the SSM training loops.

Each term (short/medium/long) gets its own last_checkpoint_{term}_s{seed}.pt
holding the full training state, saved at the end of every epoch. The
best_model_{term}_s{seed}.pt files keep their existing meaning (best
validation weights, loaded for evaluation) and are untouched.
"""
import os
import random

import numpy as np
import torch


def resume_ckpt_path(args, term):
    return os.path.join(args.log_dir, f"last_checkpoint_{term}_s{args.seed}.pt")


def peek_resume(args, term):
    """Return the resume-checkpoint dict for this term, or None."""
    path = resume_ckpt_path(args, term)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def save_resume(args, term, model, optimizer, epoch, best_val, wait, finished):
    ckpt = {
        "epoch": epoch,
        "best_val": float(best_val),
        "wait": wait,
        "finished": finished,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    path = resume_ckpt_path(args, term)
    # write-then-rename so a job killed mid-save can't corrupt the checkpoint
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)


def restore_resume(state, model, optimizer):
    """Load training state from a peeked checkpoint.

    The caller must already have rebuilt any compressed layer shapes to match
    state["model"] (see the per-model shape adaptation in mamba.py / lru.py).
    Returns (start_epoch, best_val, wait).
    """
    model.load_state_dict(state["model"])
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    rng = state.get("rng", {})
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    print(
        f"  Resumed from epoch {state['epoch']} "
        f"(best_val={state['best_val']:.4f}, wait={state['wait']})"
    )
    return state["epoch"] + 1, state["best_val"], state["wait"]
