"""
Test that LRU/S4/Mamba checkpoints saved after compression can be loaded
back correctly without shape mismatch.

Run from benchmark/:
  uv run --project fm/moirai python -m ssm.test_ckpt_load
"""

import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"


# ── LRU ──────────────────────────────────────────────────────────────────────

def test_lru_ckpt_roundtrip():
    from ssm.lru_model import LRUForecastModel, LRULayer

    d_model, d_state, num_layers = 16, 8, 2
    model = LRUForecastModel(prediction_length=3, d_model=d_model,
                             d_state=d_state, num_layers=num_layers)

    # compress each block twice: 8 → ~4 → ~2
    for _ in range(2):
        with torch.no_grad():
            for block in model.blocks:
                block.lru.compress(energy_threshold=0.5)

    d_states_after = [b.lru.d_state for b in model.blocks]
    assert any(d < d_state for d in d_states_after), "compression had no effect"

    # save & reload using lru.py's new _get_model logic
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(model.state_dict(), ckpt_path)

    # reconstruct with original (wrong) d_state, then fix via sd key inspection
    model2 = LRUForecastModel(prediction_length=3, d_model=d_model,
                              d_state=d_state, num_layers=num_layers)
    sd = torch.load(ckpt_path, map_location="cpu")
    for i, block in enumerate(model2.blocks):
        key = f"blocks.{i}.lru.nu_log"
        if key in sd:
            r = sd[key].shape[1]
            if r != block.lru.d_state:
                block.lru = LRULayer(d_model, r)
    model2.load_state_dict(sd)
    model2.eval()

    x = torch.randn(4, 12, 1)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)

    assert out1.shape == (4, 3), f"wrong output shape: {out1.shape}"
    assert torch.allclose(out1, out2, atol=1e-5), "outputs differ after reload"
    os.unlink(ckpt_path)
    print(f"{PASS} LRU: compress + save + load → shape {out1.shape}, outputs match")


# ── S4 ───────────────────────────────────────────────────────────────────────

def test_s4_ckpt_roundtrip():
    from ssm.s4_model import S4ForecastModel, S4Layer

    d_model, d_state, num_layers = 16, 8, 2
    model = S4ForecastModel(prediction_length=3, d_model=d_model,
                            d_state=d_state, num_layers=num_layers)

    with torch.no_grad():
        for block in model.blocks:
            block.s4.compress(energy_threshold=0.5)

    d_states_after = [b.s4.d_state for b in model.blocks]
    assert any(d < d_state for d in d_states_after), "compression had no effect"

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(model.state_dict(), ckpt_path)

    # reload using s4.py's existing _get_model logic
    model2 = S4ForecastModel(prediction_length=3, d_model=d_model,
                             d_state=d_state, num_layers=num_layers)
    sd = torch.load(ckpt_path, map_location="cpu")
    for i, block in enumerate(model2.blocks):
        key = f"blocks.{i}.s4.Lambda_re"
        if key in sd:
            r = sd[key].shape[1]
            if r != block.s4.d_state:
                block.s4 = S4Layer(d_model, r).to("cpu")
    model2.load_state_dict(sd)
    model2.eval()

    x = torch.randn(4, 12, 1)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)

    assert out1.shape == (4, 3), f"wrong output shape: {out1.shape}"
    assert torch.allclose(out1, out2, atol=1e-5), "outputs differ after reload"
    os.unlink(ckpt_path)
    print(f"{PASS} S4:  compress + save + load → shape {out1.shape}, outputs match")


# ── Mamba ─────────────────────────────────────────────────────────────────────

def test_mamba_ckpt_roundtrip():
    from ssm.mamba_model_v2 import MambaForecastModelV2, SelectiveSSMParallel

    d_model, d_state, num_layers = 16, 8, 2
    model = MambaForecastModelV2(prediction_length=3, d_model=d_model,
                                 d_state=d_state, num_layers=num_layers)

    x_calib = torch.randn(4, 12, 1)
    with torch.no_grad():
        h = model.input_proj(x_calib)
        for block in model.blocks:
            if hasattr(block, 'ssm') and hasattr(block.ssm, 'compress'):
                h_norm = block.norm(h)
                x_in, _ = block.in_proj(h_norm).chunk(2, dim=-1)
                x_in = block.conv1d(x_in.transpose(1, 2))[..., :h.shape[1]]
                x_in = nn.functional.silu(x_in.transpose(1, 2))
                block.ssm.compress(x_in, energy_threshold=0.5)
            h = block(h)

    d_states_after = [b.ssm.d_state for b in model.blocks]
    assert any(d < d_state for d in d_states_after), "compression had no effect"

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save(model.state_dict(), ckpt_path)

    # reload using mamba.py's existing _get_model logic
    model2 = MambaForecastModelV2(prediction_length=3, d_model=d_model,
                                  d_state=d_state, num_layers=num_layers)
    sd = torch.load(ckpt_path, map_location="cpu")
    for i, block in enumerate(model2.blocks):
        key = f"blocks.{i}.ssm.A_log"
        if key in sd:
            r = sd[key].shape[1]
            d_inner = block.ssm.d_inner
            if r != block.ssm.d_state:
                block.ssm = SelectiveSSMParallel(d_inner, r)
    model2.load_state_dict(sd)
    model2.eval()

    x = torch.randn(4, 12, 1)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)

    assert out1.shape == (4, 3), f"wrong output shape: {out1.shape}"
    assert torch.allclose(out1, out2, atol=1e-5), "outputs differ after reload"
    os.unlink(ckpt_path)
    print(f"{PASS} Mamba: compress + save + load → shape {out1.shape}, outputs match")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [test_lru_ckpt_roundtrip, test_s4_ckpt_roundtrip, test_mamba_ckpt_roundtrip]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"{FAIL} {t.__name__}: {e}")
            failures += 1
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("\nAll checkpoint roundtrip tests passed.")
