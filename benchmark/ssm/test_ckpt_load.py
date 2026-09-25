"""
Test that an LRU checkpoint saved after compression loads back without a
shape mismatch. LRU is the only model CompreSSM shrinks mid-training, so it is
the only one whose d_state can differ between the constructor and the file.

Run from benchmark/:
  uv run --project fm/moirai python -m ssm.test_ckpt_load
"""

import os
import sys
import tempfile

import torch

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


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [test_lru_ckpt_roundtrip]
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
