"""
Compare sequential SSM vs parallel SSM + CompreSSM compression speed.

Run from benchmark/:
    uv run --project fm/moirai python fm/mamba/benchmark_scan.py
"""

import time
import torch
from mamba_model    import MambaForecastModel
from mamba_model_v2 import MambaForecastModelV2


def warmup(model, x, n=5):
    for _ in range(n):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()


def bench(model, x, n=50):
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms per forward pass


def compress_model(model, x):
    """Apply CompreSSM compression to all MambaBlockV2 SSM layers using sample input."""
    h = model.input_proj(x)  # (B, L, d_model)
    for block in model.blocks:
        if hasattr(block, 'ssm') and hasattr(block.ssm, 'compress'):
            h_norm = block.norm(h)
            x_in, _ = block.in_proj(h_norm).chunk(2, dim=-1)
            x_in = block.conv1d(x_in.transpose(1, 2))[..., :h.shape[1]]
            x_in = torch.nn.functional.silu(x_in.transpose(1, 2))
            block.ssm.compress(x_in, energy_threshold=0.99)
        h = block(h)  # update h for next block


def run(batch_size, context_len, pred_len, d_model, num_layers, device_str, n=50):
    device = torch.device(device_str)
    cfg = dict(prediction_length=pred_len, d_model=d_model,
               d_state=16, d_conv=4, expand=2, num_layers=num_layers)

    m_seq  = MambaForecastModel(**cfg).to(device).eval()
    m_comp = MambaForecastModelV2(**cfg).to(device).eval()

    x = torch.randn(batch_size, context_len, 1, device=device)

    # Apply compression with a small calibration batch
    x_calib = torch.randn(32, context_len, 1, device=device)
    with torch.no_grad():
        compress_model(m_comp, x_calib)

    # Report compressed d_state
    compressed_states = [m.ssm.d_state for m in m_comp.modules() if hasattr(m, 'ssm') and hasattr(m.ssm, 'compress')]
    r_str = str(compressed_states[0]) if compressed_states else '?'

    with torch.no_grad():
        warmup(m_seq,  x)
        warmup(m_comp, x)
        t_seq  = bench(m_seq,  x, n)
        t_comp = bench(m_comp, x, n)

    speedup = t_seq / t_comp
    print(
        f"  B={batch_size:<4} L={context_len:<5} d={d_model:<4} layers={num_layers}"
        f" | seq={t_seq:7.2f}ms  comp(r={r_str})={t_comp:7.2f}ms  speedup={speedup:.2f}x"
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("=== Short context (L=48, traffic 15T) ===")
    for bs in [64, 256, 512]:
        run(bs, 48, 3,  d_model=64,  num_layers=2, device_str=device)

    print("\n=== Longer context (L=192, 1-hour) ===")
    for bs in [64, 256, 512]:
        run(bs, 192, 12, d_model=64,  num_layers=2, device_str=device)

    print("\n=== Larger model (d_model=128, 4 layers) ===")
    for bs in [64, 256]:
        run(bs, 48, 3,  d_model=128, num_layers=4, device_str=device)

    print("\n=== Very long context (L=512) ===")
    for bs in [32, 128]:
        run(bs, 512, 12, d_model=64,  num_layers=2, device_str=device)
