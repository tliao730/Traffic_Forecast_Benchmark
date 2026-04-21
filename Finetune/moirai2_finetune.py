from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

import sys
from pathlib import Path

# Make benchmark/ importable so `from common import eval` works
_BENCHMARK_DIR = (Path(__file__).resolve().parents[1] / "benchmark").as_posix()
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARK_DIR)

from common import eval


def _iter_input_entries(test_data_input: object) -> Iterator[dict]:
    """
    Robustly iterate *input* entries from what GluonTS passes to Predictor.predict(...).

    We may receive:
    - a GluonTS `TestData` (has `.input`)
    - a Dataset iterable of dict entries
    - an iterable of (input, label) tuples
    """
    iterable: Iterable = getattr(test_data_input, "input", test_data_input)  # type: ignore
    for item in iterable:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], dict):
            yield item[0]
        else:
            yield item


@dataclass(frozen=True)
class _AffineCalib:
    scale: float
    bias: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        # x: (...,) numpy
        return (x * self.scale + self.bias).astype(np.float32, copy=False)


def _calib_path(model_name: str, ds_config: str) -> Path:
    # Store per-dataset/term finetune params inside the benchmark results dir
    safe = ds_config.replace("/", "__")
    return Path(__file__).resolve().parents[1] / "results" / model_name / "finetune" / f"{safe}.npz"

def _full_ckpt_path(model_name: str, ds_config: str) -> Path:
    safe = ds_config.replace("/", "__")
    return Path(__file__).resolve().parents[1] / "results" / model_name / "ckpt" / f"{safe}.pth"

def _lightning_ckpt_path(model_name: str, ds_config: str) -> Path:
    safe = ds_config.replace("/", "__")
    return (
        Path(__file__).resolve().parents[1]
        / "results"
        / model_name
        / "ckpt_lightning"
        / f"{safe}.ckpt"
    )


def _pinball_loss(pred: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
    # pred,y: (B,H)
    diff = y - pred
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))


def _fit_full_model(
    *,
    module: Moirai2Module,
    prediction_length: int,
    context_length: int,
    device: torch.device,
    training_dataset,
    steps: int,
    lr: float,
    series_limit: int,
    loss_type: str,
) -> Moirai2Module:
    """
    Full finetune of Moirai2Module parameters (per dataset/term).
    Uses random windows from dataset.training_dataset; does NOT touch test.
    """
    # wrap in Forecast for the differentiable forward -> quantile predictions
    forecast = Moirai2Forecast(
        module=module,
        prediction_length=int(prediction_length),
        context_length=int(context_length),
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(str(device))
    forecast.train()
    forecast.module.train()

    opt = torch.optim.AdamW(forecast.module.parameters(), lr=float(lr))

    # build pool of 1D series
    pool: List[np.ndarray] = []
    for i, entry in enumerate(training_dataset):
        if i >= int(series_limit):
            break
        x = np.asarray(entry["target"], dtype=np.float32).reshape(-1)
        if x.size < (int(prediction_length) + 4):
            continue
        pool.append(x)
    if not pool:
        return module

    rng = np.random.default_rng(42)
    q_levels = tuple(float(q) for q in getattr(forecast.module, "quantile_levels", (0.5,)))
    # median index
    med_idx = len(q_levels) // 2

    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        x = pool[int(rng.integers(0, len(pool)))]
        max_ctx = min(int(context_length), len(x) - int(prediction_length))
        if max_ctx < 8:
            continue
        t0 = int(rng.integers(max_ctx, len(x) - int(prediction_length) + 1))
        past = x[t0 - max_ctx : t0]
        fut = x[t0 : t0 + int(prediction_length)]

        # pad/slice to exactly context_length like predictor does
        if past.shape[0] > int(context_length):
            past = past[-int(context_length) :]
            pad = np.zeros((int(context_length),), dtype=bool)
        elif past.shape[0] < int(context_length):
            pad_len = int(context_length) - past.shape[0]
            pad_block = np.full((pad_len,), past[0], dtype=np.float32)
            past = np.concatenate([pad_block, past], axis=0)
            pad = np.zeros((int(context_length),), dtype=bool)
            pad[:pad_len] = True
        else:
            pad = np.zeros((int(context_length),), dtype=bool)

        past_t = torch.from_numpy(past[None, :, None]).to(device=device, dtype=torch.float32)
        past_obs = (~torch.isnan(past_t)).to(torch.bool)
        past_is_pad = torch.from_numpy(pad[None, :]).to(device=device, dtype=torch.bool)

        preds_q = forecast(
            past_target=past_t,
            past_observed_target=past_obs,
            past_is_pad=past_is_pad,
        )  # (B,Q,H) for univariate

        # ensure (B,Q,H)
        if preds_q.ndim == 2:
            preds_q = preds_q[:, None, :]

        y = torch.from_numpy(fut[None, :]).to(device=device, dtype=torch.float32)

        if loss_type == "mae_median":
            pred = preds_q[:, med_idx, :]
            loss = torch.mean(torch.abs(pred - y))
        elif loss_type == "pinball":
            loss = torch.zeros((), device=device)
            for qi, qv in enumerate(q_levels):
                loss = loss + _pinball_loss(preds_q[:, qi, :], y, float(qv))
            loss = loss / max(1, len(q_levels))
        else:
            raise ValueError(f"Unknown MOIRAI2_FULL_FT_LOSS: {loss_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(forecast.module.parameters(), max_norm=1.0)
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[moirai2 full-ft] step {step+1}/{steps} loss={loss.item():.4f}")

    return forecast.module


class _RandomWindowTorchDataset(torch.utils.data.Dataset):
    """
    Epoch-defined random-window dataset.

    Like TEMPO_gluonts, we define an epoch budget via `num_batches_per_epoch` rather than
    "iterate all possible windows". Each item is one random (past, future) window.
    """

    def __init__(
        self,
        training_dataset,
        *,
        context_length: int,
        prediction_length: int,
        batch_size: int,
        num_batches_per_epoch: int,
        series_limit: int,
        seed: int,
    ) -> None:
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self._len = int(batch_size) * int(num_batches_per_epoch)
        self.rng = np.random.default_rng(int(seed))

        def _impute_nan_1d(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x
            if not np.isnan(x).any():
                return x
            obs = ~np.isnan(x)
            if not np.any(obs):
                return np.zeros_like(x)
            # forward-fill
            idx = np.where(obs, np.arange(x.size), 0)
            np.maximum.accumulate(idx, out=idx)
            x_ff = x[idx]
            # backfill leading NaNs (if first entries were NaN)
            first = int(np.argmax(obs))
            x_ff[:first] = x_ff[first]
            return x_ff

        pool: List[np.ndarray] = []
        for i, entry in enumerate(training_dataset):
            if i >= int(series_limit):
                break
            x = np.asarray(entry["target"], dtype=np.float32).reshape(-1)
            x = _impute_nan_1d(x)
            if x.size < (self.prediction_length + 8):
                continue
            pool.append(x)
        self.pool = pool

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if not self.pool:
            past = np.zeros((self.context_length,), dtype=np.float32)
            fut = np.zeros((self.prediction_length,), dtype=np.float32)
            return {"past": past, "future": fut}

        x = self.pool[int(self.rng.integers(0, len(self.pool)))]
        max_ctx = min(self.context_length, len(x) - self.prediction_length)
        if max_ctx < 8:
            past = x[-self.context_length :]
            fut = x[-self.prediction_length :]
        else:
            t0 = int(self.rng.integers(max_ctx, len(x) - self.prediction_length + 1))
            past = x[t0 - max_ctx : t0]
            fut = x[t0 : t0 + self.prediction_length]

        if past.shape[0] > self.context_length:
            past = past[-self.context_length :]
        elif past.shape[0] < self.context_length:
            pad_len = self.context_length - past.shape[0]
            pad_block = np.full((pad_len,), past[0], dtype=np.float32)
            past = np.concatenate([pad_block, past], axis=0)

        return {
            "past": torch.from_numpy(past.astype(np.float32, copy=False)),
            "future": torch.from_numpy(fut.astype(np.float32, copy=False)),
        }


def _build_l1_lightning_module(
    *,
    module: Moirai2Module,
    context_length: int,
    prediction_length: int,
    loss_type: str,
    lr: float,
    weight_decay: float,
) -> "object":
    """
    Build a LightningModule for Route1 finetuning without creating module reference cycles.
    Returns an instance of lightning.pytorch.LightningModule.
    """
    import lightning as L  # type: ignore

    q_levels = tuple(float(q) for q in getattr(module, "quantile_levels", (0.5,)))
    med_idx = len(q_levels) // 2

    class _LM(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.module = module
            self.forecast = Moirai2Forecast(
                module=self.module,
                prediction_length=int(prediction_length),
                context_length=int(context_length),
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self.loss_type = str(loss_type)
            self.q_levels = q_levels
            self.med_idx = int(med_idx)
            self.lr = float(lr)
            self.weight_decay = float(weight_decay)

        def _loss(self, preds_q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if self.loss_type == "mae_median":
                pred = preds_q[:, self.med_idx, :]
                return torch.mean(torch.abs(pred - y))
            if self.loss_type == "pinball":
                loss = torch.zeros((), device=y.device)
                for qi, qv in enumerate(self.q_levels):
                    loss = loss + _pinball_loss(preds_q[:, qi, :], y, float(qv))
                return loss / max(1, len(self.q_levels))
            raise ValueError(f"Unknown MOIRAI2_L1_LOSS: {self.loss_type}")

        def _step(self, batch) -> torch.Tensor:
            past = batch["past"].to(self.device)  # (B,C)
            fut = batch["future"].to(self.device)  # (B,H)
            past_t = past[:, :, None]  # (B,C,1)
            past_obs = (~torch.isnan(past_t)).to(torch.bool)
            past_is_pad = torch.zeros((past_t.shape[0], past_t.shape[1]), device=self.device, dtype=torch.bool)
            preds_q = self.forecast(past_target=past_t, past_observed_target=past_obs, past_is_pad=past_is_pad)
            if preds_q.ndim == 2:
                preds_q = preds_q[:, None, :]
            return self._loss(preds_q, fut)

        def training_step(self, batch, batch_idx):
            loss = self._step(batch)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss = self._step(batch)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(
                self.module.parameters(),
                lr=float(self.lr),
                weight_decay=float(self.weight_decay),
            )

    return _LM()


def _fit_affine_calibration(
    *,
    model: Moirai2Forecast,
    training_dataset,
    prediction_length: int,
    context_length: int,
    device: torch.device,
    steps: int,
    lr: float,
    series_limit: int,
) -> _AffineCalib:
    """
    Per-dataset/term 'finetune': learn a global affine calibration y' = a*y + b
    on top of Moirai2 median forecast, using the dataset's training split.

    This is intentionally lightweight and does not modify Moirai2 weights.
    """
    # trainable parameters
    a = torch.tensor(1.0, device=device, requires_grad=True)
    b = torch.tensor(0.0, device=device, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=float(lr))

    # Build a small pool of training series (deterministic)
    pool: List[np.ndarray] = []
    for i, entry in enumerate(training_dataset):
        if i >= int(series_limit):
            break
        target = np.asarray(entry["target"], dtype=np.float32)
        if target.ndim != 1:
            target = target.reshape(-1).astype(np.float32)
        pool.append(target)
    if not pool:
        return _AffineCalib(scale=1.0, bias=0.0)

    rng = np.random.default_rng(42)
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        # sample one series and one cut-point
        x = pool[int(rng.integers(0, len(pool)))]
        if len(x) < prediction_length + 2:
            continue
        max_ctx = min(int(context_length), len(x) - int(prediction_length))
        if max_ctx < 2:
            continue
        # use last max_ctx points as context; next pred_len points as future label
        # choose a random split inside the series to get more diverse windows
        t0 = int(rng.integers(max_ctx, len(x) - int(prediction_length) + 1))
        past = x[t0 - max_ctx : t0]
        fut = x[t0 : t0 + int(prediction_length)]

        past_t = torch.from_numpy(past[None, :]).to(device)
        with torch.no_grad():
            q = model.predict([past_t.squeeze(0).detach().cpu().numpy()])[0]  # (Q, H)
        # median index: assume quantiles include 0.5 at position 4 (0.1..0.9)
        # fall back: use middle quantile
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != int(prediction_length):
            continue
        median = q[q.shape[0] // 2]
        pred = torch.from_numpy(median).to(device)
        y = torch.from_numpy(fut).to(device)

        pred_cal = a * pred + b
        loss = torch.mean(torch.abs(pred_cal - y))
        loss.backward()
        opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"[moirai2 finetune] step {step+1}/{steps} loss={loss.item():.4f} a={a.item():.4f} b={b.item():.4f}")

    return _AffineCalib(scale=float(a.detach().cpu().item()), bias=float(b.detach().cpu().item()))


class MoiraiQuantilePredictor:
    def __init__(
        self,
        model_path: str,
        module: Moirai2Module | None = None,
        prediction_length: int = 100,
        context_length: int = 4000,
        target_dim: int = 1,
        feat_dynamic_real_dim: int = 0,
        past_feat_dynamic_real_dim: int = 0,
        device_str: str = "auto",
        batch_size: int = 2048,
        quantile_levels: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        calib: _AffineCalib | None = None,
    ):
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        self.feat_dynamic_real_dim = feat_dynamic_real_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.device = device_str
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.calib = calib
        self.model = Moirai2Forecast(
            module=module if module is not None else Moirai2Module.from_pretrained(self.model_path),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            target_dim=self.target_dim,
            feat_dynamic_real_dim=self.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
        ).to(self.device)

    def predict(self, test_data_input):
        while True:
            try:
                print("Model - MoiraiQuantile loaded with batch_size:", self.batch_size)
                # Generate forecast samples
                forecast_quantiles = []
                entries = list(_iter_input_entries(test_data_input))
                for batch in batcher(entries, batch_size=self.batch_size):
                    past_target = [entry["target"] for entry in batch]
                    forecasts = self.model.predict(past_target)
                    forecast_quantiles.append(forecasts)
                forecast_quantiles = np.concatenate(forecast_quantiles)
                if self.calib is not None:
                    forecast_quantiles = self.calib.apply(forecast_quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {self.batch_size}, reducing to {self.batch_size // 2}"
                )
                self.batch_size //= 2

        # Convert forecast samples into gluonts QuantileForecast objects
        quantile_forecasts = []
        for item, ts in zip(forecast_quantiles, entries):
            forecast_start_date = ts["start"] + len(ts["target"])
            quantile_forecasts.append(
                QuantileForecast(
                    item_id=ts["item_id"],
                    forecast_arrays=item,
                    start_date=forecast_start_date,
                    forecast_keys=list(map(str, self.quantile_levels)),
                )
            )
        return quantile_forecasts


def main():
    # benchmark model name and optional finetune controls
    model_name = os.getenv("MOIRAI2_MODEL_NAME", "Moirai2").strip()
    model_path = "Salesforce/moirai-2.0-R-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    finetune = os.getenv("MOIRAI2_FINETUNE", "0").strip() == "1"
    ft_steps = int(os.getenv("MOIRAI2_FT_STEPS", "200"))
    ft_lr = float(os.getenv("MOIRAI2_FT_LR", "0.01"))
    ft_series = int(os.getenv("MOIRAI2_FT_SERIES", "128"))
    ft_cache = os.getenv("MOIRAI2_FT_CACHE", "1").strip() == "1"

    full_ft = os.getenv("MOIRAI2_FULL_FINETUNE", "0").strip() == "1"
    full_steps = int(os.getenv("MOIRAI2_FULL_FT_STEPS", "200"))
    full_lr = float(os.getenv("MOIRAI2_FULL_FT_LR", "1e-5"))
    full_series = int(os.getenv("MOIRAI2_FULL_FT_SERIES", "64"))
    full_cache = os.getenv("MOIRAI2_FULL_FT_CACHE", "1").strip() == "1"
    full_loss = os.getenv("MOIRAI2_FULL_FT_LOSS", "mae_median").strip()
    context_length = int(os.getenv("MOIRAI2_CONTEXT_LENGTH", "4000"))

    # Route 1: epoch-based finetune (TEMPO_gluonts-style)
    l1_ft = os.getenv("MOIRAI2_L1_FINETUNE", "0").strip() == "1"
    l1_epochs = int(os.getenv("MOIRAI2_L1_MAX_EPOCHS", "3"))
    l1_batch = int(os.getenv("MOIRAI2_L1_BATCH_SIZE", "64"))
    l1_nbpe = int(os.getenv("MOIRAI2_L1_NUM_BATCHES_PER_EPOCH", "200"))
    l1_lr = float(os.getenv("MOIRAI2_L1_LR", "1e-5"))
    l1_wd = float(os.getenv("MOIRAI2_L1_WEIGHT_DECAY", "0.0"))
    l1_clip = float(os.getenv("MOIRAI2_L1_GRAD_CLIP", "1.0"))
    l1_series = int(os.getenv("MOIRAI2_L1_SERIES", "256"))
    l1_cache = os.getenv("MOIRAI2_L1_CACHE", "1").strip() == "1"
    l1_loss = os.getenv("MOIRAI2_L1_LOSS", "mae_median").strip()

    def predictor_factory(dataset):
        calib: _AffineCalib | None = None
        finetuned_module: Moirai2Module | None = None
        if finetune:
            # Identify ds_config consistent with common.eval naming
            ds_name = str(getattr(dataset, "name", "dataset"))
            ds_key = ds_name.split("/")[0].lower() if "/" in ds_name else ds_name.lower()
            ds_freq = ds_name.split("/")[1] if "/" in ds_name else getattr(dataset, "freq", "unknown")
            term_obj = getattr(dataset, "term", "short")
            term = getattr(term_obj, "value", str(term_obj))
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            path = _calib_path(model_name, ds_config)
            if ft_cache and path.exists():
                d = np.load(path)
                calib = _AffineCalib(scale=float(d["scale"]), bias=float(d["bias"]))
                print(f"[moirai2 finetune] loaded calib from {path} (a={calib.scale:.4f}, b={calib.bias:.4f})")
            else:
                # Build the base model once for finetune
                base = Moirai2Forecast(
                    module=Moirai2Module.from_pretrained(model_path),
                    prediction_length=int(dataset.prediction_length),
                    context_length=4000,
                    target_dim=1,
                    feat_dynamic_real_dim=int(getattr(dataset, "past_feat_dynamic_real_dim", 0)),
                    past_feat_dynamic_real_dim=int(getattr(dataset, "past_feat_dynamic_real_dim", 0)),
                ).to(device)
                base.eval()
                calib = _fit_affine_calibration(
                    model=base,
                    training_dataset=getattr(dataset, "training_dataset"),
                    prediction_length=int(dataset.prediction_length),
                    context_length=4000,
                    device=torch.device(device),
                    steps=ft_steps,
                    lr=ft_lr,
                    series_limit=ft_series,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, scale=calib.scale, bias=calib.bias)
                print(f"[moirai2 finetune] saved calib to {path} (a={calib.scale:.4f}, b={calib.bias:.4f})")

        if full_ft:
            ds_name = str(getattr(dataset, "name", "dataset"))
            ds_key = ds_name.split("/")[0].lower() if "/" in ds_name else ds_name.lower()
            ds_freq = ds_name.split("/")[1] if "/" in ds_name else getattr(dataset, "freq", "unknown")
            term_obj = getattr(dataset, "term", "short")
            term = getattr(term_obj, "value", str(term_obj))
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            ckpt_path = _full_ckpt_path(model_name, ds_config)
            if full_cache and ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                finetuned_module = Moirai2Module.from_pretrained(model_path)
                finetuned_module.load_state_dict(ckpt["module_state_dict"])
                print(f"[moirai2 full-ft] loaded ckpt from {ckpt_path}")
            else:
                base_module = Moirai2Module.from_pretrained(model_path).to(device)
                finetuned_module = _fit_full_model(
                    module=base_module,
                    prediction_length=int(dataset.prediction_length),
                    context_length=int(context_length),
                    device=torch.device(device),
                    training_dataset=getattr(dataset, "training_dataset"),
                    steps=int(full_steps),
                    lr=float(full_lr),
                    series_limit=int(full_series),
                    loss_type=full_loss,
                )
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "module_state_dict": finetuned_module.state_dict(),
                        "model_path": model_path,
                        "prediction_length": int(dataset.prediction_length),
                        "context_length": int(context_length),
                        "loss": full_loss,
                        "steps": int(full_steps),
                        "lr": float(full_lr),
                        "series_limit": int(full_series),
                        "ds_config": ds_config,
                    },
                    ckpt_path,
                )
                print(f"[moirai2 full-ft] saved ckpt to {ckpt_path}")

        if l1_ft:
            ds_name = str(getattr(dataset, "name", "dataset"))
            ds_key = ds_name.split("/")[0].lower() if "/" in ds_name else ds_name.lower()
            ds_freq = ds_name.split("/")[1] if "/" in ds_name else getattr(dataset, "freq", "unknown")
            term_obj = getattr(dataset, "term", "short")
            term = getattr(term_obj, "value", str(term_obj))
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            ckpt_pth = _full_ckpt_path(model_name, ds_config)
            ckpt_ckpt = _lightning_ckpt_path(model_name, ds_config)

            if l1_cache and ckpt_pth.exists():
                ckpt = torch.load(ckpt_pth, map_location="cpu")
                finetuned_module = Moirai2Module.from_pretrained(model_path)
                finetuned_module.load_state_dict(ckpt["module_state_dict"])
                print(f"[moirai2 l1-ft] loaded ckpt from {ckpt_pth}")
            else:
                # Build module and finetune via Lightning Trainer
                base_module = Moirai2Module.from_pretrained(model_path)
                lm = _build_l1_lightning_module(
                    module=base_module,
                    context_length=int(context_length),
                    prediction_length=int(dataset.prediction_length),
                    loss_type=l1_loss,
                    lr=l1_lr,
                    weight_decay=l1_wd,
                )

                train_ds = _RandomWindowTorchDataset(
                    getattr(dataset, "training_dataset"),
                    context_length=int(context_length),
                    prediction_length=int(dataset.prediction_length),
                    batch_size=int(l1_batch),
                    num_batches_per_epoch=int(l1_nbpe),
                    series_limit=int(l1_series),
                    seed=42,
                )
                val_ds = _RandomWindowTorchDataset(
                    getattr(dataset, "validation_dataset"),
                    context_length=int(context_length),
                    prediction_length=int(dataset.prediction_length),
                    batch_size=int(l1_batch),
                    num_batches_per_epoch=max(1, int(l1_nbpe) // 10),
                    series_limit=min(int(l1_series), 128),
                    seed=43,
                )
                train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(l1_batch), shuffle=False, num_workers=0)
                val_loader = torch.utils.data.DataLoader(val_ds, batch_size=int(l1_batch), shuffle=False, num_workers=0)

                import lightning as L  # type: ignore
                from lightning.pytorch.callbacks import ModelCheckpoint  # type: ignore
                from lightning.pytorch.loggers import WandbLogger  # type: ignore

                ckpt_dir = ckpt_ckpt.parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_cb = ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    filename=f"{ds_config.replace('/','__')}" + "-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    save_top_k=1,
                    save_last=True,
                    mode="min",
                )
                wandb_logger = WandbLogger(
                    project=os.getenv("WANDB_PROJECT", "TrafficFM"),
                    name=f"{model_name}/{ds_config}",
                    config={
                        "model": model_name,
                        "ds_config": ds_config,
                        "epochs": l1_epochs,
                        "batch_size": l1_batch,
                        "num_batches_per_epoch": l1_nbpe,
                        "lr": l1_lr,
                        "loss": l1_loss,
                        "context_length": context_length,
                        "series_limit": l1_series,
                    },
                )
                trainer = L.Trainer(
                    max_epochs=int(l1_epochs),
                    callbacks=[ckpt_cb],
                    logger=wandb_logger,
                    enable_model_summary=False,
                    deterministic=True,
                    gradient_clip_val=float(l1_clip) if l1_clip and l1_clip > 0 else 0.0,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    num_sanity_val_steps=0,
                )
                trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)

                best_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
                if not best_path:
                    raise RuntimeError("No checkpoint was saved by Lightning.")

                # Load best checkpoint weights back into lm so `lm.module` is best.
                state = torch.load(best_path, map_location="cpu")
                lm.load_state_dict(state.get("state_dict", state), strict=False)
                finetuned_module = lm.module

                ckpt_pth.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "module_state_dict": finetuned_module.state_dict(),
                        "format": "state_dict_only",
                        "model_path": model_path,
                        "ds_config": ds_config,
                        "context_length": int(context_length),
                        "prediction_length": int(dataset.prediction_length),
                        "epochs": int(l1_epochs),
                        "batch_size": int(l1_batch),
                        "num_batches_per_epoch": int(l1_nbpe),
                        "lr": float(l1_lr),
                        "loss": l1_loss,
                    },
                    ckpt_pth,
                )
                # copy lightning checkpoint to a stable filename
                torch.save(state, ckpt_ckpt)
                print(f"[moirai2 l1-ft] saved ckpt to {ckpt_pth}")
                print(f"[moirai2 l1-ft] saved lightning ckpt to {ckpt_ckpt}")

        return MoiraiQuantilePredictor(
            model_path=model_path,
            module=finetuned_module,
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
            batch_size=64,
            device_str=device,
            calib=calib,
        )

    eval(model_name, model_path, predictor_factory)


if __name__ == "__main__":
    main()
