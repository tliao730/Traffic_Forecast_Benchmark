from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, fields
from typing import Union

import yaml


@dataclass
class BenchmarkConfig:
    """Benchmark configuration.

    All defaults live in ``configs/default.yaml``.  Every field can be
    overridden by passing another YAML file — only the keys you want to
    change need to appear in that file; the rest come from ``default.yaml``.
    """

    # ── Paths ──────────────────────────────────────────────────────────
    gift_eval_datasets_path: str
    dataset_properties_path: str
    result_root: str
    # Hugging Face / transformers cache root (HF_HOME).
    # This is used to avoid filling up small home partitions.
    hf_home: str

    # ── Datasets ───────────────────────────────────────────────────────
    short_datasets: str
    med_long_datasets: str

    # ── Device ─────────────────────────────────────────────────────────
    device: str

    # ── Default visualisation settings ─────────────────────────────────
    default_plot_dataset: str
    default_plot_term: str
    default_plot_sample_idx: int
    default_plot_quantile: float

    # ── Evaluation controls ────────────────────────────────────────────
    # Integer N uses the last N sliding windows; "all" uses all windows.
    test_num_windows: Union[int, str] = "all"

    # ── Convenience properties ─────────────────────────────────────────
    @property
    def result_root_abs(self) -> str:
        """Return *result_root* resolved to an absolute path."""
        return os.path.normpath(
            os.path.join(os.path.dirname(__file__), self.result_root)
        )

    @property
    def short_datasets_list(self) -> list[str]:
        """Split *short_datasets* into a list of individual dataset names."""
        return self.short_datasets.split()

    @property
    def med_long_datasets_list(self) -> list[str]:
        """Split *med_long_datasets* into a list of individual dataset names."""
        return self.med_long_datasets.split()

    # ── Serialisation helpers ──────────────────────────────────────────
    def to_dict(self) -> dict:
        """Return all fields as a plain dict (excludes properties)."""
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write the current config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # ── Factory methods ────────────────────────────────────────────────
    @classmethod
    def _default_yaml_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), "configs", "default.yaml")

    @classmethod
    def from_yaml(cls, path: str) -> BenchmarkConfig:
        """Load a config profile from a YAML file.

        ``configs/default.yaml`` is always loaded first as the base; the
        given file is then overlaid on top.  Only the keys you want to
        change need to appear in the override file.  Unknown keys are
        silently ignored.
        """
        with open(cls._default_yaml_path(), "r") as f:
            data = yaml.safe_load(f) or {}

        if os.path.abspath(path) != os.path.abspath(cls._default_yaml_path()):
            with open(path, "r") as f:
                data.update(yaml.safe_load(f) or {})

        valid_names = {fld.name for fld in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_names}
        return cls(**filtered)


def _resolve_config() -> BenchmarkConfig:
    """Determine which config to use at module-load time.

    Priority: ``--config`` CLI flag  >  ``BENCHMARK_CONFIG`` env var  >  configs/default.yaml.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config_path", default=None)
    known, _ = parser.parse_known_args(sys.argv[1:])

    path = known.config_path or os.environ.get("BENCHMARK_CONFIG")
    return BenchmarkConfig.from_yaml(path or BenchmarkConfig._default_yaml_path())


# ── Module-level singleton & backward-compatible exports ──────────────
config = _resolve_config()

# Hugging Face cache location
#
# Many model loaders call `from_pretrained(...)` without an explicit `cache_dir`,
# which makes transformers/huggingface hub fall back to their default caches
# (often under `~/.cache`). Centralizing this here makes the whole benchmark
# honor the YAML config.
_hf_home = getattr(config, "hf_home", None) or "/work/nvme/bevu/dcao1/HuggingFace"

# IMPORTANT: YAML should win over any pre-existing env vars set by the cluster.
os.environ["HF_HOME"] = _hf_home
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_home, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(_hf_home, "modules")
os.environ["TRANSFORMERS_MODULES_CACHE"] = os.path.join(_hf_home, "modules")

# Re-export every field so ``from config import X`` keeps working.
gift_eval_datasets_path = config.gift_eval_datasets_path
dataset_properties_path = config.dataset_properties_path
result_root = config.result_root
hf_home = config.hf_home
short_datasets = config.short_datasets
med_long_datasets = config.med_long_datasets
test_num_windows = config.test_num_windows
device = config.device
default_plot_dataset = config.default_plot_dataset
default_plot_term = config.default_plot_term
default_plot_sample_idx = config.default_plot_sample_idx
default_plot_quantile = config.default_plot_quantile
