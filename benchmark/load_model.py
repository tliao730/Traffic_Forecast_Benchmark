import importlib.util
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Iterable, Optional

from path_utils import ensure_path_in_sys_path, find_first_existing_path


@dataclass(frozen=True)
class ScriptRoots:
    script_dir: str
    fm_root: str
    benchmark_root: str
    repo_root: str


@dataclass(frozen=True)
class ModelEnvironment:
    model_name: str
    roots: ScriptRoots
    repo_path: Optional[str] = None

    def require_repo_path(self) -> str:
        if self.repo_path is None:
            raise RuntimeError(f"Model '{self.model_name}' does not define a local repo path.")
        return self.repo_path

    def repo_file(self, *parts: str) -> str:
        return os.path.join(self.require_repo_path(), *parts)

    def load_repo_module(self, module_name: str, *parts: str) -> ModuleType:
        return load_module_from_file(module_name, self.repo_file(*parts))


def resolve_script_roots(file_path: str) -> ScriptRoots:
    """Resolve the standard directory anchors for a benchmark model script."""
    script_dir = os.path.dirname(os.path.abspath(file_path))
    fm_root = os.path.dirname(script_dir)
    benchmark_root = os.path.dirname(fm_root)
    repo_root = os.path.dirname(benchmark_root)
    return ScriptRoots(
        script_dir=script_dir,
        fm_root=fm_root,
        benchmark_root=benchmark_root,
        repo_root=repo_root,
    )


def ensure_benchmark_imports(file_path: str, prepend: bool = True) -> ScriptRoots:
    """Ensure benchmark-local modules such as common.py are importable."""
    roots = resolve_script_roots(file_path)
    ensure_path_in_sys_path(roots.benchmark_root, prepend=prepend)
    return roots


def resolve_model_repo(
    candidates: Iterable[Optional[str]],
    *,
    missing_message: str,
    required_subpath: Optional[str] = None,
    add_to_sys_path: bool = False,
    prepend: bool = False,
) -> str:
    """
    Resolve a local checkout for an external model repository.

    When requested, the resolved repository root is also added to sys.path.
    """
    repo_path, searched_paths = find_first_existing_path(
        candidates,
        required_subpath=required_subpath,
    )
    if repo_path is None:
        searched = ", ".join(searched_paths) if searched_paths else "<none>"
        raise FileNotFoundError(f"{missing_message}\nSearched in: {searched}")

    if add_to_sys_path:
        ensure_path_in_sys_path(repo_path, prepend=prepend)
    return repo_path


def ensure_optional_subdir_in_sys_path(
    root_path: str,
    subdir: str,
    *,
    prepend: bool = False,
) -> Optional[str]:
    """Add an existing child directory to sys.path and return its path."""
    subdir_path = os.path.join(root_path, subdir)
    if not os.path.exists(subdir_path):
        return None
    ensure_path_in_sys_path(subdir_path, prepend=prepend)
    return subdir_path


def load_module_from_file(module_name: str, file_path: str) -> ModuleType:
    """Load a Python module directly from a file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load module spec from {file_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_model_repo_config(
    model_name: str,
    roots: ScriptRoots,
) -> Optional[dict]:
    if model_name == "tabpfn_ts":
        tabpfn_env_path = os.environ.get("TABPFN_TS_PATH") or os.environ.get(
            "TABPFN_TIME_SERIES_PATH"
        )
        return {
            "candidates": [
                tabpfn_env_path,
                os.path.join(roots.fm_root, "envs", "tabpfn_ts", "tabpfn-time-series"),
                os.path.join(os.path.expanduser("~"), "tabpfn-time-series"),
                os.path.join(roots.fm_root, "tabpfn-time-series"),
                "tabpfn-time-series",
            ],
            "missing_message": (
                "tabpfn-time-series repository not found. Please clone it first:\n"
                "  git clone https://github.com/PriorLabs/tabpfn-time-series.git\n"
                "  cd tabpfn-time-series && git checkout v1.0.0"
            ),
            "add_to_sys_path": True,
            "prepend": False,
            "optional_sys_path_subdirs": ["gift_eval"],
        }

    if model_name == "flowstate":
        return {
            "candidates": [
                os.environ.get("GRANITE_TSFM_PATH"),
                os.path.join(roots.fm_root, "envs", "flowstate", "granite-tsfm"),
            ],
            "missing_message": (
                "granite-tsfm repo not found. Clone it: "
                "git clone https://github.com/ibm-granite/granite-tsfm.git "
                "or set GRANITE_TSFM_PATH to your clone path."
            ),
            "required_subpath": "tsfm_public",
            "add_to_sys_path": True,
            "prepend": True,
        }

    if model_name == "kairos":
        return {
            "candidates": [
                os.environ.get("KAIROS_PATH"),
                os.path.join(roots.fm_root, "Kairos"),
                os.path.join(roots.fm_root, "envs", "kairos", "Kairos"),
            ],
            "missing_message": (
                "Kairos repo not found. Clone it and set KAIROS_PATH:\n"
                "  git clone https://github.com/foundation-model-research/Kairos.git\n"
                "  export KAIROS_PATH=/path/to/Kairos"
            ),
            "required_subpath": "tsfm",
            "add_to_sys_path": True,
            "prepend": True,
        }

    if model_name == "toto":
        return {
            "candidates": [
                os.environ.get("TOTO_PATH"),
                os.path.join(roots.fm_root, "envs", "toto", "toto"),
            ],
            "missing_message": (
                "toto repository not found. Clone it: "
                "git clone https://github.com/DataDog/toto.git "
                "or set TOTO_PATH to your clone path."
            ),
            "required_subpath": "toto",
            "add_to_sys_path": True,
            "prepend": True,
        }

    return None


def setup_model_environment(model_name: str, file_path: str) -> ModelEnvironment:
    """
    Apply centralized import/path setup for a benchmark model script.

    This keeps per-model scripts small by moving repo lookup rules into one place.
    """
    roots = ensure_benchmark_imports(file_path)
    repo_config = _resolve_model_repo_config(model_name, roots)
    if repo_config is None:
        return ModelEnvironment(model_name=model_name, roots=roots)

    repo_path = resolve_model_repo(
        repo_config["candidates"],
        missing_message=repo_config["missing_message"],
        required_subpath=repo_config.get("required_subpath"),
        add_to_sys_path=repo_config.get("add_to_sys_path", False),
        prepend=repo_config.get("prepend", False),
    )

    for subdir in repo_config.get("optional_sys_path_subdirs", []):
        ensure_optional_subdir_in_sys_path(
            repo_path,
            subdir,
            prepend=repo_config.get("prepend", False),
        )

    return ModelEnvironment(
        model_name=model_name,
        roots=roots,
        repo_path=repo_path,
    )
