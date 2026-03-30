import importlib.util
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Iterable, Optional

from fm.fm_utils import load_benchmark_config_module
from path_utils import ensure_path_in_sys_path, find_first_existing_path


@dataclass(frozen=True)
class ScriptRoots:
    script_dir: str
    fm_root: str
    benchmark_root: str
    repo_root: str


CandidateFactory = Callable[[ScriptRoots], Optional[str]]


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


@dataclass(frozen=True)
class ModelRepoConfig:
    candidates: tuple[CandidateFactory, ...]
    missing_message: str
    required_subpath: Optional[str] = None
    add_to_sys_path: bool = False
    prepend: bool = False
    optional_sys_path_subdirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelRuntime:
    model_env: ModelEnvironment
    config_module: ModuleType

    @property
    def model_name(self) -> str:
        return self.model_env.model_name

    @property
    def roots(self) -> ScriptRoots:
        return self.model_env.roots

    @property
    def repo_path(self) -> Optional[str]:
        return self.model_env.repo_path

    @property
    def config(self):
        return self.config_module.config

    @property
    def device(self):
        return self.config_module.device

    def require_repo_path(self) -> str:
        return self.model_env.require_repo_path()

    def repo_file(self, *parts: str) -> str:
        return self.model_env.repo_file(*parts)

    def load_repo_module(self, module_name: str, *parts: str) -> ModuleType:
        return self.model_env.load_repo_module(module_name, *parts)


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


def _env_candidate(*var_names: str) -> CandidateFactory:
    def resolve(_: ScriptRoots) -> Optional[str]:
        for var_name in var_names:
            value = os.environ.get(var_name)
            if value:
                return value
        return None

    return resolve


def _fm_candidate(*parts: str) -> CandidateFactory:
    def resolve(roots: ScriptRoots) -> str:
        return os.path.join(roots.fm_root, *parts)

    return resolve


def _home_candidate(*parts: str) -> CandidateFactory:
    def resolve(_: ScriptRoots) -> str:
        return os.path.join(os.path.expanduser("~"), *parts)

    return resolve


def _literal_candidate(path: str) -> CandidateFactory:
    def resolve(_: ScriptRoots) -> str:
        return path

    return resolve


MODEL_REPO_CONFIGS: dict[str, ModelRepoConfig] = {
    "tabpfn_ts": ModelRepoConfig(
        candidates=(
            _env_candidate("TABPFN_TS_PATH", "TABPFN_TIME_SERIES_PATH"),
            _fm_candidate("envs", "tabpfn_ts", "tabpfn-time-series"),
            _home_candidate("tabpfn-time-series"),
            _fm_candidate("tabpfn-time-series"),
            _literal_candidate("tabpfn-time-series"),
        ),
        missing_message=(
            "tabpfn-time-series repository not found. Please clone it first:\n"
            "  git clone https://github.com/PriorLabs/tabpfn-time-series.git\n"
            "  cd tabpfn-time-series && git checkout v1.0.0"
        ),
        add_to_sys_path=True,
        optional_sys_path_subdirs=("gift_eval",),
    ),
    "flowstate": ModelRepoConfig(
        candidates=(
            _env_candidate("GRANITE_TSFM_PATH"),
            _fm_candidate("envs", "flowstate", "granite-tsfm"),
        ),
        missing_message=(
            "granite-tsfm repo not found. Clone it: "
            "git clone https://github.com/ibm-granite/granite-tsfm.git "
            "or set GRANITE_TSFM_PATH to your clone path."
        ),
        required_subpath="tsfm_public",
        add_to_sys_path=True,
        prepend=True,
    ),
    "kairos": ModelRepoConfig(
        candidates=(
            _env_candidate("KAIROS_PATH"),
            _fm_candidate("Kairos"),
            _fm_candidate("envs", "kairos", "Kairos"),
        ),
        missing_message=(
            "Kairos repo not found. Clone it and set KAIROS_PATH:\n"
            "  git clone https://github.com/foundation-model-research/Kairos.git\n"
            "  export KAIROS_PATH=/path/to/Kairos"
        ),
        required_subpath="tsfm",
        add_to_sys_path=True,
        prepend=True,
    ),
    "toto": ModelRepoConfig(
        candidates=(
            _env_candidate("TOTO_PATH"),
            _fm_candidate("envs", "toto", "toto"),
        ),
        missing_message=(
            "toto repository not found. Clone it: "
            "git clone https://github.com/DataDog/toto.git "
            "or set TOTO_PATH to your clone path."
        ),
        required_subpath="toto",
        add_to_sys_path=True,
        prepend=True,
    ),
}


def _resolve_model_repo_config(model_name: str) -> Optional[ModelRepoConfig]:
    return MODEL_REPO_CONFIGS.get(model_name)


def _resolve_repo_candidates(
    config: ModelRepoConfig,
    roots: ScriptRoots,
) -> list[Optional[str]]:
    return [candidate(roots) for candidate in config.candidates]


def setup_model_environment(model_name: str, file_path: str) -> ModelEnvironment:
    """
    Apply centralized import/path setup for a benchmark model script.

    This keeps per-model scripts small by moving repo lookup rules into one place.
    """
    roots = ensure_benchmark_imports(file_path)
    repo_config = _resolve_model_repo_config(model_name)
    if repo_config is None:
        return ModelEnvironment(model_name=model_name, roots=roots)

    repo_path = resolve_model_repo(
        _resolve_repo_candidates(repo_config, roots),
        missing_message=repo_config.missing_message,
        required_subpath=repo_config.required_subpath,
        add_to_sys_path=repo_config.add_to_sys_path,
        prepend=repo_config.prepend,
    )

    for subdir in repo_config.optional_sys_path_subdirs:
        ensure_optional_subdir_in_sys_path(
            repo_path,
            subdir,
            prepend=repo_config.prepend,
        )

    return ModelEnvironment(
        model_name=model_name,
        roots=roots,
        repo_path=repo_path,
    )


def setup_model_runtime(model_name: str, file_path: str) -> ModelRuntime:
    """
    Load benchmark config/bootstrap state and resolve model repo setup together.

    This keeps FM entrypoints close to a single-line runtime bootstrap.
    """
    config_module = load_benchmark_config_module()
    model_env = setup_model_environment(model_name, file_path)
    return ModelRuntime(model_env=model_env, config_module=config_module)
