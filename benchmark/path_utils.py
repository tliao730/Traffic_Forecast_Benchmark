import os
import sys
from typing import Iterable, List, Optional


def normalize_candidate_paths(candidates: Iterable[Optional[str]]) -> List[str]:
    """Expand, absolutize, and deduplicate non-empty path candidates."""
    normalized: List[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        abs_candidate = os.path.abspath(os.path.expanduser(candidate))
        if abs_candidate in seen:
            continue
        normalized.append(abs_candidate)
        seen.add(abs_candidate)
    return normalized


def find_first_existing_path(
    candidates: Iterable[Optional[str]],
    required_subpath: Optional[str] = None,
) -> tuple[Optional[str], List[str]]:
    """
    Return the first existing candidate path and the normalized candidate list.
    If required_subpath is provided, the path is considered valid only if that
    child path exists under the candidate directory.
    """
    normalized_candidates = normalize_candidate_paths(candidates)
    for candidate in normalized_candidates:
        check_path = (
            os.path.join(candidate, required_subpath)
            if required_subpath
            else candidate
        )
        if os.path.exists(check_path):
            return candidate, normalized_candidates
    return None, normalized_candidates


def ensure_path_in_sys_path(path: str, prepend: bool = False) -> str:
    """Insert a path into sys.path once; return its real path."""
    real_path = os.path.realpath(path)
    if real_path not in sys.path:
        if prepend:
            sys.path.insert(0, real_path)
        else:
            sys.path.append(real_path)
    return real_path
