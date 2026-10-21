"""Per-browser-session storage without coupling file ownership to Streamlit."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.settings import PROJECT_ROOT

SESSION_KEY = "_workspace_session_id"
RESULTS_KEY = "_session_detection_results"


def session_identifier(state: MutableMapping[str, Any]) -> str:
    """Create one unpredictable namespace for the current browser session."""
    if SESSION_KEY not in state:
        state[SESSION_KEY] = uuid4().hex
    return validate_session_identifier(state[SESSION_KEY])


def validate_session_identifier(value: str) -> str:
    """Do not allow a session identifier to escape its assigned directory."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("Identificador de sesión no válido.")
    return value


def session_output_root(session_id: str, root: Path = PROJECT_ROOT) -> Path:
    return root / "outputs" / "inference" / "sessions" / validate_session_identifier(session_id)


def new_detection_output(session_id: str, root: Path = PROJECT_ROOT) -> Path:
    """Every attempt gets its own output; bundled demos are never writable targets."""
    return session_output_root(session_id, root) / uuid4().hex


def selected_result(state: MutableMapping[str, Any], input_id: str, fallback: Path) -> Path:
    """Only this session can select its completed result for a particular image."""
    saved = state.get(RESULTS_KEY, {}).get(input_id)
    if saved is not None and (Path(saved) / "manifest.json").is_file():
        return Path(saved)
    return fallback


def remember_result(state: MutableMapping[str, Any], input_id: str, output_dir: Path) -> None:
    """Call only after a worker has confirmed publication of a completed run."""
    results = dict(state.get(RESULTS_KEY, {}))
    results[input_id] = str(output_dir)
    state[RESULTS_KEY] = results
