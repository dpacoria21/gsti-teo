"""Project paths shared by UI, services and the isolated inference process.

No files are opened and no frameworks are initialized when importing this module.
Paths are based on the repository location, never on the caller's working directory.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def environment_python(name: str, root: Path = PROJECT_ROOT) -> Path:
    """Locate Python in a project-owned virtual environment on the current OS."""
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return root / name / relative


def ml_python(root: Path = PROJECT_ROOT) -> Path:
    """Use an explicitly configured worker interpreter, or the local ML venv.

    Cloud installations can point CHULLO_ML_PYTHON at their active interpreter.
    Model execution still happens in a separate process from the interface.
    """
    configured = os.environ.get("CHULLO_ML_PYTHON", "").strip()
    if configured:
        # A Linux venv's python often symlinks to the base executable. Resolving
        # that link loses the venv and its installed dependencies in the worker.
        return Path(configured).expanduser().absolute()
    return environment_python(".venv-ml", root)
