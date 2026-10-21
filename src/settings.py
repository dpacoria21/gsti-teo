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
