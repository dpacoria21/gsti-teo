"""Validate uploads with the same raster contract used by the inference pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect(path: str | Path) -> dict:
    from src.inference.raster import inspect_raster

    return inspect_raster(path)


def main() -> int:
    # The UI consumes this subprocess as UTF-8 on Windows and Linux alike.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        if len(sys.argv) != 2:
            raise ValueError("Uso: inspect_raster.py archivo.tif")
        print(json.dumps(inspect(sys.argv[1])))
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
