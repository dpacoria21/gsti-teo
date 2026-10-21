"""Publish a completed inference attempt without discarding the last good run.

The caller holds the destination's running lock. All staging and backups stay on
the same filesystem; the manifest is the final commit marker. Handled write
failures roll changed files back while leaving diagnostics/backups inspectable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

RESULT_FILES = ("buildings.geojson", "preview.png", "manifest.json")


def _replace(source: Path, destination: Path) -> None:
    source.replace(destination)


def publish_run(staging: Path, destination: Path) -> dict:
    """Validate, back up, then publish the three result files under the run lock."""
    if any(not (staging / name).is_file() for name in RESULT_FILES):
        raise ValueError("La nueva ejecución no produjo todos sus archivos de resultados.")
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("La nueva ejecución no confirmó resultados completos.")
    digest = hashlib.sha256((staging / "buildings.geojson").read_bytes()).hexdigest()
    if manifest.get("outputs", {}).get("buildings_sha256") != digest:
        raise ValueError("La nueva ejecución tiene un GeoJSON cuyo SHA256 no coincide.")

    backup = staging.parent / "previous"
    backup.mkdir(exist_ok=False)
    previous = {name for name in RESULT_FILES if (destination / name).exists()}
    # Finish every backup before replacing even the first output file.
    for name in previous:
        shutil.copy2(destination / name, backup / name)
    changed = []
    try:
        for name in RESULT_FILES:
            _replace(staging / name, destination / name)
            changed.append(name)
    except Exception:
        for name in reversed(changed):
            if name in previous:
                _replace(backup / name, destination / name)
            else:
                (destination / name).unlink()
        raise
    return manifest
