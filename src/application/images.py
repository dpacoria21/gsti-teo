"""Local image inputs and file fingerprints, independent of the user interface."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.application.session_workspace import session_output_root, validate_session_identifier
from src.settings import PROJECT_ROOT, environment_python


def bundled_image(area_id: str, root: Path = PROJECT_ROOT) -> tuple[Path | None, Path | None, dict[str, Any]]:
    """Locate an area's raster and supplied provenance, including portable bundles."""
    manifest_path = root / "data" / "imagery" / area_id / "manifest.json"
    if not manifest_path.exists():
        return None, None, {}
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        named = record.get("raster_path") or record.get("path") or f"{area_id}_rgb.tif"
        candidate = Path(named)
        candidates = [candidate] if candidate.is_absolute() else [root / candidate, manifest_path.parent / candidate]
        candidates.append(manifest_path.parent / f"{area_id}_rgb.tif")
        raster_path = next((path for path in candidates if path.is_file()), None)
        return raster_path, manifest_path, record
    except (OSError, ValueError, TypeError):
        return None, None, {}


def stored_upload(
    content: bytes,
    original_filename: str,
    area_id: str,
    root: Path = PROJECT_ROOT,
    *,
    session_id: str | None = None,
) -> tuple[Path, Path, dict[str, Any], Path]:
    """Persist by hash within the caller's session; preserve CLI storage defaults."""
    digest = hashlib.sha256(content).hexdigest()
    uploads_root = root / "data" / "uploads"
    output_root = root / "outputs" / "inference"
    if session_id is not None:
        uploads_root = uploads_root / "sessions" / validate_session_identifier(session_id)
        output_root = session_output_root(session_id, root)
    folder = uploads_root / digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    raster = folder / "image.tif"
    if not raster.exists():
        raster.write_bytes(content)
    manifest_path = folder / "manifest.json"
    manifest = {
        "image_id": f"upload_{digest[:20]}",
        "source_id": f"user_upload_{digest[:20]}",
        "title": "Imagen proporcionada por el usuario",
        "original_filename": original_filename,
        "raster_path": str(raster),
        "observed_at": None,
        "capture_date": None,
        "license": "Condiciones declaradas por quien aporta la imagen; no verificadas",
        "attribution": "Imagen proporcionada por el usuario",
        "sha256": digest,
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return raster, manifest_path, manifest, output_root / f"{area_id}_upload_{digest[:20]}"


def data_fingerprint(root: Path = PROJECT_ROOT) -> tuple[tuple[str, int], ...]:
    """Version cache entries when prepared configuration or vector files change."""
    return tuple(
        (str(path.relative_to(root)), path.stat().st_mtime_ns)
        for folder in (root / "config", root / "data")
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".geojson", ".gpkg"}
        and not path.is_relative_to(root / "data" / "uploads")
    )


def detection_fingerprint(output_dir: Path) -> tuple[tuple[str, int], ...]:
    """Version a completed detection from its local output files."""
    return tuple(
        (path.name, path.stat().st_mtime_ns)
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix in {".json", ".geojson", ".png"}
    )


def imagery_download_available(root: Path = PROJECT_ROOT) -> bool:
    """Whether the optional official-imagery downloader can be launched locally."""
    return environment_python(".venv-ml", root).is_file() and (root / "scripts/download_chullo_imagery.py").is_file()


def retry_chullo_imagery(root: Path = PROJECT_ROOT, timeout: float = 45) -> subprocess.CompletedProcess[str]:
    """Run the optional acquisition script only following an explicit UI action."""
    return subprocess.run(
        [str(environment_python(".venv-ml", root)), str(root / "scripts/download_chullo_imagery.py")],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
