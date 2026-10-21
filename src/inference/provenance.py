"""Bind source declarations to the actual raster without replacing measured facts."""

from __future__ import annotations

import json
from pathlib import Path

from .model import sha256_file
from .raster import inspect_raster


def inspect_image(image_path: Path, image_manifest: str | Path | None = None) -> dict:
    """Validate RGB/georeferencing and reject source metadata for another image."""
    image = inspect_raster(image_path)
    image["sha256"] = sha256_file(image_path)
    metadata = {}
    if image_manifest is None:
        sibling = image_path.parent / "manifest.json"
        if sibling.exists():
            image_manifest = sibling
    if image_manifest:
        metadata = json.loads(Path(image_manifest).read_text(encoding="utf-8-sig"))
        declared_sha = metadata.get("sha256") or metadata.get("image", {}).get("sha256")
        if declared_sha and declared_sha != image["sha256"]:
            raise ValueError("La procedencia adjunta corresponde a otra imagen (SHA256 distinto).")
    image.update(
        {
            "source_id": metadata.get("source_id", "user_geotiff"),
            "observed_at": metadata.get("observed_at") or metadata.get("capture_date"),
            "license": metadata.get("license", "No declarada por el usuario"),
            "provenance": metadata,
        }
    )
    # Provider credits and accuracy context accompany the map and exports.
    # Measured dimensions, hash and bounds cannot be overwritten by declarations.
    for field in (
        "title",
        "attribution",
        "source",
        "organization",
        "image_kind",
        "license_url",
        "accuracy_notes",
        "georeferencing",
    ):
        if field in metadata:
            image[field] = metadata[field]
    return image
