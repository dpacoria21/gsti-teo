"""Read completed detections, verify integrity and retain full study-area footprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd

from ..data_loader import load_study_area, validate_layer


def load_detection(output_dir: str | Path, area_id: str = "chullo") -> dict:
    """Load only completed model outputs, retaining provenance and the pilot scope."""
    directory = Path(output_dir).resolve()
    if (directory / ".running.lock").exists():
        raise ValueError("La detección está ejecutándose; espere a que termine.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Esta ejecución todavía no está completa.")
    building_path = directory / "buildings.geojson"
    building_bytes = building_path.read_bytes()
    expected_sha = manifest.get("outputs", {}).get("buildings_sha256")
    if expected_sha and hashlib.sha256(building_bytes).hexdigest() != expected_sha:
        raise ValueError("El GeoJSON no corresponde al manifiesto de esta ejecución (SHA256 distinto).")
    raw = json.loads(building_bytes)
    if raw.get("type") != "FeatureCollection":
        raise ValueError("El detector debe devolver un GeoJSON FeatureCollection.")
    model_id = manifest.get("model", {}).get("id") or manifest.get("model_id")
    if manifest.get("model_id") and model_id != manifest["model_id"]:
        raise ValueError("El identificador del modelo no coincide dentro del manifiesto.")
    if model_id and any(
        feature.get("properties", {}).get("model_id", model_id) != model_id for feature in raw.get("features", [])
    ):
        raise ValueError("Las detecciones no corresponden al modelo registrado en el manifiesto.")
    if raw.get("features"):
        buildings = gpd.read_file(building_path)
        validate_layer(
            buildings, "Detecciones", "building_id", {"Polygon", "MultiPolygon"}, ("source_id", "confidence")
        )
        if not buildings["confidence"].between(0, 1).all():
            raise ValueError("Confianzas fuera del intervalo [0,1].")
        buildings = buildings.to_crs(4326)
        # GDAL may parse a date-only string as midnight. Preserve source precision.
        observed = {
            feature["properties"]["building_id"]: feature["properties"].get("observed_at")
            for feature in raw["features"]
        }
        buildings["observed_at"] = buildings["building_id"].map(observed)
    else:
        buildings = gpd.GeoDataFrame(
            {name: [] for name in ("building_id", "source_id", "observed_at", "review_status", "confidence")},
            geometry=[],
            crs=4326,
        )
    payload = load_study_area(area_id)
    raw_count = len(buildings)
    if manifest.get("count") != raw_count:
        raise ValueError("El conteo del manifiesto no coincide con las detecciones guardadas.")
    study_geometry = payload["area"].geometry.union_all()
    buildings = buildings.loc[buildings.intersects(study_geometry)].reset_index(drop=True)
    preview = directory / "preview.png"
    bounds = manifest.get("preview_bounds_wgs84")
    if preview.is_file() and (not isinstance(bounds, list) or len(bounds) != 2):
        raise ValueError("La previsualización no tiene coordenadas verificables.")
    image = manifest.get("image", {})
    model = manifest.get("model", {})
    metadata = dict(payload["metadata"])
    used_sources = set(payload["rivers"]["source_id"]) | {image.get("source_id")}
    metadata["sources"] = [source for source in metadata["sources"] if source["source_id"] in used_sources]
    metadata["sources"].append(
        {
            "source_id": manifest.get("source_id", "geoai_building_inference"),
            "title": "Detecciones ejecutadas con Mask R-CNN",
            "organization": (
                "Proyecto Chullo; ajuste supervisado sobre pesos de giswqs/GeoAI"
                if model.get("trained_locally")
                else "Proyecto Chullo; pesos de giswqs/GeoAI"
            ),
            "url": "https://huggingface.co/giswqs/geoai",
            "observed_at": image.get("observed_at"),
            "license": model.get("license", "MIT (pesos)"),
            "review_status": "Detección automática; no validación de campo ni uso residencial confirmado.",
            "notes": "Inferencia real sobre la imagen documentada; fecha de procesamiento no equivale a captura.",
            "model_id": model.get("id", manifest.get("model_id")),
            "model_sha256": model.get("sha256"),
        }
    )
    metadata["analysis_mode"] = "image_inference"
    metadata["inference"] = manifest
    metadata["selection"] = {
        "raw_count": raw_count,
        "selected_count": len(buildings),
        "area_id": area_id,
        "rule": "Intersección con sector piloto; contornos completos.",
    }
    return {
        "buildings": buildings,
        "manifest": manifest,
        "preview_path": str(preview) if preview.is_file() else None,
        "preview_bounds": bounds,
        "metadata": metadata,
        "raw_count": raw_count,
        "selected_count": len(buildings),
    }
