"""Carga de áreas preparadas y validación explícita de sus datos geográficos."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS

from .settings import PROJECT_ROOT


def validate_layer(
    layer: gpd.GeoDataFrame,
    label: str,
    id_column: str,
    allowed_types: set[str],
    required_columns: tuple[str, ...] = (),
) -> None:
    """Reject unusable inputs instead of guessing coordinates or repairing silently."""
    if not isinstance(layer, gpd.GeoDataFrame):
        raise ValueError(f"{label}: se requiere una capa geográfica GeoDataFrame.")
    if layer.empty:
        raise ValueError(f"{label}: la capa está vacía; no es posible evaluar distancias.")
    missing = {id_column, *required_columns} - set(layer.columns)
    if missing:
        raise ValueError(f"{label}: faltan campos obligatorios: {', '.join(sorted(missing))}.")
    if layer.crs is None:
        raise ValueError(f"{label}: falta el CRS. Declare el sistema original antes de analizar.")
    identifiers = layer[id_column]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label}: hay identificadores vacíos en {id_column}.")
    if identifiers.astype(str).duplicated().any():
        raise ValueError(f"{label}: hay identificadores duplicados en {id_column}.")
    for column in required_columns:
        if layer[column].isna().any() or layer[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}: hay valores vacíos en {column}.")
    try:
        geometry = layer.geometry
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label}: falta la columna geométrica activa.") from exc
    if geometry.isna().any() or geometry.is_empty.any():
        raise ValueError(f"{label}: hay geometrías vacías o ausentes.")
    if not geometry.geom_type.isin(allowed_types).all():
        raise ValueError(f"{label}: se admiten geometrías {', '.join(sorted(allowed_types))}.")
    invalid = ~geometry.is_valid
    if invalid.any():
        ids = ", ".join(layer.loc[invalid, id_column].astype(str).head(5))
        raise ValueError(f"{label}: geometrías inválidas ({ids}). Revise los datos originales.")
    if not np.isfinite(layer.total_bounds).all():
        raise ValueError(f"{label}: hay coordenadas no finitas.")
    if CRS.from_user_input(layer.crs).is_geographic:
        min_x, min_y, max_x, max_y = layer.total_bounds
        if min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90:
            raise ValueError(f"{label}: coordenadas fuera del rango del CRS geográfico declarado.")


def validate_rivers(rivers: gpd.GeoDataFrame) -> None:
    validate_layer(
        rivers,
        "Cauces",
        "river_id",
        {"LineString", "MultiLineString", "Polygon", "MultiPolygon"},
        ("name", "geometry_role", "source_id"),
    )
    expected = {
        "axis": {"LineString", "MultiLineString"},
        "channel_polygon": {"Polygon", "MultiPolygon"},
    }
    for _, river in rivers.iterrows():
        role = river["geometry_role"]
        if role not in expected or river.geometry.geom_type not in expected[role]:
            raise ValueError(
                f"Cauce {river['river_id']}: geometry_role debe ser 'axis' para líneas "
                "o 'channel_polygon' para polígonos."
            )


def metric_crs(value: str | CRS) -> CRS:
    try:
        crs = CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError("El CRS de análisis no es válido.") from exc
    if not crs.is_projected or len(crs.axis_info) < 2:
        raise ValueError("El CRS de análisis debe ser proyectado y tener unidades en metros.")
    if any(abs(axis.unit_conversion_factor - 1.0) > 1e-10 for axis in crs.axis_info[:2]):
        raise ValueError("El CRS de análisis debe usar metros, no grados ni pies.")
    return crs


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer {path.name}: {exc}") from exc


def _read_layer(path: Path) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path)
    except Exception as exc:
        raise ValueError(f"No se pudo cargar la capa {path.name}: {exc}") from exc


def load_study_area(area_id: str = "chullo", root: str | Path | None = None) -> dict:
    """Read a configured study area; return independent WGS84 layers and provenance."""
    root_path = Path(root) if root is not None else PROJECT_ROOT
    configuration = _read_json(root_path / "config" / "study_areas.json")
    candidates = configuration.get("study_areas", [])
    matches = [area for area in candidates if area.get("id") == area_id]
    if len(matches) != 1:
        raise ValueError(f"El área '{area_id}' no está configurada de forma única.")
    config = matches[0]
    if not config.get("path") or not config.get("analysis_crs"):
        raise ValueError("El área necesita path y analysis_crs en su configuración.")
    metric_crs(config["analysis_crs"])
    folder = root_path / config["path"]
    buildings = _read_layer(folder / "buildings.geojson")
    rivers = _read_layer(folder / "rivers.geojson")
    area_path = folder / "study_area.geojson"
    if not area_path.exists():
        area_path = folder / "area.geojson"
    area = _read_layer(area_path)
    validate_layer(buildings, "Edificaciones", "building_id", {"Polygon", "MultiPolygon"}, ("source_id",))
    validate_rivers(rivers)
    validate_layer(area, "Área de estudio", "area_id", {"Polygon", "MultiPolygon"})
    for field in ("observed_at", "review_status"):
        if field not in buildings:
            raise ValueError(f"Edificaciones: falta el campo de procedencia {field}.")
    sources = _read_json(root_path / "data" / "sources.json")
    if not isinstance(sources.get("sources"), list):
        raise ValueError("sources.json debe contener una lista 'sources'.")
    source_ids = [source.get("source_id") for source in sources["sources"]]
    if None in source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("sources.json contiene identificadores ausentes o duplicados.")
    used_sources = set(buildings["source_id"]) | set(rivers["source_id"])
    if used_sources - set(source_ids):
        raise ValueError("Hay capas cuya fuente no figura en data/sources.json.")
    return {
        "buildings": buildings.to_crs("EPSG:4326"),
        "rivers": rivers.to_crs("EPSG:4326"),
        "area": area.to_crs("EPSG:4326"),
        "metadata": sources,
        "config": config,
    }
