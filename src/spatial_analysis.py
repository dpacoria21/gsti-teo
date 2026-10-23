"""Distancias mínimas métricas y relaciones de cada edificación con todos los cauces."""

from __future__ import annotations

from numbers import Real

import geopandas as gpd
import numpy as np
import pandas as pd

from .data_loader import metric_crs, validate_layer, validate_rivers
from .proximity import BOUNDARY_TOLERANCE_M as BOUNDARY_TOLERANCE_M
from .proximity import (
    PROXIMITY_LABELS,
    classify_distances,
    validate_thresholds,
    within_threshold,
)

DISTANCE_REFERENCES = {"axis": "eje cartografiado", "channel_polygon": "polígono del cauce"}
INSIDE_STATUS = "Dentro de banda analítica"
OUTSIDE_STATUS = "Fuera de banda analítica"


def analyze_proximity(
    buildings: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    band_m: float = 25,
    analysis_crs: str = "EPSG:32719",
    red_band_m: float | None = None,
) -> dict:
    """Return summary (one building), relations (all pairs), and display buffers.

    Distances use original full geometries after projection. A line represents the
    cartographed axis. A channel polygon measures separation, zero on intersection.
    With red_band_m, band_m is the OUTER yellow limit; yellow width is their
    difference. The nearest river determines the most restrictive proximity class.
    Proximity alone does not assess hazard, permission to build, or an official
    marginal strip.
    """
    if isinstance(band_m, bool) or not isinstance(band_m, Real) or not np.isfinite(band_m) or not 0 <= band_m <= 1000:
        raise ValueError("La banda analítica debe ser un número entre 0 y 1000 metros.")
    if red_band_m is not None:
        red_band_m, band_m = validate_thresholds(red_band_m, band_m)
    validate_layer(buildings, "Edificaciones", "building_id", {"Polygon", "MultiPolygon"}, ("source_id",))
    validate_rivers(rivers)
    crs = metric_crs(analysis_crs)
    projected_buildings = buildings.to_crs(crs).reset_index(drop=True)
    projected_rivers = rivers.to_crs(crs).reset_index(drop=True)
    if not np.isfinite(projected_buildings.total_bounds).all() or not np.isfinite(projected_rivers.total_bounds).all():
        raise ValueError("La transformación produjo coordenadas no finitas. Revise los CRS y la cobertura.")
    distances = np.column_stack(
        [projected_buildings.geometry.distance(river).to_numpy() for river in projected_rivers.geometry]
    )
    if not np.isfinite(distances).all():
        raise ValueError("No se pudieron calcular todas las distancias. Revise las geometrías y el CRS.")
    within = within_threshold(distances, float(band_m))
    closest = distances.argmin(axis=1)
    nearest_rivers = projected_rivers.iloc[closest].reset_index(drop=True)
    summary = buildings.to_crs("EPSG:4326").reset_index(drop=True).copy()
    summary["distance_m"] = distances[np.arange(len(summary)), closest]
    summary["nearest_river_id"] = nearest_rivers["river_id"].to_numpy()
    summary["nearest_river_name"] = nearest_rivers["name"].to_numpy()
    summary["river_source_id"] = nearest_rivers["source_id"].to_numpy()
    summary["distance_reference"] = nearest_rivers["geometry_role"].map(DISTANCE_REFERENCES).to_numpy()
    summary["within_band"] = within.any(axis=1)
    summary["status"] = np.where(summary["within_band"], INSIDE_STATUS, OUTSIDE_STATUS)
    summary["hazard_status"] = "No evaluado"
    summary["margin_status"] = "No evaluado"
    summary["band_m"] = float(band_m)
    summary["analysis_crs"] = crs.to_string()
    summary["rivers_within_band"] = within.sum(axis=1)
    detail = []
    for index, river in projected_rivers.iterrows():
        detail.append(
            pd.DataFrame(
                {
                    "building_id": summary["building_id"],
                    "river_id": river["river_id"],
                    "river_name": river["name"],
                    "distance_m": distances[:, index],
                    "distance_reference": DISTANCE_REFERENCES[river["geometry_role"]],
                    "band_m": float(band_m),
                    "within_band": within[:, index],
                    "status": np.where(within[:, index], INSIDE_STATUS, OUTSIDE_STATUS),
                    "source_id": summary["source_id"],
                    "river_source_id": river["source_id"],
                    "analysis_crs": crs.to_string(),
                }
            )
        )
    relations = pd.concat(detail, ignore_index=True)
    if red_band_m is not None:
        for table in (summary, relations):
            table["proximity_class"] = classify_distances(table["distance_m"], red_band_m, band_m)
            table["proximity_label"] = table["proximity_class"].map(PROXIMITY_LABELS)
            table["status"] = table["proximity_label"]
            table["red_band_m"] = red_band_m
            table["yellow_width_m"] = band_m - red_band_m
            table["hazard_status"] = "No evaluado"
            table["margin_status"] = "No evaluado"
            table["construction_status"] = "No evaluado"
    bands = projected_rivers.copy()
    bands["distance_reference"] = bands["geometry_role"].map(DISTANCE_REFERENCES)
    bands["band_m"] = float(band_m)
    bands.geometry = bands.geometry.buffer(float(band_m))
    bands = bands.loc[~bands.geometry.is_empty].to_crs("EPSG:4326")
    result = {"summary": summary, "relations": relations, "bands": bands}
    if red_band_m is not None:
        # Subtract ALL red buffers, so yellow cannot cover another river's red zone.
        red_geometry = projected_rivers.geometry.buffer(red_band_m).union_all()
        outer_geometry = projected_rivers.geometry.buffer(band_m).union_all()
        zones = gpd.GeoDataFrame(
            {
                "proximity_class": ["red", "yellow"],
                "proximity_label": [PROXIMITY_LABELS["red"], PROXIMITY_LABELS["yellow"]],
                "red_band_m": [red_band_m, red_band_m],
                "yellow_width_m": [band_m - red_band_m] * 2,
                "band_m": [band_m, band_m],
            },
            geometry=[red_geometry, outer_geometry.difference(red_geometry)],
            crs=crs,
        )
        result["traffic_zones"] = zones.loc[~zones.geometry.is_empty].to_crs("EPSG:4326")
    return result
