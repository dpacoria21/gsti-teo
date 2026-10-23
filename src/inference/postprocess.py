"""Instance-mask geometry and cross-tile suppression; no model dependencies."""

from __future__ import annotations

import hashlib

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely import make_valid, normalize
from shapely.geometry import Polygon, shape
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union

from .model import MODEL_ID
from .raster import RasterTile

SOURCE_ID = "geoai_building_inference"


def candidates_from_masks(
    scores: np.ndarray,
    masks: np.ndarray,
    tile: RasterTile,
    tile_index: int,
    metric_transformer,
    mask_threshold: float,
    min_area_m2: float,
) -> list[dict]:
    """Restore selected masks to full world coordinates and apply area/edge rules."""
    column, row = int(tile.window.col_off), int(tile.window.row_off)
    width, height = int(tile.window.width), int(tile.window.height)
    candidates = []
    for score, mask in zip(scores, masks):
        binary = (mask >= mask_threshold) & tile.valid
        if not binary.any():
            continue
        touches_internal = bool(
            (column > 0 and binary[:, 0].any())
            or (row > 0 and binary[0, :].any())
            or (column + width < tile.image_width and binary[:, -1].any())
            or (row + height < tile.image_height and binary[-1, :].any())
        )
        touches_image = bool(
            (column == 0 and binary[:, 0].any())
            or (row == 0 and binary[0, :].any())
            or (column + width == tile.image_width and binary[:, -1].any())
            or (row + height == tile.image_height and binary[-1, :].any())
        )
        geometry = mask_to_geometry(binary, tile.transform)
        area = transform_geometry(metric_transformer.transform, geometry).area
        if geometry.is_empty or area < min_area_m2:
            continue
        candidates.append(
            {
                "geometry": geometry,
                "confidence": float(score),
                "area_m2": area,
                "tile_edge_clipped": touches_internal,
                "image_edge_clipped": touches_image,
                "tile_index": tile_index,
            }
        )
    return candidates


def detection_frame(
    candidates: list[dict], image_id: str, observed_at: str | None, native_crs, model_id: str = MODEL_ID
) -> gpd.GeoDataFrame:
    """Give each footprint a stable identity and serialize coordinates as WGS84."""
    columns = [
        "building_id",
        "source_id",
        "observed_at",
        "review_status",
        "confidence",
        "model_id",
        "image_id",
        "area_m2",
        "tile_edge_clipped",
        "image_edge_clipped",
        "tile_index",
        "geometry",
    ]
    records = []
    for item in candidates:
        identity = hashlib.sha256((image_id + normalize(item["geometry"]).wkb_hex).encode()).hexdigest()[:16]
        records.append(
            {
                "building_id": "ai_" + identity,
                "source_id": SOURCE_ID,
                "observed_at": observed_at,
                "review_status": "Sin revisión manual · inferencia IA",
                "model_id": model_id,
                "image_id": image_id,
                **item,
            }
        )
    return gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=native_crs).to_crs("EPSG:4326")


def polygonal(geometry):
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    members = [part for part in getattr(geometry, "geoms", []) if part.geom_type in {"Polygon", "MultiPolygon"}]
    return unary_union(members) if members else Polygon()


def mask_to_geometry(binary_mask: np.ndarray, transform):
    """Preserve instance-mask pixel boundaries and holes, rather than a box."""
    pieces = [
        shape(item)
        for item, value in shapes(binary_mask.astype("uint8"), mask=binary_mask.astype(bool), transform=transform)
        if value == 1
    ]
    return polygonal(unary_union(pieces)) if pieces else Polygon()


def suppress_duplicates(
    candidates: list[dict], iou_threshold: float = 0.35, containment_threshold: float = 0.7
) -> list[dict]:
    """Geometric NMS across overlaps. Prefer instances not cut at an internal tile edge."""
    candidates = sorted(
        candidates, key=lambda record: (record["tile_edge_clipped"], -record["confidence"], -record["geometry"].area)
    )
    kept = []
    for candidate in candidates:
        geometry = candidate["geometry"]
        duplicate = False
        for previous in kept:
            other = previous["geometry"]
            if not geometry.intersects(other):
                continue
            intersection = geometry.intersection(other).area
            union = geometry.area + other.area - intersection
            if (
                intersection / max(union, 1e-20) >= iou_threshold
                or intersection / max(min(geometry.area, other.area), 1e-20) >= containment_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept
