"""GeoTIFF validation, deterministic tiling and georeferenced map previews."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import Geod, Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterTile:
    """One RGB window and the spatial metadata required to restore its masks."""

    pixels: np.ndarray
    valid: np.ndarray
    window: Window
    transform: Affine
    image_width: int
    image_height: int


def read_tile(src, column: int, row: int, tile_size: int) -> RasterTile:
    """Read RGB in [0, 1], padding small edge windows without changing their CRS."""
    width, height = min(tile_size, src.width - column), min(tile_size, src.height - row)
    window = Window(column, row, width, height)
    rgb = src.read((1, 2, 3), window=window)
    valid = src.dataset_mask(window=window) > 0
    pixels = np.zeros((3, tile_size, tile_size), dtype="float32")
    pixels[:, :height, :width] = rgb.astype("float32") / 255.0
    return RasterTile(pixels, valid, window, src.window_transform(window), src.width, src.height)


def tile_starts(length: int, size: int, overlap: int) -> list[int]:
    if length <= 0 or size <= 0 or not 0 <= overlap < size:
        raise ValueError("Dimensiones o solapamiento de mosaicos inválidos.")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size - overlap))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def inspect_raster(path: str | Path) -> dict:
    """Validate georeferencing, RGB byte range and useful building-scale imagery."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("No se encuentra el GeoTIFF de entrada.")
    with rasterio.open(path) as src:
        if src.driver != "GTiff":
            raise ValueError("La entrada debe ser un GeoTIFF.")
        if src.crs is None:
            raise ValueError("El GeoTIFF necesita CRS y georreferenciación.")
        if src.count not in (3, 4) or any(t != "uint8" for t in src.dtypes[:3]):
            raise ValueError("Se requieren 3 bandas RGB uint8 (o RGB más alfa).")
        interpretations = [x.name for x in src.colorinterp[:3]]
        if interpretations not in (["red", "green", "blue"], ["undefined"] * 3):
            raise ValueError("Las bandas 1, 2 y 3 deben estar ordenadas rojo, verde y azul.")
        if min(src.width, src.height) < 64 or max(src.width, src.height) > 8192 or src.width * src.height > 16_000_000:
            raise ValueError(
                "Use una imagen de al menos 64 px por lado, hasta 8192 px por lado y 16 millones de píxeles."
            )
        if not np.isfinite(tuple(src.transform)).all() or abs(src.transform.determinant) < 1e-18:
            raise ValueError("La transformación geográfica no es válida.")
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        if not all(math.isfinite(x) for x in (west, south, east, north)) or not (
            -180 <= west < east <= 180 and -90 <= south < north <= 90
        ):
            raise ValueError("La imagen tiene límites geográficos inválidos.")
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        x, y = src.width / 2, src.height / 2
        center = transformer.transform(*(src.transform * (x, y)))
        x_neighbor = transformer.transform(*(src.transform * (x + 1, y)))
        y_neighbor = transformer.transform(*(src.transform * (x, y + 1)))
        geod = Geod(ellps="WGS84")
        resolution = [abs(geod.inv(*center, *neighbor)[2]) for neighbor in (x_neighbor, y_neighbor)]
        if min(resolution) < 0.05 or max(resolution) > 2.0:
            raise ValueError(
                "La imagen debe tener entre 0.05 y 2 metros por píxel. Sentinel-2 de 10 m no permite distinguir viviendas individuales."
            )
        sample = src.read((1, 2, 3), out_shape=(3, min(256, src.height), min(256, src.width)))
        if np.ptp(sample) < 5:
            raise ValueError("La imagen no contiene contraste suficiente para analizar edificaciones.")
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string(),
            "transform": list(src.transform)[:6],
            "resolution_m_approx": resolution,
            "bounds_wgs84": [west, south, east, north],
            "preview_bounds_wgs84": [[south, west], [north, east]],
            "rgb_bands": [1, 2, 3],
            "size_bytes": path.stat().st_size,
        }


def save_preview(image_path: Path, output_path: Path) -> list[list[float]]:
    """Reproject actual RGB pixels to north-up WGS84 for a correct Leaflet overlay."""
    with rasterio.open(image_path) as src:
        affine, width, height = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        scale = min(1.0, 2048 / max(width, height))
        if scale < 1:
            new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
            affine = affine * affine.scale(width / new_width, height / new_height)
            width, height = new_width, new_height
        rgb = np.zeros((3, height, width), dtype="uint8")
        for band in range(3):
            reproject(
                source=rasterio.band(src, band + 1),
                destination=rgb[band],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=affine,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
        alpha = np.zeros((height, width), dtype="uint8")
        reproject(
            source=src.dataset_mask(),
            destination=alpha,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=affine,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
        Image.fromarray(np.dstack((rgb.transpose(1, 2, 0), alpha))).save(output_path)
        west, north = affine * (0, 0)
        east, south = affine * (width, height)
        return [[south, west], [north, east]]
