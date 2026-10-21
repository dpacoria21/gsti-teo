"""Spatial safety tests; no pretrained output is replaced with fake detections."""

import numpy as np
import pytest
from shapely.geometry import box

# The map environment deliberately does not contain the raster/ML dependency set.
pytest.importorskip("rasterio")
from affine import Affine

from src.building_detection import inspect_raster, mask_to_geometry, suppress_duplicates, tile_starts


def test_tiles_cover_edges_without_repeated_last_window():
    for length in (64, 512, 513, 1024, 1200):
        starts = tile_starts(length, 512, 128)
        assert len(starts) == len(set(starts))
        covered = np.zeros(length, dtype=bool)
        for start in starts:
            covered[start : min(length, start + 512)] = True
        assert covered.all()


def test_mask_preserves_hole_and_l_shape_in_world_coordinates():
    mask = np.ones((10, 10), dtype="uint8")
    mask[3:6, 3:6] = 0
    mask[7:, 7:] = 0
    geometry = mask_to_geometry(mask, Affine(0.5, 0, 100, 0, -0.5, 200))
    assert geometry.is_valid
    assert geometry.area == pytest.approx(mask.sum() * 0.25)
    assert len(geometry.interiors) == 1
    assert geometry.area < geometry.envelope.area
    assert geometry.bounds == (100, 195, 105, 200)


def test_nms_prefers_complete_instance_and_keeps_nearby_building():
    candidates = [
        {"geometry": box(0, 0, 10, 10), "confidence": 0.8, "tile_edge_clipped": False},
        {"geometry": box(0, 0, 6, 10), "confidence": 0.95, "tile_edge_clipped": True},
        {"geometry": box(11, 0, 21, 10), "confidence": 0.7, "tile_edge_clipped": False},
    ]
    result = suppress_duplicates(candidates)
    assert len(result) == 2
    assert result[0]["geometry"].equals(box(0, 0, 10, 10))


def test_geotiff_validation_rejects_missing_crs(tmp_path):
    import rasterio

    path = tmp_path / "unreferenced.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=3,
        dtype="uint8",
        transform=Affine(0.5, 0, 227000, 0, -0.5, 8185000),
    ) as dst:
        dst.write(np.random.default_rng(7).integers(0, 256, (3, 128, 128), dtype="uint8"))
    with pytest.raises(ValueError, match="CRS"):
        inspect_raster(path)


def test_geotiff_validation_rejects_coarse_pixels(tmp_path):
    import rasterio

    path = tmp_path / "coarse.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=3,
        dtype="uint8",
        crs="EPSG:32719",
        transform=Affine(10, 0, 227000, 0, -10, 8185000),
    ) as dst:
        dst.write(np.random.default_rng(7).integers(0, 256, (3, 128, 128), dtype="uint8"))
    with pytest.raises(ValueError, match="metros por píxel"):
        inspect_raster(path)


def test_geotiff_metadata_reports_real_ground_resolution(tmp_path):
    import rasterio

    path = tmp_path / "rgb.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=3,
        dtype="uint8",
        crs="EPSG:32719",
        transform=Affine(0.5, 0, 227000, 0, -0.5, 8185000),
    ) as dst:
        dst.write(np.random.default_rng(7).integers(0, 256, (3, 128, 128), dtype="uint8"))
    metadata = inspect_raster(path)
    assert metadata["resolution_m_approx"][0] == pytest.approx(0.5, abs=0.005)
    assert metadata["preview_bounds_wgs84"][0][0] < metadata["preview_bounds_wgs84"][1][0]
