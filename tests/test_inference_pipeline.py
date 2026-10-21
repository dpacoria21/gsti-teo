"""Regression checks for data and geometry contracts; no mock detections in demos."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import Polygon

from src.inference.model import ensure_model, sha256_file
from src.inference.postprocess import candidates_from_masks, detection_frame
from src.inference.provenance import inspect_image
from src.inference.raster import RasterTile, read_tile
from src.settings import PROJECT_ROOT


@pytest.fixture
def rgb_raster(tmp_path):
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
        validity = np.full((128, 128), 255, dtype="uint8")
        validity[0, 0] = 0
        dst.write_mask(validity)
    return path


def test_importing_compatibility_api_never_imports_torch_or_downloads():
    code = """
import sys
import requests
def unexpected_network(*args, **kwargs):
    raise AssertionError('Import attempted network access')
requests.get = unexpected_network
import src.building_detection
assert 'torch' not in sys.modules
assert 'torchvision' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True, timeout=30)


def test_corrupt_local_checkpoint_is_rejected_before_any_download(tmp_path, monkeypatch):
    path = tmp_path / "bad.pth"
    path.write_bytes(b"not the pinned pretrained weights")

    def unexpected_download(*args, **kwargs):
        pytest.fail("A corrupt local checkpoint must fail, not fetch another model")

    monkeypatch.setattr("src.inference.model.requests.get", unexpected_download)
    with pytest.raises(ValueError, match="SHA256"):
        ensure_model(path)


def test_provenance_for_different_image_is_rejected(rgb_raster):
    metadata = rgb_raster.parent / "manifest.json"
    metadata.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 distinto"):
        inspect_image(rgb_raster)


def test_provenance_keeps_credits_but_cannot_replace_measured_raster_facts(rgb_raster):
    actual_hash = sha256_file(rgb_raster)
    metadata = {
        "sha256": actual_hash,
        "width": 999,
        "crs": "EPSG:4326",
        "source_id": "test_source",
        "capture_date": "2026-03-04",
        "attribution": "Image provider",
        "accuracy_notes": "Approximate alignment",
    }
    (rgb_raster.parent / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    image = inspect_image(rgb_raster)
    assert image["width"] == 128
    assert image["crs"] == "EPSG:32719"
    assert image["sha256"] == actual_hash
    assert image["observed_at"] == "2026-03-04"
    assert image["attribution"] == "Image provider"
    assert image["provenance"] == metadata


def test_tile_padding_preserves_pixels_mask_and_window_coordinates(rgb_raster):
    with rasterio.open(rgb_raster) as src:
        tile = read_tile(src, 0, 0, 512)
        np.testing.assert_array_equal(tile.pixels[:, :128, :128], src.read().astype("float32") / 255)
        assert tile.transform == src.transform
    assert tile.pixels.shape == (3, 512, 512)
    assert not tile.pixels[:, 128:, :].any()
    assert not tile.pixels[:, :, 128:].any()
    assert tile.valid.shape == (128, 128)
    assert not tile.valid[0, 0]
    assert tile.window == Window(0, 0, 128, 128)


def test_mask_nodata_area_and_internal_edge_are_preserved():
    validity = np.ones((10, 10), dtype=bool)
    validity[3:5, 3:5] = False
    tile = RasterTile(
        np.empty((3, 10, 10)),
        validity,
        Window(10, 10, 10, 10),
        Affine(0.5, 0, 227005, 0, -0.5, 8184995),
        30,
        30,
    )
    masks = np.ones((2, 10, 10), dtype="float32")
    masks[1] = 0
    masks[1, 8, 8] = 1  # Below the minimum footprint area.
    candidates = candidates_from_masks(
        np.array([0.8, 0.9]),
        masks,
        tile,
        3,
        Transformer.from_crs("EPSG:32719", "EPSG:32719", always_xy=True),
        0.5,
        10,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["area_m2"] == pytest.approx(24)
    assert len(candidate["geometry"].interiors) == 1
    assert candidate["tile_edge_clipped"] is True
    assert candidate["image_edge_clipped"] is False
    assert candidate["geometry"].bounds == (227005, 8184990, 227010, 8184995)


def test_empty_detections_keep_schema_and_ids_ignore_ring_start():
    empty = detection_frame([], "image_test", None, "EPSG:32719")
    assert empty.empty
    assert empty.crs.to_epsg() == 4326
    assert "building_id" in empty.columns
    coordinates = [(227000, 8185000), (227010, 8185000), (227010, 8184990), (227000, 8184990)]
    item = {
        "confidence": 0.8,
        "area_m2": 100,
        "tile_edge_clipped": False,
        "image_edge_clipped": False,
        "tile_index": 1,
    }
    frames = [
        detection_frame([{**item, "geometry": Polygon(ring)}], "image_test", None, "EPSG:32719")
        for ring in (coordinates, coordinates[1:] + coordinates[:1])
    ]
    assert frames[0].iloc[0].building_id == frames[1].iloc[0].building_id
    assert frames[0].iloc[0].geometry.equals(frames[1].iloc[0].geometry)
