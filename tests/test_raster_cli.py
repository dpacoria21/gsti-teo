"""Uploading and direct inference must enforce one GeoTIFF input contract."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
import rasterio
from affine import Affine

from src.inference.raster import inspect_raster

ROOT = Path(__file__).resolve().parents[1]


def raster_fixture(tmp_path, resolution):
    path = tmp_path / "rgb.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=3,
        dtype="uint8",
        crs="EPSG:32719",
        transform=Affine(resolution, 0, 227000, 0, -resolution, 8185000),
    ) as output:
        output.write(np.random.default_rng(7).integers(0, 256, (3, 64, 64), dtype="uint8"))
    return path


def test_cli_returns_same_spatial_metadata_as_detector(tmp_path):
    path = raster_fixture(tmp_path, 0.5)
    expected = inspect_raster(path)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/inspect_raster.py"), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert json.loads(result.stdout) == expected


def test_upload_validation_rejects_pixels_unsuitable_for_individual_buildings(tmp_path):
    path = raster_fixture(tmp_path, 10)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/inspect_raster.py"), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 1
    assert "metros por píxel" in result.stderr
    assert not result.stdout.strip()
