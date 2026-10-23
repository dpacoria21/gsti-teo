"""Synthetic loader fixtures only: no demo detections, images or model inference."""

import hashlib
import json

import pytest
from shapely.geometry import box, mapping

from src.data_loader import load_study_area
from src.detection_client import load_detection


@pytest.fixture(scope="module")
def pilot_bounds():
    return load_study_area("chullo")["config"]["bbox"]


def synthetic_feature(identifier, geometry, confidence=0.91):
    return {
        "type": "Feature",
        "properties": {
            "building_id": identifier,
            "source_id": "test_inference",
            "confidence": confidence,
            "observed_at": None,
            "review_status": "Synthetic unit-test geometry; not a model prediction",
        },
        "geometry": mapping(geometry),
    }


@pytest.fixture
def completed_run(tmp_path, pilot_bounds):
    """Create a reproducible pair of test GeoJSON and matching run manifest."""
    west, south, east, north = pilot_bounds
    center_x, center_y = (west + east) / 2, (south + north) / 2
    default = synthetic_feature("test_inside", box(center_x, center_y, center_x + 0.0001, center_y + 0.0001))

    def write_run(features=None, **manifest_changes):
        if features is None:
            features = [default]
        encoded = json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")
        (tmp_path / "buildings.geojson").write_bytes(encoded)
        manifest = {
            "status": "completed",
            "count": len(features),
            "image": {"observed_at": None, "image_id": "synthetic_test_image"},
            "model": {"id": "synthetic_test_model", "license": "test-only"},
            "outputs": {"buildings_sha256": hashlib.sha256(encoded).hexdigest()},
            "preview_bounds_wgs84": [[south, west], [north, east]],
            **manifest_changes,
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return tmp_path, manifest

    return write_run


def test_completed_run_preserves_confidence_and_inference_provenance(completed_run):
    directory, manifest = completed_run()
    loaded = load_detection(directory)
    row = loaded["buildings"].iloc[0]
    assert row.building_id == "test_inside"
    assert row.confidence == pytest.approx(0.91)
    assert loaded["buildings"].crs.to_epsg() == 4326
    assert loaded["raw_count"] == loaded["selected_count"] == 1
    assert loaded["metadata"]["analysis_mode"] == "image_inference"
    assert loaded["metadata"]["inference"] == manifest
    assert loaded["metadata"]["inference"]["image"]["observed_at"] is None
    assert all(source["source_id"] != "microsoft" for source in loaded["metadata"]["sources"])


@pytest.mark.parametrize("status", ["running", "failed", None])
def test_incomplete_results_never_look_like_completed_detections(completed_run, status):
    directory, _ = completed_run(status=status)
    with pytest.raises(ValueError, match="completa"):
        load_detection(directory)


def test_mismatched_geojson_is_rejected_even_if_feature_count_did_not_change(completed_run):
    directory, _ = completed_run()
    path = directory / "buildings.geojson"
    modified = json.loads(path.read_text(encoding="utf-8"))
    modified["features"][0]["properties"]["confidence"] = 0.22
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_detection(directory)


def test_matching_hash_does_not_hide_a_wrong_manifest_count(completed_run):
    directory, _ = completed_run(count=2)
    with pytest.raises(ValueError, match="conteo"):
        load_detection(directory)


def test_aoi_selection_retains_entire_crossing_footprint(completed_run, pilot_bounds):
    west, south, east, north = pilot_bounds
    center_y = (south + north) / 2
    crossing = box(west - 0.0002, center_y, west + 0.0002, center_y + 0.0001)
    outside = box(east + 0.002, center_y, east + 0.0021, center_y + 0.0001)
    directory, _ = completed_run(
        [
            synthetic_feature("test_crosses_aoi", crossing),
            synthetic_feature("test_outside_aoi", outside),
        ]
    )
    loaded = load_detection(directory)
    assert loaded["raw_count"] == 2
    assert loaded["selected_count"] == 1
    assert loaded["buildings"].building_id.tolist() == ["test_crosses_aoi"]
    selected = loaded["buildings"].geometry.iloc[0]
    assert selected.equals(crossing)
    assert selected.bounds[0] < west
    assert selected.area > selected.intersection(box(*pilot_bounds)).area
    assert loaded["metadata"]["selection"]["raw_count"] == 2


def test_empty_completed_run_is_a_valid_zero_result_not_an_exception(completed_run):
    directory, _ = completed_run([])
    loaded = load_detection(directory)
    assert loaded["buildings"].empty
    assert loaded["buildings"].crs.to_epsg() == 4326
    assert {"building_id", "source_id", "confidence", "geometry"}.issubset(loaded["buildings"].columns)
    assert loaded["raw_count"] == loaded["selected_count"] == 0
    assert loaded["preview_path"] is None
    assert loaded["metadata"]["inference"]["count"] == 0


def test_outside_only_predictions_keep_raw_count_and_zero_selected(completed_run, pilot_bounds):
    _, south, east, _ = pilot_bounds
    directory, _ = completed_run(
        [synthetic_feature("test_outside", box(east + 0.002, south, east + 0.003, south + 0.001))]
    )
    loaded = load_detection(directory)
    assert loaded["buildings"].empty
    assert loaded["raw_count"] == 1
    assert loaded["selected_count"] == 0


def test_running_lock_blocks_reading_previous_completed_results(completed_run):
    directory, _ = completed_run()
    (directory / ".running.lock").write_text("synthetic-unit-test-lock", encoding="utf-8")
    with pytest.raises(ValueError, match="ejecutándose"):
        load_detection(directory)


def test_model_identity_must_agree_inside_manifest(completed_run):
    directory, _ = completed_run(model_id="another_model")
    with pytest.raises(ValueError, match="modelo"):
        load_detection(directory)


def test_valid_geojson_hash_cannot_hide_detections_from_a_different_model(completed_run, pilot_bounds):
    west, south, east, north = pilot_bounds
    feature = synthetic_feature("wrong_model", box(west, south, (west + east) / 2, (south + north) / 2))
    feature["properties"]["model_id"] = "another_model"
    directory, _ = completed_run([feature])
    with pytest.raises(ValueError, match="modelo"):
        load_detection(directory)


def test_adjusted_model_keeps_actual_hash_and_training_provenance_in_export_metadata(completed_run):
    from src.application.provenance import detection_metadata

    model = {
        "id": "synthetic_adjusted_model",
        "trained_locally": True,
        "sha256": "1" * 64,
        "variant": "spacenet2_khartoum_v1",
        "license": "test-only",
    }
    directory, manifest = completed_run(model=model, model_id=model["id"])
    loaded = load_detection(directory)
    metadata = detection_metadata({}, loaded)
    assert metadata["building_mode"] == "finetuned_model_inference"
    assert metadata["detection"] == manifest
    source = next(source for source in metadata["sources"] if source.get("model_id") == model["id"])
    assert source["model_sha256"] == model["sha256"]
    assert "ajuste SpaceNet" in source["organization"]
