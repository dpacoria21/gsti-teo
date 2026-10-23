"""Reproducible local references: no detector, GPU or production data writes."""

# ruff: noqa: E402

import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.ops import unary_union

from scripts.prepare_chullo_labels import (
    METHOD,
    clipped_references,
    feature_collection,
    prepare,
    projected_geometry,
    safe_training_chips,
    sha256,
    validate_annotations,
    validate_split_regions,
)


def references():
    return [
        (box(x + 2, y + 2, x + 12, y + 12), {"id": f"r{x}_{y}"}) for y in range(0, 192, 24) for x in range(0, 192, 24)
    ]


def test_train_selection_excludes_ignore_with_one_pixel_margin_and_adds_new_area():
    ignored = [(box(0, 0, 70, 192), {"reason": "ambiguous"})]
    result = safe_training_chips(references(), ignored, np.ones((192, 192), dtype=bool))
    covered = box(0, 0, 0, 0)
    assert 1 <= len(result) <= 4
    for chip in result:
        assert not chip["region"].intersects(ignored[0][0].buffer(1))
        assert chip["reference_count"] >= 8
        assert chip["region"].difference(covered).area >= chip["size"] ** 2 * 0.25
        covered = unary_union((covered, chip["region"]))
    assert result == safe_training_chips(references(), ignored, np.ones((192, 192), dtype=bool))


def test_complete_parent_stays_one_full_chip_and_partial_nodata_is_excluded():
    valid = np.ones((192, 192), dtype=bool)
    result = safe_training_chips(references(), [], valid)
    assert [(chip["x"], chip["y"], chip["size"]) for chip in result] == [(0, 0, 192)]
    valid[:, :70] = False
    result = safe_training_chips(references(), [], valid)
    assert all(valid[c["y"] : c["y"] + c["size"], c["x"] : c["x"] + c["size"]].all() for c in result)


def test_no_safe_training_chip_fails_instead_of_using_partial_annotations():
    with pytest.raises(ValueError, match="No safe train chip"):
        safe_training_chips(references(), [(box(0, 0, 192, 192), {})], np.ones((192, 192), bool))


def test_cross_split_overlap_rejected_before_derived_chips_can_hide_leakage():
    rows = [
        {"scene_id": "a", "split": "train", "window": [0, 0, 192, 192]},
        {"scene_id": "b", "split": "val", "window": [100, 0, 192, 192]},
        {"scene_id": "c", "split": "test", "window": [600, 0, 192, 192]},
    ]
    with pytest.raises(ValueError, match="Cross-split"):
        validate_split_regions(rows, from_origin(200_000, 8_000_000, 0.5, 0.5))
    rows[1]["window"][0] = 300
    gaps = validate_split_regions(rows, from_origin(200_000, 8_000_000, 0.5, 0.5))
    assert gaps["minimum_parent_gap_m"]["train_val"] == pytest.approx(54)


def test_holes_and_metadata_survive_affine_projection_and_clipping():
    record = {
        "id": "courtyard",
        "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "holes": [[[20, 20], [40, 20], [40, 40], [20, 40]]],
        "notes": "Open courtyard is background",
        "confidence": "approximate",
    }
    document = {"scene_id": "sample", "annotation_method": METHOD, "human_reviewed": False, "polygons": [record]}
    refs, _ = validate_annotations(document, "sample")
    clipped = clipped_references(refs, box(10, 10, 90, 90))
    transform = Affine(0.5, 0.1, 200_000, 0.05, -0.5, 8_000_000)
    collection = feature_collection(
        clipped, transform, rasterio.crs.CRS.from_epsg(32719), "sample", region=box(10, 10, 90, 90)
    )
    geometry = shape(collection["features"][0]["geometry"])
    assert len(geometry.interiors) == 1
    assert geometry.area == pytest.approx((80 * 80 - 20 * 20) * abs(transform.determinant))
    properties = collection["features"][0]["properties"]
    assert properties["notes"] == record["notes"] and properties["confidence"] == "approximate"
    assert properties["human_reviewed"] is False and properties["clipped_to_chip"] is True
    assert collection["crs"]["properties"]["name"] == "EPSG:32719"


def test_line_only_intersections_do_not_create_spurious_references():
    assert clipped_references([(box(0, 0, 10, 10), {"id": "edge"})], box(10, 0, 20, 10)) == []


def test_invalid_geometry_overlap_and_prediction_sources_fail_without_repair():
    document = {
        "scene_id": "s",
        "annotation_method": METHOD,
        "human_reviewed": False,
        "polygons": [{"id": "a", "points": [[0, 0], [20, 0], [20, 20], [0, 20]]}],
    }
    document["polygons"].append({"id": "b", "points": [[10, 0], [30, 0], [30, 20], [10, 20]]})
    with pytest.raises(ValueError, match="overlaps >1px2"):
        validate_annotations(document, "s")
    document["polygons"] = [{"id": "a", "points": [[0, 0], [20, 20], [0, 20], [20, 0]]}]
    with pytest.raises(ValueError, match="Invalid polygon"):
        validate_annotations(document, "s")
    document["annotation_method"] = "model_predictions"
    with pytest.raises(ValueError, match="prediction labels are forbidden"):
        validate_annotations(document, "s")


def synthetic_source(tmp_path: Path):
    image = tmp_path / "original.tif"
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=600,
        height=1200,
        count=4,
        dtype="uint8",
        crs="EPSG:32719",
        transform=from_origin(200_000, 8_000_000, 0.5, 0.5),
    ) as output:
        output.write(np.stack([np.full((1200, 600), value, dtype="uint8") for value in (40, 90, 150, 200)]))
    directory = tmp_path / "annotations"
    directory.mkdir()
    scenes = []
    for index, split in enumerate(("train", "train", "train", "train", "val", "val", "test", "test")):
        scene_id = f"{split}_{index}"
        x, y = (index % 2) * 300, (index // 2) * 300
        scene = {
            "scene_id": scene_id,
            "split": split,
            "window": [x, y, 192, 192],
            "image": scene_id + ".tif",
            "labels": scene_id + ".geojson",
        }
        scenes.append(scene)
        roofs = [{**record, "points": list(geometry.exterior.coords)} for geometry, record in references()]
        annotation = {
            "scene_id": scene_id,
            "annotation_method": METHOD,
            "human_reviewed": False,
            "polygons": roofs,
            "ignore_polygons": [],
        }
        if index in (0, 4):
            annotation["ignore_polygons"] = [
                {"points": [[0, 0], [70, 0], [70, 192], [0, 192]], "reason": "synthetic ambiguous strip"}
            ]
        (directory / (scene_id + ".pixels.json")).write_text(json.dumps(annotation), encoding="utf-8")
    selection = directory / "selection.json"
    selection.write_text(
        json.dumps({"parent_image": image.name, "source_sha256": sha256(image), "scenes": scenes}), encoding="utf-8"
    )
    return selection


def test_end_to_end_rebuilds_parents_preserves_splits_and_writes_rgb_only(tmp_path):
    selection = synthetic_source(tmp_path)
    manifest_path = prepare(selection, tmp_path, render_overlays=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["parent_split_counts"] == {"train": 4, "val": 2, "test": 2}
    assert manifest["human_reviewed"] is False
    for scene in manifest["scenes"]:
        assert scene["parent_scene_id"].split("_")[0] == scene["split"]
        assert sha256(manifest_path.parent / scene["image"]) == scene["image_sha256"]
        with rasterio.open(manifest_path.parent / scene["image"]) as image:
            assert image.count == 3 and image.read()[:, 0, 0].tolist() == [40, 90, 150]
            collection = json.loads((manifest_path.parent / scene["labels"]).read_text(encoding="utf-8"))
            for feature in collection["features"]:
                local = projected_geometry(shape(feature["geometry"]), ~image.transform)
                assert box(0, 0, image.width, image.height).covers(local)
        if scene["split"] == "train":
            assert "ignore_regions" not in scene and scene["usable_reference_count_before_ignore"] >= 8
    assert any("ignore_regions" in scene for scene in manifest["scenes"] if scene["split"] == "val")
    val = next(scene for scene in manifest["scenes"] if scene["scene_id"] == "val_4")
    assert val["usable_reference_count"] == 40
    assert val["evaluable_pixel_fraction"] == pytest.approx(122 / 192)
    assert manifest["unique_train_reference_count"] == len(manifest["unique_train_reference_ids"])
    assert manifest["unique_train_reference_count"] <= manifest["split_reference_counts_with_training_overlap"]["train"]
    assert manifest["annotation_provenance"]["method"] == METHOD
    assert "sin revisión humana" in manifest["assessment_caveat"]
    # Missing parent rasters are recreated from the hash-pinned original.
    parent = selection.parent / "train_0.tif"
    expected_hash = sha256(parent)
    parent.unlink()
    prepare(selection, tmp_path, render_overlays=False)
    assert sha256(parent) == expected_hash


def test_source_hash_mismatch_stops_before_writing_manifest(tmp_path):
    selection = synthetic_source(tmp_path)
    content = json.loads(selection.read_text(encoding="utf-8"))
    content["source_sha256"] = "0" * 64
    selection.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="SHA256 differs"):
        prepare(selection, tmp_path, render_overlays=False)
    assert not selection.with_name("manifest.json").exists()


def test_later_unusable_parent_is_rejected_before_earlier_parent_outputs(tmp_path):
    selection = synthetic_source(tmp_path)
    annotation_path = selection.parent / "train_3.pixels.json"
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    document["ignore_polygons"] = [{"points": [[0, 0], [192, 0], [192, 192], [0, 192]], "reason": "No complete region"}]
    annotation_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="No safe train chip"):
        prepare(selection, tmp_path, render_overlays=False)
    assert not selection.with_name("train_0.tif").exists()
    assert not selection.with_name("manifest.json").exists()


def test_explicit_train_exclusion_keeps_annotations_and_inventory_without_silent_fallback(tmp_path):
    selection = synthetic_source(tmp_path)
    document = json.loads(selection.read_text(encoding="utf-8"))
    document["scenes"][3].update(
        {"excluded_from_training": True, "exclusion_reason": "No safe complete chip after annotation QA"}
    )
    selection.write_text(json.dumps(document), encoding="utf-8")
    path = prepare(selection, tmp_path, render_overlays=False)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["parent_split_counts"]["train"] == 4
    assert manifest["used_parent_split_counts"] == {"train": 3, "val": 2, "test": 2}
    assert manifest["excluded_parents"][0]["scene_id"] == "train_3"
    assert not any(row["parent_scene_id"] == "train_3" for row in manifest["scenes"])
    assert next(row for row in manifest["parents"] if row["scene_id"] == "train_3")["derived_scene_count"] == 0
    assert path.with_name("train_3.geojson").is_file() and path.with_name("train_3.tif").is_file()
    document["scenes"][4].update({"excluded_from_training": True, "exclusion_reason": "Forbidden validation exclusion"})
    selection.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="only allowed for train"):
        prepare(selection, tmp_path, render_overlays=False)
