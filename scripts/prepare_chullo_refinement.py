"""Build a new full-context dataset; never overwrite the frozen first pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_chullo_labels import (  # noqa: E402
    ATTRIBUTION,
    LABEL_SOURCE,
    feature_collection,
    sha256,
    validate_annotations,
    validate_split_regions,
    write_json,
    write_rgb,
)

DIRECTORY = ROOT / "data/training/chullo_local_v2"
ORIGINAL = ROOT / "data/training/chullo_local_v1"
HOLDOUT = ROOT / "data/training/chullo_local_v2_holdout"
NEGATIVES = [
    ("negative_field_west", "train", [20, 700, 192, 192]),
    ("negative_river_north", "train", [1390, 60, 96, 96]),
    ("negative_park", "train", [1950, 1720, 160, 160]),
    ("negative_field_centre", "val", [360, 720, 192, 192]),
    ("negative_field_south", "test", [720, 1610, 192, 192]),
]


def frozen_development(parents, source_sha256):
    """Reuse immutable development artifacts when adding the final test set."""
    path = DIRECTORY / "development_manifest.json"
    if not path.is_file():
        return {}, {}
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("phase") != "development_train_val_only" or frozen.get("source_sha256") != source_sha256:
        raise ValueError("Development source or phase changed; use a new dataset version.")
    scenes = {scene["scene_id"]: scene for scene in frozen["scenes"]}
    records = {record["scene_id"]: record for record in frozen["parents"]}
    expected = {parent["scene_id"] for parent in parents if parent["split"] != "test"}
    if set(scenes) != expected:
        raise ValueError("Development scene inventory changed; use a new dataset version.")
    for parent in parents:
        if parent["split"] == "test":
            continue
        scene, record = scenes[parent["scene_id"]], records[parent["scene_id"]]
        if scene["split"] != parent["split"] or scene["window_in_source"] != parent["window"]:
            raise ValueError("Development split/window changed; use a new dataset version.")
        if "annotation_path" in parent and sha256(parent["annotation_path"]) != record.get("annotation_sha256"):
            raise ValueError("Development annotations changed after freezing; use a new dataset version.")
        for key, hash_key in (
            ("image", "image_sha256"),
            ("labels", "labels_sha256"),
            ("ignore_regions", "ignore_sha256"),
        ):
            if key in scene and sha256(DIRECTORY / scene[key]) != scene.get(hash_key):
                raise ValueError(f"Frozen development artifact changed: {scene['scene_id']} {key}")
    return scenes, records


def prepare(development_only=False):
    destination = DIRECTORY / ("development_manifest.json" if development_only else "manifest.json")
    if destination.exists():
        raise FileExistsError("Do not overwrite a frozen manifest; use a new dataset version.")
    frozen = json.loads((ORIGINAL / "manifest.json").read_text(encoding="utf-8"))
    source_path = ROOT / frozen["source"]
    if sha256(source_path) != frozen["source_sha256"]:
        raise ValueError("Source image changed.")
    parents = []
    for old in frozen["parents"]:
        if old["split"] == "test":
            continue  # Historical test is never used to select the new candidate.
        label_dir = DIRECTORY if old["split"] == "train" else ORIGINAL
        parents.append(
            {
                "scene_id": old["scene_id"],
                "split": old["split"],
                "window": old["window"],
                "annotation_path": label_dir / (old["scene_id"] + ".pixels.json"),
            }
        )
    for scene_id, window in [
        ("holdout_300_1600", [300, 1600, 192, 192]),
        ("holdout_2100_1100", [2100, 1100, 192, 192]),
    ]:
        parents.append(
            {
                "scene_id": scene_id,
                "split": "test",
                "window": window,
                "annotation_path": HOLDOUT / (scene_id + ".pixels.json"),
            }
        )
    for scene_id, split, window in NEGATIVES:
        parents.append({"scene_id": scene_id, "split": split, "window": window, "verified_negative": True})
    frozen_scenes, frozen_records = frozen_development(parents, frozen["source_sha256"])
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    scenes, inventory = [], []
    with rasterio.open(source_path) as source:
        geography = validate_split_regions(parents, source.transform)
        for parent in parents:
            if parent["scene_id"] in frozen_scenes:
                scenes.append(frozen_scenes[parent["scene_id"]])
                inventory.append(frozen_records[parent["scene_id"]])
                continue
            if development_only and parent["split"] == "test":
                inventory.append({k: v for k, v in parent.items() if k != "annotation_path"})
                continue
            scene_id = parent["scene_id"]
            x, y, width, height = parent["window"]
            window = Window(x, y, width, height)
            if not source.dataset_mask(window=window).all():
                raise ValueError(f"Invalid source pixels in {scene_id}")
            transform = source.window_transform(window)
            refs, ignores = [], []
            if not parent.get("verified_negative"):
                annotation = json.loads(parent["annotation_path"].read_text(encoding="utf-8"))
                refs, ignores = validate_annotations(annotation, scene_id)
            image_path = DIRECTORY / (scene_id + ".tif")
            labels_path = DIRECTORY / (scene_id + ".geojson")
            write_rgb(image_path, source, window)
            write_json(labels_path, feature_collection(refs, transform, source.crs, scene_id))
            row = {
                "scene_id": scene_id,
                "split": parent["split"],
                "parent_scene_id": scene_id,
                "image": image_path.name,
                "labels": labels_path.name,
                "labels_source": LABEL_SOURCE,
                "annotation_type": "reference",
                "window_in_source": parent["window"],
                "crs": source.crs.to_string(),
                "image_sha256": sha256(image_path),
                "labels_sha256": sha256(labels_path),
                "reference_count": len(refs),
                "verified_negative": bool(parent.get("verified_negative")),
            }
            if ignores:
                ignore_path = DIRECTORY / (scene_id + ".ignore.geojson")
                write_json(ignore_path, feature_collection(ignores, transform, source.crs, scene_id, ignored=True))
                row.update(ignore_regions=ignore_path.name, ignore_sha256=sha256(ignore_path))
            scenes.append(row)
            record = {k: v for k, v in parent.items() if k != "annotation_path"}
            if "annotation_path" in parent:
                record.update(
                    annotation_file=str(parent["annotation_path"].relative_to(ROOT)),
                    annotation_sha256=sha256(parent["annotation_path"]),
                )
            else:
                record["review"] = (
                    "Assistant inspected original RGB before any v2 model inference: visible field, riverbed or plaza without visible roofs. Not human reviewed."
                )
            inventory.append(record)
    manifest = {
        "schema_version": 1,
        "dataset_id": "chullo_local_assistant_provisional_v2",
        "annotation_type": "reference",
        "annotation_method": "assistant_visual_manual",
        "human_reviewed": False,
        "source": frozen["source"],
        "source_sha256": frozen["source_sha256"],
        "annotation_provenance": {
            "method": "assistant_visual_manual",
            "human_reviewed": False,
            "source_capture_date": "2026-03-04",
            "review_round": 2,
        },
        "attribution": ATTRIBUTION,
        "geographic_validation": geography,
        "training_policy": "Full annotated parent windows; ignore-aware losses; any roof intersecting ignore excluded whole; verified negative scenes included.",
        "test_policy": "Two new fixed spatial windows and one negative window annotated without predictions; one capture, no independent acquisition; historical v1 test is not used for selection.",
        "assessment_caveat": "Provisional assistant labels of visible roofs, not human truth or verified houses. One capture; new geographic test is not independent acquisition. Old validation is reused for model selection.",
        "parents": inventory,
        "scenes": scenes,
    }
    manifest["phase"] = "development_train_val_only" if development_only else "frozen_final_evaluation"
    write_json(destination, manifest)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true")
    prepare(parser.parse_args().development_only)
