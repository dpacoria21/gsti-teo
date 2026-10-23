"""Guard against improving precision by hiding false negatives or ignoring negatives."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")

from src.inference.model import sha256_file  # noqa: E402
from src.training.refinement import eligible, score_predictions, selection_key, verify_development_artifacts  # noqa: E402


def test_selection_rejects_false_positive_reduction_that_loses_true_positives():
    baseline = dict(tp=37, fp=65, merged_predictions=4, references_affected_by_merges=5)
    assert not eligible(dict(tp=36, fp=1, merged_predictions=0, references_affected_by_merges=0), baseline)
    assert not eligible(dict(tp=38, fp=20, merged_predictions=5, references_affected_by_merges=5), baseline)
    assert not eligible(dict(tp=38, fp=20, merged_predictions=3, references_affected_by_merges=6), baseline)
    assert eligible(dict(tp=38, fp=20, merged_predictions=3, references_affected_by_merges=4), baseline)


def test_negative_scene_false_detections_count_in_precision():
    roof = np.zeros((1, 8, 8), dtype=bool)
    roof[0, 2:6, 2:6] = True
    scenes = [
        dict(
            scene_id="roof",
            scores=np.array([0.95]),
            probabilities=roof.astype(float),
            valid=np.ones((8, 8), bool),
            reference=roof,
        ),
        dict(
            scene_id="field",
            scores=np.array([0.7]),
            probabilities=roof.astype(float),
            valid=np.ones((8, 8), bool),
            reference=np.zeros((0, 8, 8), bool),
        ),
    ]
    default = score_predictions(scenes, 0.5, 0.5)
    filtered = score_predictions(scenes, 0.8, 0.5)
    assert default["summary"]["tp"] == 1 and default["summary"]["fp"] == 1
    assert filtered["summary"]["tp"] == 1 and filtered["summary"]["fp"] == 0
    assert filtered["summary"]["f0_5"] == 1
    assert selection_key(filtered) > selection_key(default)


def test_ambiguous_predictions_do_not_become_valid_false_positives():
    probabilities = np.ones((1, 8, 8), dtype=float)
    report = score_predictions(
        [
            dict(
                scene_id="ignore",
                scores=np.array([0.9]),
                probabilities=probabilities,
                valid=np.zeros((8, 8), bool),
                reference=np.zeros((0, 8, 8), bool),
            )
        ],
        0.5,
        0.5,
    )
    assert report["summary"]["fp"] == 0
    assert report["summary"]["f0_5"] == 0


def test_preparation_cli_can_start_from_outside_repository(tmp_path):
    from scripts import prepare_chullo_refinement

    result = subprocess.run(
        [sys.executable, str(prepare_chullo_refinement.ROOT / "scripts/prepare_chullo_refinement.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--development-only" in result.stdout


def test_training_rejects_changed_scene_bytes_before_they_can_affect_selection(tmp_path):
    scene = {"scene_id": "fixture"}
    for key, hash_key in (("image", "image_sha256"), ("labels", "labels_sha256"), ("ignore_regions", "ignore_sha256")):
        path = tmp_path / key
        path.write_bytes(key.encode())
        scene[key], scene[hash_key] = str(path), sha256_file(path)
    verify_development_artifacts({"scenes": [scene]})
    (tmp_path / "ignore_regions").write_bytes(b"changed exclusion area")
    with pytest.raises(ValueError, match="fixture ignore_regions"):
        verify_development_artifacts({"scenes": [scene]})


def test_final_preparation_reuses_development_and_refuses_changed_annotations(tmp_path, monkeypatch):
    from scripts import prepare_chullo_refinement as preparation

    monkeypatch.setattr(preparation, "DIRECTORY", tmp_path)
    annotation = tmp_path / "train.pixels.json"
    annotation.write_text("{}", encoding="utf-8")
    scene = {"scene_id": "train", "split": "train", "window_in_source": [0, 0, 192, 192]}
    for key, hash_key in (("image", "image_sha256"), ("labels", "labels_sha256")):
        path = tmp_path / key
        path.write_bytes(key.encode())
        scene[key], scene[hash_key] = path.name, preparation.sha256(path)
    record = {"scene_id": "train", "annotation_sha256": preparation.sha256(annotation)}
    frozen = {
        "phase": "development_train_val_only",
        "source_sha256": "source-hash",
        "scenes": [scene],
        "parents": [record],
    }
    (tmp_path / "development_manifest.json").write_text(json.dumps(frozen), encoding="utf-8")
    parents = [
        {"scene_id": "train", "split": "train", "window": [0, 0, 192, 192], "annotation_path": annotation},
        {
            "scene_id": "test",
            "split": "test",
            "window": [600, 600, 192, 192],
            "annotation_path": tmp_path / "never-opened.json",
        },
    ]
    scenes, records = preparation.frozen_development(parents, "source-hash")
    assert scenes == {"train": scene} and records == {"train": record}
    before = {name: (tmp_path / name).read_bytes() for name in ("image", "labels", "development_manifest.json")}
    annotation.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="annotations changed"):
        preparation.frozen_development(parents, "source-hash")
    assert before == {name: (tmp_path / name).read_bytes() for name in before}
