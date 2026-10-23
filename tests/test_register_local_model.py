"""Registration tests use synthetic bytes only; never train or touch real models."""

import pytest

from scripts import register_local_model as registrar
from src import model_catalog
from src.inference.model import MODEL_SHA256, resolve_variant, sha256_file


@pytest.fixture
def completed_pilot(tmp_path):
    run_dir, data_dir = tmp_path / "run", tmp_path / "dataset"
    run_dir.mkdir()
    data_dir.mkdir()
    dataset = {
        "dataset_id": registrar.DATASET_ID,
        "annotation_method": "assistant_visual_manual",
        "human_reviewed": False,
        "parent_split_counts": {"train": 4, "val": 2, "test": 2},
        "used_parent_split_counts": {"train": 3, "val": 2, "test": 2},
        "excluded_parents": [{"scene_id": "train_excluded", "split": "train", "reason": "No safe chips"}],
        "geographic_validation": {
            "cross_split_overlap_count": 0,
            "minimum_parent_gap_m": {"train_val": 104, "train_test": 354, "val_test": 154},
        },
        "scenes": [
            {"scene_id": f"train_chip_{i}", "parent_scene_id": f"train_{min(i, 2)}", "split": "train"} for i in range(4)
        ]
        + [
            {"scene_id": f"{split}_{i}", "parent_scene_id": f"{split}_{i}", "split": split, "image": f"{split}_{i}.tif"}
            for split in ("val", "test")
            for i in range(2)
        ],
        "source": "data/imagery/chullo/chullo_rgb.tif",
        "assessment_caveat": "Synthetic fixture; no actual accuracy claim.",
    }
    data_manifest = data_dir / "manifest.json"
    registrar.write_json(data_manifest, dataset)
    checkpoint = run_dir / "best_model.pth"
    checkpoint.write_bytes(b"synthetic registration test; not a usable model")
    baseline = {
        "images": 2,
        "f1": 0.2,
        "tp": 1,
        "fp": 4,
        "fn": 4,
        "precision": 0.2,
        "recall": 0.2,
        "merged_predictions": 1,
        "references_affected_by_merges": 2,
    }
    candidate = {
        "images": 2,
        "f1": 0.4,
        "tp": 2,
        "fp": 3,
        "fn": 3,
        "precision": 0.4,
        "recall": 0.4,
        "merged_predictions": 2,
        "references_affected_by_merges": 4,
    }
    run = {
        "status": "completed",
        "dataset_id": dataset["dataset_id"],
        "config": {
            "epochs": 50,
            "batch_size": 1,
            "image_scale": 1.5625,
            "score_threshold": 0.5,
            "mask_threshold": 0.5,
            "match_iou": 0.5,
        },
        "dataset_manifest": str(data_manifest),
        "dataset_manifest_sha256": sha256_file(data_manifest),
        "initial_checkpoint": "C:\\Users\\PrivateName\\weights\\original.pth",
        "initial_checkpoint_sha256": MODEL_SHA256,
        "production_model_modified": False,
        "best_epoch": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "private_note": "Local file C:\\Users\\PrivateName\\secret\\notes.json",
    }
    reports = {
        "run.json": run,
        "epochs.json": [{"epoch": i, "validation": {"f1": 0.4}} for i in range(1, 51)],
        "comparison.json": {"baseline": baseline, "candidate": candidate, "selection_used_test": False},
        "baseline_val.json": {"summary": baseline},
        "best_val.json": {"summary": candidate},
        "best_val_stored_checkpoint.json": {"summary": candidate},
        "baseline_test.json": {
            "summary": baseline,
            "per_scene": [{"scene_id": f"test_{i}", "reference_instances": 3 - i} for i in range(2)],
        },
        "candidate_test.json": {
            "summary": candidate,
            "per_scene": [{"scene_id": f"test_{i}", "reference_instances": 3 - i} for i in range(2)],
        },
    }
    for name, value in reports.items():
        registrar.write_json(run_dir / name, value)
    return run_dir, data_manifest, tmp_path / "registered", checkpoint


def test_registers_matching_checkpoint_and_portable_evidence(completed_pilot, monkeypatch):
    run_dir, data_manifest, output, checkpoint = completed_pilot
    original_checksum = sha256_file(checkpoint)
    manifest_path = registrar.register(run_dir, data_manifest, output)
    metadata = registrar.read_json(manifest_path)
    assert metadata["title"] == "Ajustado · Chullo (piloto)"
    assert metadata["parent_split_counts"] == {"train": 3, "val": 2, "test": 2}
    assert metadata["annotated_parent_split_counts"]["train"] == 4
    assert "3 recortes originales de entrenamiento" in metadata["description"]
    assert metadata["description"].startswith("Modelo experimental.")
    assert any("uniones no están resueltas" in limitation for limitation in metadata["limitations"])
    assert metadata["training_image_scale"] == 1.5625 and metadata["inference_tile_size"] == 512
    assert metadata["human_reviewed"] is False and metadata["initial_model_sha256"] == MODEL_SHA256
    assert metadata["sha256"] == sha256_file(output / "weights.pth") == original_checksum
    assert sha256_file(checkpoint) == original_checksum
    assert metadata["evaluation_summary"]["reference_instances"] == 5
    assert metadata["evaluation_summary"]["selection_used_test"] is False
    for name, checksum in metadata["public_reports_sha256"].items():
        assert sha256_file(output / name) == checksum
        text = (output / name).read_text(encoding="utf-8")
        assert "PrivateName" not in text and str(run_dir) not in text and str(data_manifest) not in text
    public_run = registrar.read_json(output / "run.json")
    assert public_run["checkpoint"] == "training_run/best_model.pth"
    assert public_run["dataset_manifest"] == registrar.DATASET_BASE + "/manifest.json"
    assert "no se afirma una licencia abierta" in (output / "LICENSE_AND_ATTRIBUTION.txt").read_text(encoding="utf-8")
    monkeypatch.setattr(model_catalog, "LOCAL_MANIFEST", manifest_path)
    path, actual = resolve_variant(model_catalog.LOCAL_VARIANT)
    assert path == output / "weights.pth" and actual["sha256"] == original_checksum
    with pytest.raises(FileExistsError):
        registrar.register(run_dir, data_manifest, output)
    assert sha256_file(output / "weights.pth") == original_checksum


@pytest.mark.parametrize(
    "filename,field,value,message",
    [
        ("run.json", "status", "running", "completada"),
        ("run.json", "initial_checkpoint_sha256", "0" * 64, "GeoAI USA"),
        ("run.json", "checkpoint_sha256", "0" * 64, "SHA256"),
        ("run.json", "best_epoch", 2, "primera en empates"),
        ("comparison.json", "selection_used_test", True, "no debe utilizar test"),
    ],
)
def test_rejects_unfinished_or_inconsistent_run_before_creating_destination(
    completed_pilot, filename, field, value, message
):
    run_dir, data_manifest, output, _ = completed_pilot
    report = registrar.read_json(run_dir / filename)
    report[field] = value
    registrar.write_json(run_dir / filename, report)
    with pytest.raises(ValueError, match=message):
        registrar.register(run_dir, data_manifest, output)
    assert not output.exists()


def test_rejects_incomplete_epoch_history_and_changed_dataset(completed_pilot):
    run_dir, data_manifest, output, _ = completed_pilot
    history = registrar.read_json(run_dir / "epochs.json")
    registrar.write_json(run_dir / "epochs.json", history[:-1])
    with pytest.raises(ValueError, match="50 épocas completas"):
        registrar.register(run_dir, data_manifest, output)
    registrar.write_json(run_dir / "epochs.json", history)
    data_manifest.write_text(data_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset cambió"):
        registrar.register(run_dir, data_manifest, output)
    assert not output.exists()


def test_never_overwrites_existing_directory_even_if_empty(completed_pilot):
    run_dir, data_manifest, output, _ = completed_pilot
    output.mkdir()
    with pytest.raises(FileExistsError):
        registrar.register(run_dir, data_manifest, output)
    assert list(output.iterdir()) == []


def test_metric_caveats_expose_recall_false_positive_and_storage_tradeoffs(completed_pilot):
    run_dir, _, _, _ = completed_pilot
    reports = {name: registrar.read_json(run_dir / name) for name in registrar.REPORTS}
    reports["candidate_test.json"]["summary"]["fp"] = 7
    reports["best_val_stored_checkpoint.json"]["summary"]["f1"] = 0.375
    notes = registrar.metric_limitations(reports)
    assert any("más falsos positivos" in note for note in notes)
    assert any("4 → 7" in note for note in notes)
    assert any("1 → 2" in note and "no están resueltas" in note for note in notes)
    assert any("FP16" in note and "0.375" in note and "diferencia" in note for note in notes)
