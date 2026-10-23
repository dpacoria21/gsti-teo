"""Final protocol tests use synthetic bytes/results, never the real holdout."""

# ruff: noqa: E402
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")

from src.inference.model import sha256_file
from src.training import final_evaluation as evaluation


def report(tp=10, fp=10, fn=5, merges=1, affected=2, score=0.5, mask=0.5):
    return {
        "score_threshold": score,
        "mask_threshold": mask,
        "summary": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn),
            "f0_5": 1.25 * tp / (1.25 * tp + 0.25 * fn + fp),
            "merged_predictions": merges,
            "references_affected_by_merges": affected,
        },
    }


@pytest.fixture
def completed_development(tmp_path, monkeypatch):
    run_dir, data_dir, output = tmp_path / "run", tmp_path / "data", tmp_path / "frozen_evaluation"
    run_dir.mkdir()
    data_dir.mkdir()
    checkpoints = {name: tmp_path / f"{name}.pth" for name in ("usa", "chullo_v1")}
    for name, path in checkpoints.items():
        path.write_bytes(name.encode())
    hashes = {name: sha256_file(path) for name, path in checkpoints.items()}
    monkeypatch.setattr(evaluation, "verified_checkpoints", lambda: (checkpoints, hashes))
    candidate = run_dir / "best_model.pth"
    candidate.write_bytes(b"synthetic candidate")
    # Deliberately absent image/label files prove the freeze phase never reads them.
    rows = [
        {"scene_id": split, "split": split, "image": f"{split}.tif", "labels": f"{split}.geojson"}
        for split in ("train", "val", "test")
    ]
    development = {
        "dataset_id": "synthetic_v2",
        "source_sha256": "source-hash",
        "phase": "development_train_val_only",
        "scenes": rows[:2],
    }
    final = {
        **development,
        "phase": "frozen_final_evaluation",
        "scenes": rows,
        "geographic_validation": {
            "cross_split_overlap_count": 0,
            "minimum_parent_gap_m": {"train_val": 100.0, "train_test": 200.0, "val_test": 100.0},
        },
        "assessment_caveat": "Synthetic unit fixture; not an actual test evaluation.",
    }
    dev_path, final_path = data_dir / "development_manifest.json", data_dir / "manifest.json"
    evaluation.write_json(dev_path, development)
    evaluation.write_json(final_path, final)
    experiment = {"name": "synthetic", "anchors": [16, 32, 64, 128, 256], "unfreeze_layer4": False}
    selected = {
        "experiment": experiment,
        "epoch": 20,
        "report": report(tp=12, fp=3, fn=3, merges=0, affected=0, score=0.65, mask=0.6),
        "checkpoint_sha256": sha256_file(candidate),
    }
    run = {
        "status": "completed",
        "validation_gate_passed": True,
        "test_opened": False,
        "selected": selected,
        "experiments": [experiment],
        "manifest_sha256": sha256_file(dev_path),
        "image_scale": 3.125,
        "baseline_image_scale": 1.5625,
        "base_checkpoint_sha256": hashes["chullo_v1"],
        "baseline_checkpoints_sha256": hashes,
        "training_config": {"epochs_per_experiment": 60, "initialization": "v1"},
    }
    evaluation.write_json(run_dir / "protocol.json", run)
    evaluation.write_json(run_dir / "selected_validation.json", selected)
    evaluation.write_json(
        run_dir / "baselines_validation.json",
        {
            "usa": [report()],
            "chullo_v1": [report(), report(fp=5, score=0.8), report(tp=9, fp=0, score=0.9)],
        },
    )
    return SimpleNamespace(
        run_dir=run_dir,
        development=dev_path,
        final=final_path,
        output=output,
        run=run,
        checkpoints=checkpoints,
        hashes=hashes,
    )


def freeze(fixture):
    return evaluation.freeze_protocol(fixture.run_dir, fixture.development, fixture.final, fixture.output)


def test_calibration_retains_tp_and_fusion_constraints_instead_of_optimizing_precision_alone():
    default = report()
    accepted = report(fp=5, score=0.8)
    lower_recall = report(tp=9, fp=0, score=0.9)
    more_merges = report(fp=0, merges=2, score=0.65)
    more_affected = report(fp=0, affected=3, mask=0.6)
    actual_default, selected = evaluation.choose_calibrated_v1(
        [default, accepted, lower_recall, more_merges, more_affected]
    )
    assert actual_default == default and selected == accepted


def test_freeze_reads_no_test_images_or_labels_and_preserves_actual_initialization(completed_development, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Freeze must not access test raster/labels or run any model.")

    monkeypatch.setattr(evaluation, "BuildingDataset", forbidden)
    monkeypatch.setattr(evaluation, "predict_scenes", forbidden)
    monkeypatch.setattr(evaluation, "read_manifest", forbidden)
    monkeypatch.setattr(evaluation, "verify_development_artifacts", forbidden)
    path = freeze(completed_development)
    protocol, inputs = evaluation.verify_frozen_protocol(path)
    assert protocol["initial_checkpoint_sha256"] == completed_development.hashes["chullo_v1"]
    assert protocol["image_scale"] == 3.125 and protocol["baseline_image_scale"] == 1.5625
    assert protocol["operating_points"]["v1_calibrated"]["score_threshold"] == 0.8
    assert (
        protocol["operating_points"]["v1_candidate_scale_at_v2_operating_point"]
        == protocol["operating_points"]["v2_selected"]
    )
    assert inputs["v2_checkpoint"] == completed_development.run_dir / "best_model.pth"
    assert not (completed_development.final.parent / "test.tif").exists()
    assert all(not Path(item["path"]).is_absolute() for item in protocol["inputs"].values())
    with pytest.raises(FileExistsError):
        freeze(completed_development)


def test_freeze_refuses_failed_gate_before_any_test_evaluation(completed_development):
    run = completed_development.run
    run["validation_gate_passed"] = False
    run["selected"] = None
    evaluation.write_json(completed_development.run_dir / "protocol.json", run)
    (completed_development.run_dir / "selected_validation.json").unlink()
    completed_development.final.unlink()
    with pytest.raises(ValueError, match="passing the validation gate"):
        freeze(completed_development)
    assert not completed_development.output.exists()


def test_freeze_refuses_changed_development_scenes(completed_development):
    final = evaluation.read_json(completed_development.final)
    final["scenes"][0]["labels"] = "changed-after-training.geojson"
    evaluation.write_json(completed_development.final, final)
    with pytest.raises(ValueError, match="changed the frozen development"):
        freeze(completed_development)


@pytest.mark.parametrize("changed", ["protocol", "checkpoint", "manifest"])
def test_evaluation_fails_closed_on_changed_frozen_inputs(completed_development, monkeypatch, changed):
    path = freeze(completed_development)
    targets = {
        "protocol": path,
        "checkpoint": completed_development.run_dir / "best_model.pth",
        "manifest": completed_development.final,
    }
    target = targets[changed]
    target.write_bytes(target.read_bytes() + b" ")
    monkeypatch.setattr(
        evaluation, "BuildingDataset", lambda *args, **kwargs: pytest.fail("No test access after mutation")
    )
    with pytest.raises(ValueError, match="changed"):
        evaluation.evaluate_frozen(path, device="cpu")
    assert not (completed_development.output / "evaluation_started.json").exists()


def test_final_evaluation_uses_fixed_scales_points_and_only_three_model_loads(completed_development, monkeypatch):
    path = freeze(completed_development)
    calls, checks = [], []
    fake_manifest = evaluation.read_json(completed_development.final)
    monkeypatch.setattr(evaluation, "read_manifest", lambda path: fake_manifest)
    monkeypatch.setattr(evaluation, "BuildingDataset", lambda manifest, split: {"split": split})

    def verify_rows(manifest):
        assert (completed_development.output / "evaluation_started.json").is_file()
        assert [row["split"] for row in manifest["scenes"]] == ["test"]
        checks.append(True)

    monkeypatch.setattr(evaluation, "verify_development_artifacts", verify_rows)
    loads, anchors = [], {}

    def load_model(weight_path, device):
        name = weight_path.stem
        loads.append(name)
        return name, "cpu"

    monkeypatch.setattr(evaluation, "load_model", load_model)
    monkeypatch.setattr(evaluation, "configure_anchors", lambda model, sizes: anchors.update({model: sizes}))
    reference = np.zeros((1, 8, 8), dtype=bool)
    reference[0, 1:4, 1:4] = True
    false = np.zeros_like(reference)
    false[0, 5:8, 5:8] = True

    def predict(model, dataset, device, scale):
        assert dataset["split"] == "test"
        calls.append((model, scale))
        if model == "usa":
            probabilities, scores = np.zeros((0, 8, 8)), np.array([])
        elif model == "chullo_v1":
            probabilities, scores = np.concatenate([reference, false]).astype(float), np.array([0.95, 0.7])
        else:
            probabilities, scores = reference.astype(float), np.array([0.95])
        return [
            {
                "scene_id": "test",
                "probabilities": probabilities,
                "scores": scores,
                "reference": reference,
                "valid": np.ones((8, 8), dtype=bool),
            }
        ]

    monkeypatch.setattr(evaluation, "predict_scenes", predict)
    result = evaluation.evaluate_frozen(path, device="cpu")
    assert checks == [True] and loads == ["usa", "chullo_v1", "best_model"]
    assert calls == [("usa", 1.5625), ("chullo_v1", 1.5625), ("chullo_v1", 3.125), ("best_model", 3.125)]
    assert anchors["best_model"] == [16, 32, 64, 128, 256]
    assert tuple(result["reports"]) == evaluation.REPORT_NAMES
    assert result["comparison"]["promotion_gate_passed"] is True
    assert result["comparison"]["calibrated_v1_passes_default_gate"] is True
    assert result["comparison"]["v2_adds_conservative_advantage_over_calibrated_v1"] is False
    assert "No atribuir toda la ventaja" in result["comparison"]["interpretation"]
    assert result["historical_test_used"] is False
    with pytest.raises(FileExistsError):
        evaluation.evaluate_frozen(path, device="cpu")
    assert len(loads) == 3


def test_more_fusions_block_promotion_even_when_f0_5_increases():
    reports = {name: report() for name in evaluation.REPORT_NAMES}
    reports["v2_selected"] = report(tp=12, fp=1, fn=3, merges=2)
    result = evaluation.interpret_reports(reports)
    assert result["promotion_gate_passed"] is False
    assert result["differences_v2_minus"]["v1_default"]["f0_5"] > 0


def test_scale_control_prevents_attributing_scale_only_advantage_to_weights():
    reports = {name: report() for name in evaluation.REPORT_NAMES}
    reports["v2_selected"] = report(tp=11, fp=3, fn=4, merges=0, affected=0)
    reports["v1_candidate_scale_at_v2_operating_point"] = copy.deepcopy(reports["v2_selected"])
    result = evaluation.interpret_reports(reports)
    assert result["promotion_gate_passed"] is True
    assert result["v1_candidate_scale_passes_default_gate"] is True
    assert result["v2_adds_conservative_advantage_over_v1_at_same_scale_and_thresholds"] is False
    assert "atribuible a los nuevos pesos" in result["interpretation"]
