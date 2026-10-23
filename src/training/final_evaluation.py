"""Freeze a new-holdout protocol, then evaluate it once without model selection.

Freezing reads completed development reports, checkpoint bytes and manifest JSON
only. Test rasters and label files are first accessed by the separate evaluate
phase, after the protocol and all input hashes have been verified.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from src.inference.model import load_model, sha256_file
from src.training.dataset import BuildingDataset, read_manifest
from src.training.refinement import (
    configure_anchors,
    eligible,
    predict_scenes,
    score_predictions,
    selection_key,
    verified_checkpoints,
    verify_development_artifacts,
)

REPORT_NAMES = (
    "usa_default",
    "v1_default",
    "v1_calibrated",
    "v2_selected",
    "v1_at_v2_operating_point",
    "v1_candidate_scale_at_v2_operating_point",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def operating_point(report):
    result = {key: report[key] for key in ("score_threshold", "mask_threshold")}
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1
        for value in result.values()
    ):
        raise ValueError("Invalid frozen operating point.")
    return result


def choose_calibrated_v1(reports):
    """Choose exclusively from validation; retain the first point on exact ties."""
    defaults = [
        report for report in reports if operating_point(report) == {"score_threshold": 0.5, "mask_threshold": 0.5}
    ]
    if len(defaults) != 1:
        raise ValueError("Exactly one v1 default validation point is required.")
    baseline = defaults[0]["summary"]
    allowed = [
        report
        for report in reports
        if report["summary"]["tp"] >= baseline["tp"]
        and report["summary"]["merged_predictions"] <= baseline["merged_predictions"]
        and report["summary"]["references_affected_by_merges"] <= baseline["references_affected_by_merges"]
    ]
    return defaults[0], max(allowed, key=selection_key)


def validate_manifest_pair(development, final):
    """Inspect metadata only, never the held-out imagery or annotations."""
    if development.get("phase") != "development_train_val_only" or final.get("phase") != "frozen_final_evaluation":
        raise ValueError("Expected frozen development and final-evaluation manifests.")
    if development.get("source_sha256") != final.get("source_sha256") or development.get("dataset_id") != final.get(
        "dataset_id"
    ):
        raise ValueError("Development/final dataset identity or source differs.")
    dev_scenes, final_scenes = development["scenes"], final["scenes"]
    if {row["split"] for row in dev_scenes} != {"train", "val"}:
        raise ValueError("Development must contain train/val only.")
    if {row["split"] for row in final_scenes} != {"train", "val", "test"}:
        raise ValueError("Final manifest must contain train, val and the new test split.")
    if len({row["scene_id"] for row in final_scenes}) != len(final_scenes):
        raise ValueError("Duplicate final scene IDs.")
    final_dev = {row["scene_id"]: row for row in final_scenes if row["split"] != "test"}
    if {row["scene_id"]: row for row in dev_scenes} != final_dev:
        raise ValueError("Final preparation changed the frozen development scenes.")
    geography = final.get("geographic_validation", {})
    gaps = geography.get("minimum_parent_gap_m", {})
    if geography.get("cross_split_overlap_count") != 0 or set(gaps) != {"train_val", "train_test", "val_test"}:
        raise ValueError("Missing geographic separation evidence.")
    if any(not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0 for gap in gaps.values()):
        raise ValueError("All geographic split gaps must be positive and finite.")


def freeze_protocol(run_dir, development_manifest, final_manifest, output_dir):
    """Predeclare operating points, scale controls and rules without test IO."""
    run_dir, development_manifest, final_manifest, output_dir = map(
        lambda value: Path(value).resolve(), (run_dir, development_manifest, final_manifest, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError("A frozen protocol/output directory must never be overwritten.")
    run = read_json(run_dir / "protocol.json")
    if run.get("status") != "completed" or run.get("validation_gate_passed") is not True or not run.get("selected"):
        raise ValueError(
            "Only a completed run with a selected candidate passing the validation gate can access final test."
        )
    selected = read_json(run_dir / "selected_validation.json")
    baselines = read_json(run_dir / "baselines_validation.json")
    development, final = read_json(development_manifest), read_json(final_manifest)
    validate_manifest_pair(development, final)
    if run.get("test_opened") is not False or run.get("selected") != selected:
        raise ValueError("The selected development report must match the completed, test-unopened run.")
    if run.get("manifest_sha256") != sha256_file(development_manifest):
        raise ValueError("Development manifest changed after training.")
    if selected.get("experiment") not in run.get("experiments", []):
        raise ValueError("The selected experiment was not predeclared in this run.")
    anchors = selected["experiment"].get("anchors")
    if (
        not isinstance(anchors, list)
        or len(anchors) != 5
        or any(type(size) is not int or size <= 0 for size in anchors)
    ):
        raise ValueError("The selected five-level anchor configuration is invalid.")
    image_scale = run.get("image_scale")
    if not isinstance(image_scale, (int, float)) or not math.isfinite(image_scale) or image_scale <= 0:
        raise ValueError("The candidate requires a finite positive predeclared image scale.")
    if run.get("baseline_image_scale", 1.5625) != 1.5625:
        raise ValueError("The published USA/v1 baselines must retain image scale 1.5625.")
    default_v1, calibrated_v1 = choose_calibrated_v1(baselines["chullo_v1"])
    if not eligible(selected["report"]["summary"], default_v1["summary"]):
        raise ValueError("The saved selected report does not pass the declared validation gate.")
    checkpoints, hashes = verified_checkpoints()
    if run.get("baseline_checkpoints_sha256") != hashes:
        raise ValueError("The baseline checkpoints differ from those evaluated in development.")
    candidate = run_dir / "best_model.pth"
    if sha256_file(candidate) != selected.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint bytes do not match the validation report.")
    paths = {
        "run_protocol": run_dir / "protocol.json",
        "selected_validation": run_dir / "selected_validation.json",
        "baselines_validation": run_dir / "baselines_validation.json",
        "development_manifest": development_manifest,
        "final_manifest": final_manifest,
        "usa_checkpoint": checkpoints["usa"],
        "v1_checkpoint": checkpoints["chullo_v1"],
        "v2_checkpoint": candidate,
    }
    points = {
        "usa_default": {"score_threshold": 0.5, "mask_threshold": 0.5},
        "v1_default": operating_point(default_v1),
        "v1_calibrated": operating_point(calibrated_v1),
        "v2_selected": operating_point(selected["report"]),
        "v1_at_v2_operating_point": operating_point(selected["report"]),
        "v1_candidate_scale_at_v2_operating_point": operating_point(selected["report"]),
    }
    protocol = {
        "schema_version": 1,
        "phase": "frozen_before_test_access",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": Path(os.path.relpath(path, output_dir)).as_posix(), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "selected_experiment": selected["experiment"],
        "selected_epoch": selected["epoch"],
        "initial_checkpoint_sha256": run.get("base_checkpoint_sha256"),
        "run_training_config": run.get("training_config"),
        "operating_points": points,
        "v1_calibration_selection": "Max validation F0.5 subject to TP >= v1 default and merged predictions/references affected <= default; tie: fewer affected, fewer FP, higher recall, then first grid point.",
        "promotion_rule": {
            "compare": "v2_selected versus v1_default on new holdout",
            "tp": ">=",
            "fp": "<",
            "merged_predictions": "<=",
            "references_affected_by_merges": "<=",
            "failure_policy": "Do not promote; do not tune against this test.",
        },
        "image_scale": image_scale,
        "baseline_image_scale": 1.5625,
        "instance_iou": 0.5,
        "test_scene_ids": [row["scene_id"] for row in final["scenes"] if row["split"] == "test"],
        "evaluation_scope": "new_geographic_holdout",
        "historical_test_used": False,
        "checkpoint_evaluation": "Load the stored checkpoint values into the same FP32 architecture; apply the selected anchors and fixed image scale. No test threshold search.",
        "deployment_difference": "Whole reference scenes, no production tiling, area filtering or cross-tile suppression.",
        "caveat": final.get("assessment_caveat"),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "protocol.json"
    write_json(path, protocol)
    (output_dir / "protocol.sha256").write_text(sha256_file(path) + "\n", encoding="ascii")
    return path


def verify_frozen_protocol(path):
    path = Path(path).resolve()
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").strip()
    if sha256_file(path) != expected:
        raise ValueError("The frozen protocol has changed.")
    protocol = read_json(path)
    if protocol.get("phase") != "frozen_before_test_access":
        raise ValueError("The evaluation requires a previously frozen protocol.")
    resolved = {}
    for name, item in protocol["inputs"].items():
        resolved[name] = (path.parent / item["path"]).resolve()
        if sha256_file(resolved[name]) != item["sha256"]:
            raise ValueError(f"Frozen input changed before test evaluation: {name}")
    return protocol, resolved


def report_differences(candidate, baseline):
    fields = (
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "f0_5",
        "merged_predictions",
        "references_affected_by_merges",
    )
    return {field: candidate[field] - baseline[field] for field in fields}


def conservative_advantage(candidate, baseline):
    no_worse = (
        candidate["tp"] >= baseline["tp"]
        and candidate["fp"] <= baseline["fp"]
        and candidate["merged_predictions"] <= baseline["merged_predictions"]
        and candidate["references_affected_by_merges"] <= baseline["references_affected_by_merges"]
    )
    return no_worse and any(
        candidate[key] != baseline[key] for key in ("tp", "fp", "merged_predictions", "references_affected_by_merges")
    )


def interpret_reports(reports):
    baseline = reports["v1_default"]["summary"]
    candidate = reports["v2_selected"]["summary"]
    calibrated = reports["v1_calibrated"]["summary"]
    scale_control = reports["v1_candidate_scale_at_v2_operating_point"]["summary"]
    calibration_passes = eligible(calibrated, baseline)
    advantage = conservative_advantage(candidate, calibrated)
    if calibration_passes and not advantage:
        note = "La calibración de v1 también cumple el criterio frente a v1 default; no se demuestra una ventaja conservadora adicional de v2 frente a v1 calibrado. No atribuir toda la ventaja al reentrenamiento."
    elif calibration_passes:
        note = "La calibración de v1 también aporta una ventaja; v2 añade una mejora conservadora frente a ese baseline calibrado en este holdout."
    else:
        note = "La calibración de v1 no cumple por sí sola el criterio de promoción frente a v1 default en este holdout; consultar las diferencias frente al calibrado y al mismo punto operativo."
    scale_passes = eligible(scale_control, baseline)
    scale_advantage = conservative_advantage(candidate, scale_control)
    if scale_passes and not scale_advantage:
        note += " V1 con la escala y los umbrales de v2 también cumple el criterio; no se demuestra una ventaja conservadora adicional atribuible a los nuevos pesos."
    return {
        "promotion_gate_passed": eligible(candidate, baseline),
        "calibrated_v1_passes_default_gate": calibration_passes,
        "v2_adds_conservative_advantage_over_calibrated_v1": advantage,
        "v1_candidate_scale_passes_default_gate": scale_passes,
        "v2_adds_conservative_advantage_over_v1_at_same_scale_and_thresholds": scale_advantage,
        "differences_v2_minus": {
            name: report_differences(candidate, reports[name]["summary"])
            for name in (
                "usa_default",
                "v1_default",
                "v1_calibrated",
                "v1_at_v2_operating_point",
                "v1_candidate_scale_at_v2_operating_point",
            )
        },
        "interpretation": note,
        "action": "Report the fixed holdout outcome. This function never changes the application default or selects new thresholds.",
    }


def evaluate_frozen(protocol_path, device="cuda"):
    """One final evaluation; reject edits and repeated access to this holdout run."""
    protocol_path = Path(protocol_path).resolve()
    protocol, paths = verify_frozen_protocol(protocol_path)
    output_dir = protocol_path.parent
    with (output_dir / "evaluation_started.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"started_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256_file(protocol_path)},
            stream,
        )
    try:
        manifest = read_manifest(paths["final_manifest"])
        if [row["scene_id"] for row in manifest["scenes"] if row["split"] == "test"] != protocol["test_scene_ids"]:
            raise ValueError("Test identities differ from the frozen protocol.")
        # Image/label bytes are accessed only after the immutable protocol and
        # single-evaluation marker exist. Validate the exact test input bytes.
        test_rows = [row for row in manifest["scenes"] if row["split"] == "test"]
        verify_development_artifacts({"scenes": test_rows})
        dataset = BuildingDataset(manifest, "test")
        reports, executions = {}, {}
        baseline_scale, candidate_scale = protocol["baseline_image_scale"], protocol["image_scale"]
        v1_names = ["v1_default", "v1_calibrated", "v1_at_v2_operating_point"]
        v1_groups = [(baseline_scale, v1_names)]
        if candidate_scale == baseline_scale:
            v1_names.append("v1_candidate_scale_at_v2_operating_point")
        else:
            v1_groups.append((candidate_scale, ["v1_candidate_scale_at_v2_operating_point"]))
        groups = (
            ("usa_checkpoint", [(baseline_scale, ["usa_default"])], [32, 64, 128, 256, 512]),
            ("v1_checkpoint", v1_groups, [32, 64, 128, 256, 512]),
            ("v2_checkpoint", [(candidate_scale, ["v2_selected"])], protocol["selected_experiment"]["anchors"]),
        )
        for key, scale_groups, anchors in groups:
            model, actual_device = load_model(paths[key], device)
            configure_anchors(model, anchors)
            for scale, names in scale_groups:
                predictions = predict_scenes(model, dataset, actual_device, scale=scale)
                for name in names:
                    point = protocol["operating_points"][name]
                    reports[name] = score_predictions(predictions, point["score_threshold"], point["mask_threshold"])
                del predictions
            executions[key] = {
                "device": actual_device,
                "anchors": anchors,
                "image_scales": [scale for scale, _ in scale_groups],
            }
            del model
        report = {
            "status": "completed",
            "scope": "new_geographic_holdout",
            "historical_test_used": False,
            "protocol_sha256": sha256_file(protocol_path),
            "test_scene_ids": protocol["test_scene_ids"],
            "reports": {name: reports[name] for name in REPORT_NAMES},
            "comparison": interpret_reports(reports),
            "execution": executions,
            "caveat": protocol.get("caveat"),
            "deployment_difference": protocol["deployment_difference"],
        }
        write_json(output_dir / "new_holdout_report.json", report)
        return report
    except Exception as error:
        write_json(output_dir / "evaluation_failure.json", {"status": "failed", "error": str(error)})
        raise
