"""Archive completed validation experiments without promoting a checkpoint."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("chullo_refinement_v2", "chullo_refinement_v2_warm", "chullo_refinement_v2_detail")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(destination):
    if destination.exists():
        raise FileExistsError("Use a new report folder; completed evidence is immutable.")
    rows, inventory = [], []
    for name in RUNS:
        directory = ROOT / "outputs/training" / name
        protocol, history = read(directory / "protocol.json"), read(directory / "epochs.json")
        if protocol["status"] != "completed" or protocol["test_opened"] is not False:
            raise ValueError("Only completed validation-only runs can be summarized.")
        if protocol["validation_gate_passed"] is not False or protocol["selected"] is not None:
            raise ValueError("This non-promotion report requires all runs to have failed the gate.")
        inventory.append(
            {
                "run": name,
                "epochs": len(history),
                "optimizer_updates": sum(row["optimizer_updates"] for row in history),
                "skipped_updates": sum(
                    max(row["skipped_total"] for row in history if row["experiment"] == experiment)
                    for experiment in {row["experiment"] for row in history}
                ),
                "elapsed_training_seconds": protocol["elapsed_seconds"],
                "validation_gate_passed": protocol["validation_gate_passed"],
                "selected": protocol["selected"],
            }
        )
        for epoch in history:
            rows.extend(
                {
                    "run": name,
                    "experiment": epoch["experiment"],
                    "epoch": epoch["epoch"],
                    "image_scale": protocol["image_scale"],
                    **report,
                }
                for report in epoch.get("validation_grid", [])
            )
    baselines = read(ROOT / "outputs/training" / RUNS[-1] / "baselines_validation.json")
    baseline = next(r for r in baselines["chullo_v1"] if r["score_threshold"] == 0.5 and r["mask_threshold"] == 0.5)
    summary = {
        "status": "completed_no_candidate_approved",
        "annotation_method": "assistant_visual_manual",
        "human_reviewed": False,
        "production_model_changed": False,
        "new_test_evaluated": False,
        "new_weights_registered": False,
        "total_experiments": 7,
        "total_epochs": sum(r["epochs"] for r in inventory),
        "total_optimizer_updates": sum(r["optimizer_updates"] for r in inventory),
        "runs": inventory,
        "baseline_validation": baseline,
        "exploratory_highest_validation_f1": max(rows, key=lambda r: r["summary"]["f1"]),
        "exploratory_highest_validation_f0_5": max(rows, key=lambda r: r["summary"]["f0_5"]),
        "interpretation": "No tested configuration met all predeclared validation criteria. Exploratory maxima are post-hoc descriptions, not selected or stored checkpoints. The historical test F1 0.24885 has not been superseded. The new 53-reference test remains unevaluated.",
        "next_evidence_needed": "Human review of visible-roof boundaries, additional legally usable independent imagery and spatially separated reference annotations. The current labels do not establish individual houses or cadastral limits.",
        "artifact_sha256": {},
    }
    destination.mkdir(parents=True)
    for name in RUNS:
        target = destination / name
        target.mkdir()
        for filename in ("protocol.json", "epochs.json", "baselines_validation.json"):
            source = ROOT / "outputs/training" / name / filename
            shutil.copyfile(source, target / filename)
            summary["artifact_sha256"][f"{name}/{filename}"] = checksum(target / filename)
    audit = destination / "label_audit"
    audit.mkdir()
    for filename in (
        "audit.json",
        "AUDITORIA_ETIQUETAS.md",
        "holdout_freeze.json",
        "validation_review.md",
        "train_1100_500_v2_review.png",
        "holdout_300_1600_annotation_qa.png",
        "holdout_2100_1100_annotation_qa.png",
    ):
        source = ROOT / "outputs/label_audit_v2" / filename
        shutil.copyfile(source, audit / filename)
        summary["artifact_sha256"][f"label_audit/{filename}"] = checksum(audit / filename)
    (destination / "resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    summarize(ROOT / "reports/training/chullo_refinement_v2")
