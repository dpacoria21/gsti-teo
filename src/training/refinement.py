"""Validation-only local experiments with explicit false-positive/fusion gates.

The test partition is never loaded here. Every evaluated candidate is rounded
to the exact FP16 checkpoint representation before inference in FP32.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.anchor_utils import AnchorGenerator

from src.inference.model import MODEL_PATH, MODEL_SHA256, load_model, resolve_variant, sha256_file
from src.model_catalog import LOCAL_VARIANT
from src.training.dataset import BuildingDataset, collate_scenes, read_manifest
from src.training.ignore import enable_training_ignore
from src.training.metrics import aggregate_counts, match_masks, merged_instances
from src.training.runner import TrainingConfig, configure_image_scale

SCORES = (0.5, 0.65, 0.8, 0.9)
MASKS = (0.5, 0.6)
EXPERIMENTS = (
    {"name": "full_context", "anchors": [32, 64, 128, 256, 512], "unfreeze_layer4": False},
    {"name": "small_anchors", "anchors": [16, 32, 64, 128, 256], "unfreeze_layer4": False},
    {"name": "adapt_layer4", "anchors": [16, 32, 64, 128, 256], "unfreeze_layer4": True},
)
WARM_START_EXPERIMENTS = (
    {"name": "local_conservative", "anchors": [32, 64, 128, 256, 512], "unfreeze_layer4": False},
    {"name": "local_adapt_layer4", "anchors": [32, 64, 128, 256, 512], "unfreeze_layer4": True},
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def configure_anchors(model, sizes):
    model.rpn.anchor_generator = AnchorGenerator(tuple((int(size),) for size in sizes), ((0.5, 1.0, 2.0),) * 5)


def predict_scenes(model, dataset, device, scale=1.5625):
    model.eval()
    config = TrainingConfig(image_scale=scale)
    result = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            configure_image_scale(model, image, config)
            prediction = model([image.to(device)])[0]
            chosen = prediction["labels"] == 1
            result.append(
                {
                    "scene_id": dataset.scenes[index]["scene_id"],
                    "scores": prediction["scores"][chosen].cpu().numpy(),
                    "probabilities": prediction["masks"][chosen, 0].cpu().numpy(),
                    "valid": target["valid_mask"].numpy(),
                    "reference": target["masks"].numpy(),
                }
            )
    return result


def score_predictions(predictions, score, mask_threshold):
    rows = []
    for scene in predictions:
        masks = scene["probabilities"][scene["scores"] >= score] >= mask_threshold
        masks &= scene["valid"][None]
        masks = masks[masks.any(axis=(1, 2))]
        rows.append(
            {
                "scene_id": scene["scene_id"],
                **match_masks(masks, scene["reference"], 0.5),
                **merged_instances(masks, scene["reference"]),
                "reference_instances": len(scene["reference"]),
                "predicted_instances": len(masks),
            }
        )
    summary = aggregate_counts(rows)
    summary.update(
        {key: sum(row[key] for row in rows) for key in ("merged_predictions", "references_affected_by_merges")}
    )
    denominator = 1.25 * summary["tp"] + 0.25 * summary["fn"] + summary["fp"]
    summary["f0_5"] = 1.25 * summary["tp"] / denominator if denominator else 0.0
    return {"score_threshold": score, "mask_threshold": mask_threshold, "summary": summary, "per_scene": rows}


def eligible(summary, baseline):
    """Require fewer FP without losing TP or increasing merged predictions."""
    return (
        summary["tp"] >= baseline["tp"]
        and summary["fp"] < baseline["fp"]
        and summary["merged_predictions"] <= baseline["merged_predictions"]
        and summary["references_affected_by_merges"] <= baseline["references_affected_by_merges"]
    )


def selection_key(report):
    s = report["summary"]
    return (s["f0_5"], -s["references_affected_by_merges"], -s["fp"], s["recall"])


def verify_development_artifacts(manifest):
    """Validate the bytes that will be consumed, not only the manifest itself."""
    for scene in manifest["scenes"]:
        for key, hash_key in (
            ("image", "image_sha256"),
            ("labels", "labels_sha256"),
            ("ignore_regions", "ignore_sha256"),
        ):
            if key in scene and sha256_file(scene[key]) != scene.get(hash_key):
                raise ValueError(f"Development artifact does not match its SHA256: {scene['scene_id']} {key}")


def verified_checkpoints():
    if not MODEL_PATH.is_file() or sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise ValueError("The local USA initialization must match its pinned SHA256; no model is downloaded.")
    local_path, local_metadata = resolve_variant(LOCAL_VARIANT)
    return {"usa": MODEL_PATH, "chullo_v1": local_path}, {"usa": MODEL_SHA256, "chullo_v1": local_metadata["sha256"]}


def run(manifest_path, output_dir, epochs=80, suite="from_usa"):
    if suite not in {"from_usa", "warm_start", "detail_scale"}:
        raise ValueError("Unknown experiment suite.")
    warm_start = suite == "warm_start"
    detail_scale = suite == "detail_scale"
    experiments = WARM_START_EXPERIMENTS if warm_start else (EXPERIMENTS[:2] if detail_scale else EXPERIMENTS)
    snapshot_every = 2 if warm_start else 5
    head_lr, body_lr = (2e-5, 2e-6) if warm_start else (1e-4, 1e-5)
    image_scale = 3.125 if detail_scale else 1.5625
    scores = (0.25, 0.35, *SCORES) if detail_scale else SCORES
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = read_manifest(manifest_path, required_splits=("train", "val"))
    verify_development_artifacts(manifest)
    checkpoints, checkpoint_hashes = verified_checkpoints()
    initial_name = "chullo_v1" if warm_start else "usa"
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded local experiment requires the available CUDA device.")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = "cuda"
    validation = BuildingDataset(manifest, "val")
    train = BuildingDataset(manifest, "train", augment=True, allow_training_ignore=True)
    protocol = {
        "status": "running",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": str(manifest_path),
        "seed": 42,
        "epochs": epochs,
        "suite": suite,
        "experiments": experiments,
        "score_grid": scores,
        "mask_grid": MASKS,
        "snapshot_every_epochs": snapshot_every,
        "selection": "Max F0.5 among candidates with TP >= v1 default, FP < v1 default, merged predictions and references affected <= v1 default, on validation only; earliest tie kept.",
        "test_opened": False,
        "checkpoint_dtype": "FP16; evaluation reloads those values into FP32",
        "initial_model": initial_name,
        "base_checkpoint_sha256": checkpoint_hashes[initial_name],
        "baseline_checkpoints_sha256": checkpoint_hashes,
        "image_scale": image_scale,
        "baseline_image_scale": 1.5625,
        "training": f"Full parent windows, ignore-aware losses, flips/quarter rotations, explicit negative scenes; layer4 adaptation begins after epoch10 at LR {body_lr}; heads LR {head_lr}; AdamW wd1e-4; gradientclip1; AMP; batch1.",
        "development_history": (
            "Second validation-only round: the initial three USA-initialized experiments failed the unchanged promotion gate. This round retains local v1 knowledge with a lower learning rate. No new test predictions informed either round."
            if warm_start
            else (
                "Third and final validation-only round after five candidates failed the unchanged gate: enlarge small roofs for the network and include lower confidence thresholds because negative examples may recalibrate scores. No test predictions used. Enlarging adds no new source detail."
                if detail_scale
                else "First validation-only round."
            )
        ),
        "caveat": manifest.get("assessment_caveat"),
    }
    write_json(output_dir / "protocol.json", protocol)
    baselines = {}
    for name, checkpoint in checkpoints.items():
        model, _ = load_model(checkpoint, device)
        predictions = predict_scenes(model, validation, device)
        baselines[name] = [score_predictions(predictions, score, mask) for score in scores for mask in MASKS]
        if detail_scale:
            enlarged = predict_scenes(model, validation, device, image_scale)
            baselines[name + "_candidate_scale"] = [
                score_predictions(enlarged, score, mask) for score in scores for mask in MASKS
            ]
            del enlarged
        del model, predictions
    write_json(output_dir / "baselines_validation.json", baselines)
    baseline = next(
        report["summary"]
        for report in baselines["chullo_v1"]
        if report["score_threshold"] == 0.5 and report["mask_threshold"] == 0.5
    )
    history, best = [], None
    clock_start = time.perf_counter()
    for experiment in experiments:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        model, _ = load_model(checkpoints[initial_name], device)
        evaluator, _ = load_model(checkpoints[initial_name], device)
        configure_anchors(model, experiment["anchors"])
        configure_anchors(evaluator, experiment["anchors"])
        enable_training_ignore(model)
        body_ids = {id(p) for p in model.backbone.body.parameters()}
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.parameters() if id(p) not in body_ids], "lr": head_lr},
                {"params": list(model.backbone.body.parameters()), "lr": body_lr},
            ],
            weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=8192)
        loader = DataLoader(
            train,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_scenes,
            generator=torch.Generator().manual_seed(42),
        )
        config = TrainingConfig(image_scale=image_scale)
        skipped = 0
        for epoch in range(1, epochs + 1):
            model.backbone.body.requires_grad_(False)
            if experiment["unfreeze_layer4"] and epoch > 10:
                model.backbone.body.layer4.requires_grad_(True)
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            total_loss, updates = 0.0, 0
            for images, targets in loader:
                configure_image_scale(model, images[0], config)
                images = [im.to(device) for im in images]
                targets = [
                    {key: value.to(device) for key, value in target.items() if key != "valid_mask"}
                    for target in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss {experiment['name']} epoch{epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    skipped += 1
                else:
                    updates += 1
                    total_loss += float(loss.detach())
                scaler.step(optimizer)
                scaler.update()
            row = {
                "experiment": experiment["name"],
                "epoch": epoch,
                "optimizer_updates": updates,
                "loss": total_loss / max(updates, 1),
                "skipped_total": skipped,
            }
            if epoch % snapshot_every == 0:
                state = {
                    key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
                evaluator.load_state_dict(state, strict=True)
                predictions = predict_scenes(evaluator, validation, device, image_scale)
                reports = [score_predictions(predictions, score, mask) for score in scores for mask in MASKS]
                row["validation_grid"] = reports
                qualifying = [report for report in reports if eligible(report["summary"], baseline)]
                for report in qualifying:
                    if best is None or selection_key(report) > selection_key(best["report"]):
                        best = {"experiment": experiment, "epoch": epoch, "report": report}
                        torch.save(state, output_dir / "best_model.pth")
                        best["checkpoint_sha256"] = sha256_file(output_dir / "best_model.pth")
                        write_json(output_dir / "selected_validation.json", best)
                row["eligible_operating_points"] = len(qualifying)
                print(
                    json.dumps(
                        {
                            "experiment": experiment["name"],
                            "epoch": epoch,
                            "loss": row["loss"],
                            "eligible": len(qualifying),
                            "best": best["report"]["summary"] if best else None,
                        }
                    ),
                    flush=True,
                )
                del predictions, state
            history.append(row)
            write_json(output_dir / "epochs.json", history)
        del model, evaluator, optimizer, scaler, loader
        torch.cuda.empty_cache()
    protocol.update(
        status="completed",
        elapsed_seconds=time.perf_counter() - clock_start,
        selected=best,
        validation_gate_passed=best is not None,
        training_config={
            "epochs_per_experiment": epochs,
            "batch_size": 1,
            "device": device,
            "image_scale": config.image_scale,
            "head_learning_rate": head_lr,
            "layer4_learning_rate": body_lr,
            "layer4_unfreeze_after_epoch": 10,
            "layer4_unfreeze_experiment": "local_adapt_layer4" if warm_start else "adapt_layer4",
            "other_backbone_parameters_frozen": True,
            "batch_normalization_statistics_frozen": True,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "amp": True,
        },
    )
    write_json(output_dir / "protocol.json", protocol)
    return protocol
