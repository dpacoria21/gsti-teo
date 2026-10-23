"""Reproducible offline fine-tuning and held-out baseline/candidate comparison."""

from __future__ import annotations

import json
import math
import random
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.inference.model import MODEL_PATH, MODEL_SHA256, load_model, sha256_file
from src.training.dataset import BuildingDataset, collate_scenes, read_manifest
from src.training.metrics import aggregate_counts, match_masks, merged_instances, paired_bootstrap


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    seed: int = 42
    freeze_backbone_epochs: int = 5
    workers: int = 0
    device: str = "auto"
    amp: bool = True
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    match_iou: float = 0.5
    gradient_clip: float = 1.0
    bootstrap_samples: int = 1000
    image_scale: float | None = None

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.workers < 0 or self.freeze_backbone_epochs < 0:
            raise ValueError("epochs/batch_size deben ser positivos; workers/freeze_backbone_epochs no negativos.")
        for value in (self.learning_rate, self.weight_decay, self.gradient_clip):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Learning rate, weight decay y gradient clip deben ser positivos y finitos.")
        if any(not 0 < value <= 1 for value in (self.score_threshold, self.mask_threshold, self.match_iou)):
            raise ValueError("Los umbrales deben estar en (0,1].")
        if self.device not in {"auto", "cuda", "cpu"} or self.bootstrap_samples < 1:
            raise ValueError("Dispositivo o número de muestras bootstrap inválido.")
        if self.image_scale is not None and (
            not math.isfinite(self.image_scale) or self.image_scale <= 0 or self.batch_size != 1
        ):
            raise ValueError("image_scale debe ser positivo y finito y requiere batch_size=1.")


def configure_image_scale(model, image, config):
    """Keep roof scale equal across small annotation chips and 512px production tiles."""
    if config.image_scale is None:
        return
    height, width = image.shape[-2:]
    model.transform.min_size = (max(1, round(min(height, width) * config.image_scale)),)
    model.transform.max_size = max(1, round(max(height, width) * config.image_scale))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class RunLog:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, event, **values):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if event in {"start", "baseline_val", "epoch", "completed", "failed"} or (
            event == "step" and values.get("step", 0) % 20 == 0
        ):
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def evaluate_model(model, dataset, device, config, log=None, phase="evaluation"):
    """Evaluate whole unaugmented scenes at fixed operating point in float32."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            configure_image_scale(model, image, config)
            prediction = model([image.to(device)])[0]
            keep = (prediction["scores"] >= config.score_threshold) & (prediction["labels"] == 1)
            masks = prediction["masks"][keep, 0].detach().cpu().numpy() >= config.mask_threshold
            masks &= target["valid_mask"].numpy()[None, :, :]
            masks = masks[masks.any(axis=(1, 2))]
            counts = match_masks(masks, target["masks"].numpy(), config.match_iou)
            limit = getattr(getattr(model, "roi_heads", None), "detections_per_img", None)
            row = {
                "scene_id": dataset.scenes[index]["scene_id"],
                **counts,
                **merged_instances(masks, target["masks"].numpy()),
                "reference_instances": len(target["masks"]),
                "predicted_instances": len(masks),
                "detection_limit": limit,
                "reference_count_exceeds_detection_limit": bool(limit and len(target["masks"]) > limit),
            }
            rows.append(row)
            if log:
                log("evaluation_image", phase=phase, index=index + 1, total=len(dataset), **row)
    summary = aggregate_counts(rows)
    for key in ("merged_predictions", "references_affected_by_merges"):
        summary[key] = sum(row[key] for row in rows)
    return {"summary": summary, "per_scene": rows}


def run_training(manifest_path, output_dir, config=TrainingConfig(), checkpoint=MODEL_PATH):
    """Train a candidate in a new directory; test never influences epoch selection."""
    manifest_path, output_dir, checkpoint = Path(manifest_path), Path(output_dir), Path(checkpoint)
    manifest = read_manifest(manifest_path)
    if not checkpoint.is_file():
        raise ValueError("Se necesitan pesos locales; este entrenamiento no descarga modelos automáticamente.")
    initial_sha = sha256_file(checkpoint)
    if checkpoint.resolve() == MODEL_PATH.resolve() and initial_sha != MODEL_SHA256:
        raise ValueError("El checkpoint base fijado no coincide con su SHA256.")
    output_dir.mkdir(parents=True, exist_ok=False)
    log = RunLog(output_dir / "events.jsonl")
    started = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings("once", message="roi_align_backward_kernel does not have a deterministic implementation.*")
    run = {
        "status": "running",
        "config": asdict(config),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "annotation_type": "reference",
        "initial_checkpoint": str(checkpoint.resolve()),
        "initial_checkpoint_sha256": initial_sha,
        "architecture": "torchvision.maskrcnn_resnet50_fpn",
        "num_classes": 2,
        "resize": {"image_scale": config.image_scale} if config.image_scale else {"min_size": 800, "max_size": 1333},
        "training_images": "Prepared manifest scenes/chips; random quarter rotations and horizontal/vertical flips",
        "annotation_provenance": manifest.get("annotation_provenance"),
        "input_preprocessing": "uint8 RGB /255; raw uint16 fallback valid-pixel per-band percentiles 2/98",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "selection": "Highest validation instance F1; earliest epoch wins ties; test unused until final comparison",
        "checkpoint_storage_dtype": "float16 for floating tensors; original dtype for integer tensors",
        "inference_dtype": "float32 after strict state-dict loading",
        "evaluation_protocol": {
            "images": "Manifest scenes without inference tiling or area filtering; resize as registered above",
            "nodata": "Reference/prediction masks clipped to identical valid area, excluding declared uncertain regions; RGB unchanged",
            "fusion_diagnostic": "Prediction covers >=50% of each of >=2 distinct reference roofs",
            "max_detections_per_image": 100,
            "min_reference_pixels": 4,
            "deployment_difference": "Production uses 512px overlapping tiles and area/duplicate filtering",
        },
        "trainable_components": "FPN, RPN and ROI box/mask heads; ResNet body frozen for freeze_backbone_epochs",
        "batch_normalization": "Running statistics frozen throughout training because the default batch size is one",
        "metric": "Instance-mask precision/recall/F1 at fixed score and IoU thresholds; not COCO AP",
        "split_counts": {
            split: sum(scene["split"] == split for scene in manifest["scenes"]) for split in ("train", "val", "test")
        },
        "reproducibility": "Seeded RNGs; deterministic algorithms warn_only; cross-device bitwise equality not promised",
    }
    _write_json(output_dir / "run.json", run)
    try:
        model, device = load_model(checkpoint, config.device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        run["execution"] = {
            "device": device,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        }
        _write_json(output_dir / "run.json", run)
        log("start", config=asdict(config), execution=run["execution"], split_counts=run["split_counts"])
        train = BuildingDataset(manifest, "train", augment=True)
        validation = BuildingDataset(manifest, "val")
        test = BuildingDataset(manifest, "test")
        baseline_val = evaluate_model(model, validation, device, config, log, "baseline_val")
        _write_json(output_dir / "baseline_val.json", baseline_val)
        log("baseline_val", **baseline_val["summary"])
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            collate_fn=collate_scenes,
            generator=generator,
            pin_memory=device == "cuda",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        use_amp = config.amp and device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=8192)
        best_score, best_epoch, history = -1.0, None, []
        skipped_amp_steps = 0
        best_path = output_dir / "best_model.pth"
        for epoch in range(1, config.epochs + 1):
            frozen = epoch <= config.freeze_backbone_epochs
            model.backbone.body.requires_grad_(not frozen)
            model.train()
            if frozen:
                model.backbone.body.eval()
            # Batch size one is unsuitable for updating backbone batch-normalization statistics.
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            loss_sum = 0.0
            epoch_updates = 0
            consecutive_amp_skips = 0
            epoch_start = time.perf_counter()
            for step, (images, targets) in enumerate(loader, start=1):
                configure_image_scale(model, images[0], config)
                images = [image.to(device) for image in images]
                targets = [
                    {key: value.to(device) for key, value in target.items() if key != "valid_mask"}
                    for target in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Pérdida no finita en epoch={epoch}, step={step}.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                if not torch.isfinite(gradient_norm):
                    if not use_amp:
                        raise RuntimeError(f"Gradiente no finito en epoch={epoch}, step={step}.")
                    # GradScaler registered the overflow during unscale_; step safely skips the update.
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped_amp_steps += 1
                    consecutive_amp_skips += 1
                    log(
                        "amp_overflow",
                        epoch=epoch,
                        step=step,
                        loss=float(loss.detach()),
                        previous_scale=previous_scale,
                        new_scale=scaler.get_scale(),
                        skipped_amp_steps=skipped_amp_steps,
                    )
                    if consecutive_amp_skips >= 10:
                        raise RuntimeError("AMP produjo 10 desbordamientos consecutivos; revisar entrenamiento.")
                    continue
                scaler.step(optimizer)
                scaler.update()
                consecutive_amp_skips = 0
                epoch_updates += 1
                loss_value = float(loss.detach())
                loss_sum += loss_value
                log(
                    "step",
                    epoch=epoch,
                    step=step,
                    total_steps=len(loader),
                    loss=loss_value,
                    components={key: float(value.detach()) for key, value in losses.items()},
                    gradient_norm=float(gradient_norm),
                    backbone_frozen=frozen,
                )
            validation_result = evaluate_model(model, validation, device, config, log, f"val_epoch_{epoch}")
            score = validation_result["summary"]["f1"]
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
                temporary = best_path.with_suffix(".temporary")
                state = {
                    key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
                torch.save(state, temporary)
                del state
                temporary.replace(best_path)
                _write_json(output_dir / "best_val.json", validation_result)
            epoch_record = {
                "epoch": epoch,
                "mean_training_loss": loss_sum / epoch_updates if epoch_updates else None,
                "optimizer_updates": epoch_updates,
                "skipped_amp_steps_total": skipped_amp_steps,
                "validation": validation_result["summary"],
                "selected_as_best": improved,
                "backbone_frozen": frozen,
                "seconds": time.perf_counter() - epoch_start,
            }
            history.append(epoch_record)
            _write_json(output_dir / "epochs.json", history)
            log("epoch", **epoch_record)
        # Only after validation selected a checkpoint do we open the held-out test set.
        baseline_model, _ = load_model(checkpoint, "cpu")
        model.load_state_dict(baseline_model.state_dict(), strict=True)
        del baseline_model
        baseline_test = evaluate_model(model, test, device, config, log, "baseline_test")
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
        stored_candidate_val = evaluate_model(model, validation, device, config, log, "stored_candidate_val")
        _write_json(output_dir / "best_val_stored_checkpoint.json", stored_candidate_val)
        candidate_test = evaluate_model(model, test, device, config, log, "candidate_test")
        _write_json(output_dir / "baseline_test.json", baseline_test)
        _write_json(output_dir / "candidate_test.json", candidate_test)
        comparison = {
            "baseline": baseline_test["summary"],
            "candidate": candidate_test["summary"],
            "paired_image_bootstrap": paired_bootstrap(
                baseline_test["per_scene"], candidate_test["per_scene"], config.seed, config.bootstrap_samples
            ),
            "selection_used_test": False,
            "interpretation": manifest.get(
                "assessment_caveat", "Held-out dataset scenes only; this does not establish accuracy in Chullo."
            ),
        }
        _write_json(output_dir / "comparison.json", comparison)
        run.update(
            {
                "status": "completed",
                "best_epoch": best_epoch,
                "best_validation_f1": best_score,
                "baseline_validation_f1": baseline_val["summary"]["f1"],
                "stored_checkpoint_validation_f1": stored_candidate_val["summary"]["f1"],
                "validation_improved": best_score > baseline_val["summary"]["f1"],
                "checkpoint": str(best_path.resolve()),
                "checkpoint_sha256": sha256_file(best_path),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_cuda_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
                if device == "cuda"
                else None,
                "production_model_modified": False,
                "skipped_amp_steps": skipped_amp_steps,
            }
        )
        _write_json(output_dir / "run.json", run)
        log(
            "completed",
            best_epoch=best_epoch,
            best_validation_f1=best_score,
            comparison=comparison,
            elapsed_seconds=run["elapsed_seconds"],
        )
        return run
    except Exception as error:
        run.update({"status": "failed", "error": str(error), "elapsed_seconds": time.perf_counter() - started})
        _write_json(output_dir / "run.json", run)
        log("failed", error=str(error))
        raise
