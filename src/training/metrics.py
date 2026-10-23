"""Instance-mask metrics at fixed IoU/score thresholds, deliberately not COCO AP."""

from __future__ import annotations

import numpy as np


def merged_instances(predicted: np.ndarray, reference: np.ndarray, coverage: float = 0.5) -> dict:
    """Count predictions covering at least half of two separate reference roofs.

    This diagnostic detects fusion at a fixed operating point, independently of
    one-to-one IoU matching. It is not a cadastral or semantic house count.
    """
    predicted, reference = np.asarray(predicted, dtype=bool), np.asarray(reference, dtype=bool)
    if predicted.ndim != 3 or reference.ndim != 3 or predicted.shape[1:] != reference.shape[1:]:
        raise ValueError("Las máscaras deben compartir dimensiones [N,H,W].")
    if not 0 < coverage <= 1:
        raise ValueError("coverage debe estar en (0,1].")
    areas = reference.sum(axis=(1, 2))
    merged, affected = 0, set()
    for mask in predicted:
        intersection = reference[:, mask].sum(axis=1)
        fractions = np.divide(intersection, areas, out=np.zeros(len(reference)), where=areas > 0)
        covered = np.flatnonzero(fractions >= coverage)
        if len(covered) >= 2:
            merged += 1
            affected.update(covered.tolist())
    return {"merged_predictions": merged, "references_affected_by_merges": len(affected)}


def match_masks(predicted: np.ndarray, reference: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Greedy maximum-IoU one-to-one matching, with deterministic tie ordering."""
    predicted, reference = np.asarray(predicted, dtype=bool), np.asarray(reference, dtype=bool)
    if predicted.ndim != 3 or reference.ndim != 3 or predicted.shape[1:] != reference.shape[1:]:
        raise ValueError("Las máscaras deben tener dimensiones [instancias, alto, ancho] coincidentes.")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold debe estar en (0,1].")
    predicted_area = predicted.sum(axis=(1, 2))
    reference_area = reference.sum(axis=(1, 2))
    candidates = []
    for prediction_id, mask in enumerate(predicted):
        intersection = reference[:, mask].sum(axis=1)
        union = predicted_area[prediction_id] + reference_area - intersection
        ious = np.divide(intersection, union, out=np.zeros(len(reference)), where=union > 0)
        candidates.extend(
            (float(ious[reference_id]), prediction_id, int(reference_id))
            for reference_id in np.flatnonzero(ious >= iou_threshold)
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_predictions, matched_references, matched_ious = set(), set(), []
    for iou, prediction_id, reference_id in candidates:
        if prediction_id not in matched_predictions and reference_id not in matched_references:
            matched_predictions.add(prediction_id)
            matched_references.add(reference_id)
            matched_ious.append(iou)
    tp = len(matched_ious)
    return {
        "tp": tp,
        "fp": len(predicted) - tp,
        "fn": len(reference) - tp,
        "matched_iou_sum": float(sum(matched_ious)),
    }


def aggregate_counts(rows: list[dict]) -> dict:
    tp, fp, fn = (int(sum(row[key] for row in rows)) for key in ("tp", "fp", "fn"))
    return {
        "images": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "mean_iou_true_positives": sum(row["matched_iou_sum"] for row in rows) / tp if tp else None,
    }


def paired_bootstrap(baseline: list[dict], candidate: list[dict], seed=42, samples=1000) -> dict:
    """Paired image bootstrap; intervals do not account for spatial autocorrelation."""
    if not baseline or [row["scene_id"] for row in baseline] != [row["scene_id"] for row in candidate]:
        raise ValueError("La comparación requiere las mismas escenas en el mismo orden.")
    rng = np.random.default_rng(seed)
    deltas = []
    for indices in rng.integers(0, len(baseline), size=(samples, len(baseline))):
        base_score = aggregate_counts([baseline[index] for index in indices])["f1"]
        candidate_score = aggregate_counts([candidate[index] for index in indices])["f1"]
        deltas.append(candidate_score - base_score)
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": "candidate_f1_minus_baseline_f1",
        "samples": samples,
        "seed": seed,
        "delta": aggregate_counts(candidate)["f1"] - aggregate_counts(baseline)["f1"],
        "percentile_95_interval": [float(lower), float(upper)],
        "unit": "scene",
        "caveat": "Exploratory paired image bootstrap; spatial dependence is not modelled.",
    }
