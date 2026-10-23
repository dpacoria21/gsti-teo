"""Normalize provenance for exports and presentation without UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def source_records(metadata: Any) -> list:
    if isinstance(metadata, list):
        return metadata
    if isinstance(metadata, dict):
        sources = metadata.get("sources", [])
        return list(sources.values()) if isinstance(sources, dict) else sources
    return []


def image_details(image: dict) -> dict:
    """Surface the original image manifest retained by the inference worker."""
    provenance = image.get("provenance", {})
    return {**(provenance if isinstance(provenance, dict) else {}), **image}


def approximate_georeferencing(image: dict) -> bool:
    details = image_details(image)
    description = " ".join(
        str(details.get(key, "")) for key in ("source_id", "source", "title", "georeferencing")
    ).lower()
    return "google" in description or "aproximad" in description or "approximate" in description


def detection_metadata(base: dict, detection: dict) -> dict:
    metadata = dict(base)
    metadata.update({key: value for key, value in detection.get("metadata", {}).items() if key != "sources"})
    records = source_records(detection.get("metadata", {}))
    metadata["sources"] = list(
        {
            source.get("source_id", source.get("title", str(index))): source for index, source in enumerate(records)
        }.values()
    )
    metadata["detection"] = detection["manifest"]
    metadata["building_mode"] = (
        "finetuned_model_inference"
        if detection["manifest"].get("model", {}).get("trained_locally")
        else "pretrained_model_inference"
    )
    return metadata


@dataclass(frozen=True)
class DetectionDescription:
    """Human-readable facts taken from a saved inference, never the current controls."""

    model_name: str
    image_info: dict[str, Any]
    observed_at: str
    applied_threshold: float | None


def describe_detection(manifest: dict) -> DetectionDescription:
    model = manifest.get("model", {})
    model_name = (
        model.get("name")
        or model.get("architecture")
        or model.get("id")
        or manifest.get("model_id", "Modelo preentrenado")
    )
    image_info = image_details(manifest.get("image", {}))
    observed_at = image_info.get("observed_at") or image_info.get("capture_date") or "No informada por la fuente"
    threshold = manifest.get(
        "threshold",
        manifest.get("parameters", {}).get(
            "confidence_threshold",
            manifest.get("parameters", {}).get("threshold", manifest.get("confidence_threshold")),
        ),
    )
    return DetectionDescription(model_name, image_info, observed_at, threshold)
