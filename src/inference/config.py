"""Validated execution settings, independent of model weights and runtime state."""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import CRS


@dataclass(frozen=True)
class InferenceParameters:
    """Defaults reproduce the saved prototype runs; thresholds are not risk scores."""

    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5
    tile_size: int = 512
    overlap: int = 128
    min_area_m2: float = 10.0
    analysis_crs: str = "EPSG:32719"
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0.05 <= self.confidence_threshold <= 0.99 or not 0.05 <= self.mask_threshold <= 0.95:
            raise ValueError("Umbral de confianza o máscara fuera del rango admitido.")
        if (
            not isinstance(self.tile_size, int)
            or not isinstance(self.overlap, int)
            or not 256 <= self.tile_size <= 1024
            or not 0 <= self.overlap < self.tile_size / 2
        ):
            raise ValueError("Mosaicos de 256–1024 píxeles y solapamiento menor al 50%.")
        metric = self.metric_crs
        if not metric.is_projected or any(
            abs(axis.unit_conversion_factor - 1) > 1e-10 for axis in metric.axis_info[:2]
        ):
            raise ValueError("El CRS para áreas debe ser proyectado en metros.")
        if not 0 < self.min_area_m2 <= 500:
            raise ValueError("El área mínima debe estar entre 0 y 500 m².")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device debe ser auto, cpu o cuda.")

    @property
    def metric_crs(self) -> CRS:
        return CRS.from_user_input(self.analysis_crs)

    def as_manifest(self) -> dict:
        """Keep the public run-manifest schema stable for map/export consumers."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "mask_threshold": self.mask_threshold,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "min_area_m2": self.min_area_m2,
            "nms_iou": 0.35,
            "nms_containment": 0.7,
            "analysis_crs": self.analysis_crs,
        }
