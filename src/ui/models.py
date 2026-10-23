"""Explicit view state passed between Streamlit components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.application.provenance import DetectionDescription


@dataclass(frozen=True)
class SidebarSelection:
    area_id: str
    area_name: str
    area_lookup: dict[str, dict]
    building_mode: str
    included_raster: Path | None
    raster_path: Path | None
    image_manifest_path: Path | None
    image_record: dict
    detection_dir: Path
    threshold: float
    detect_clicked: bool
    chosen_band: float
    analyze: bool
    basemap: bool
    show_imagery: bool
    session_id: str
    input_id: str
    chosen_red_band: float = 25

    @property
    def use_detection(self) -> bool:
        return self.building_mode == "Detección IA desde imagen"

    @property
    def area_config(self) -> dict:
        return self.area_lookup[self.area_id]


@dataclass(frozen=True)
class AnalysisView:
    payload: dict[str, Any]
    analysis: dict[str, Any]
    metadata: dict[str, Any]
    detection: dict[str, Any] | None
    image_overlay: dict[str, Any] | None
    band_m: float
    detection_version: tuple
    detection_failed: bool
    description: DetectionDescription | None
    red_band_m: float = 25

    @property
    def yellow_width_m(self) -> float:
        return self.band_m - self.red_band_m

    def color_count(self, color: str) -> int:
        return int(self.summary["proximity_class"].eq(color).sum())

    @property
    def summary(self):
        return self.analysis["summary"]

    @property
    def evaluated_count(self) -> int:
        return int(self.summary["distance_m"].notna().sum())

    @property
    def inside_count(self) -> int:
        return int(self.summary["within_band"].fillna(False).sum())
