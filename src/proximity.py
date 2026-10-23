"""Shared numerical rules for the configurable proximity traffic light."""

from __future__ import annotations

from numbers import Real

import numpy as np

# One micrometre absorbs floating-point noise at exact metric boundaries.
BOUNDARY_TOLERANCE_M = 1e-6
PROXIMITY_LABELS = {
    "red": "Rojo · proximidad inmediata",
    "yellow": "Amarillo · proximidad intermedia",
    "green": "Verde · mayor distancia",
}
PROXIMITY_COLORS = {"red": "#dc2626", "yellow": "#eab308", "green": "#16a34a"}


def validate_thresholds(red_band_m: float, band_m: float) -> tuple[float, float]:
    """Return validated inner red and outer yellow distances in metres."""
    for value in (red_band_m, band_m):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("Los límites del semáforo deben ser números finitos en metros.")
    if not 0 <= red_band_m < band_m <= 1000:
        raise ValueError("El semáforo requiere 0 ≤ límite rojo < límite amarillo ≤ 1000 metros.")
    return float(red_band_m), float(band_m)


def within_threshold(distances, threshold_m: float) -> np.ndarray:
    """Include the exact boundary without rounding the underlying distances."""
    values = np.asarray(distances, dtype=float)
    return (values <= threshold_m) | np.isclose(values, threshold_m, atol=BOUNDARY_TOLERANCE_M, rtol=0)


def classify_distances(distances, red_band_m: float, band_m: float) -> np.ndarray:
    """Classify finite distances; missing/invalid data must never become green."""
    red, outer = validate_thresholds(red_band_m, band_m)
    values = np.asarray(distances, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("El semáforo requiere distancias finitas y no negativas.")
    return np.where(within_threshold(values, red), "red", np.where(within_threshold(values, outer), "yellow", "green"))
