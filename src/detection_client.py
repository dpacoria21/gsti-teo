"""Compatibility API for callers predating the service/repository separation.

New code can import the application runner or detection repository directly.
This module never imports the model runtime into the map environment.
"""

from .application.detection_runner import (
    ML_PYTHON,
    ROOT,
    inspect_raster,
    ml_available,
    ml_status,
    run_detection,
)
from .repositories.detections import load_detection

__all__ = [
    "ML_PYTHON",
    "ROOT",
    "inspect_raster",
    "load_detection",
    "ml_available",
    "ml_status",
    "run_detection",
]
