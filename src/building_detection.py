"""Compatibility facade for the modular building-inference package.

Existing scripts may keep these imports. Implementations live in src.inference;
none imports PyTorch or downloads model weights until inference is requested.
"""

from src.settings import PROJECT_ROOT as ROOT

from .inference.model import (
    MODEL_BYTES,
    MODEL_ID,
    MODEL_PATH,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_URL,
    ensure_model,
    load_model,
    sha256_file,
)
from .inference.pipeline import detect_buildings
from .inference.postprocess import SOURCE_ID, mask_to_geometry, polygonal, suppress_duplicates
from .inference.raster import inspect_raster, save_preview, tile_starts

__all__ = [
    "ROOT",
    "MODEL_BYTES",
    "MODEL_ID",
    "MODEL_PATH",
    "MODEL_REVISION",
    "MODEL_SHA256",
    "MODEL_URL",
    "SOURCE_ID",
    "detect_buildings",
    "ensure_model",
    "inspect_raster",
    "load_model",
    "mask_to_geometry",
    "polygonal",
    "save_preview",
    "sha256_file",
    "suppress_duplicates",
    "tile_starts",
]
