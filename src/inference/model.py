"""Pinned checkpoint acquisition and lazy PyTorch model construction."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import requests

from src.settings import PROJECT_ROOT

MODEL_ID = "giswqs/geoai/building_footprints_usa"
MODEL_REVISION = "43b6aa04156d7d9d2b90fe321096f15f03353cd2"
MODEL_SHA256 = "3aea5d0da7803be31ff4e2e1e5ac747f15cab2a13d6bb1a9319b6a6ba8bc7bca"
MODEL_BYTES = 176225905
MODEL_URL = f"https://huggingface.co/giswqs/geoai/resolve/{MODEL_REVISION}/building_footprints_usa.pth"
MODEL_PATH = PROJECT_ROOT / "models" / "building_footprints_usa.pth"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(path: str | Path = MODEL_PATH, progress: Callable | None = None) -> Path:
    """Download only the pinned public checkpoint and verify before loading."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        if progress:
            progress(stage="model_download", message="Descargando pesos preentrenados (176 MB).", progress=0.02)
        with requests.get(MODEL_URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if temporary.stat().st_size != MODEL_BYTES or sha256_file(temporary) != MODEL_SHA256:
            temporary.unlink(missing_ok=True)
            raise ValueError("El archivo de pesos descargado no coincide con el SHA256 fijado.")
        temporary.replace(path)
    if path.stat().st_size != MODEL_BYTES or sha256_file(path) != MODEL_SHA256:
        raise ValueError("Los pesos locales no coinciden con el modelo autorizado y su SHA256.")
    return path


def load_model(path: Path, device: str = "auto"):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    if selected not in {"cpu", "cuda"}:
        raise ValueError("device debe ser auto, cpu o cuda.")
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible. Use --device cpu.")
    # Identical architecture, normalization and default resize to GeoAI extractor.
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        progress=False,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    try:
        model.to(selected)
    except RuntimeError:
        if device != "auto" or selected != "cuda":
            raise
        selected = "cpu"
        model.to(selected)
    return model, selected


def resolve_variant(variant: str = "usa", progress: Callable | None = None) -> tuple[Path, dict]:
    """Only serve registered checkpoints whose bytes match their provenance."""
    from src.model_catalog import ORIGINAL_VARIANT, TRAINED_VARIANT, trained_metadata

    if variant == ORIGINAL_VARIANT:
        return ensure_model(progress=progress), {
            "id": MODEL_ID,
            "variant": ORIGINAL_VARIANT,
            "revision": MODEL_REVISION,
            "sha256": MODEL_SHA256,
            "url": MODEL_URL,
            "license": "MIT (model repository)",
            "trained_locally": False,
            "geography": "Publicado como modelo USA; no validado cuantitativamente en Arequipa.",
        }
    if variant != TRAINED_VARIANT:
        raise ValueError("Modelo desconocido; seleccione un modelo del catálogo registrado.")
    metadata = trained_metadata()
    path = Path(metadata["path"])
    if path.stat().st_size != metadata["size_bytes"] or sha256_file(path) != metadata["sha256"]:
        raise ValueError("Los pesos ajustados no coinciden con el tamaño y SHA256 registrados.")
    return path, metadata
