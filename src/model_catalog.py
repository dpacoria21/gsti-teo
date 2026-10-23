"""Explicit model choices and trusted local checkpoint metadata; no ML imports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.settings import PROJECT_ROOT

ORIGINAL_VARIANT = "usa"
TRAINED_VARIANT = "spacenet2_khartoum_v1"
TRAINED_MANIFEST = PROJECT_ROOT / "models" / TRAINED_VARIANT / "manifest.json"


def trained_metadata() -> dict:
    metadata = json.loads(TRAINED_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("El manifiesto del modelo debe ser un objeto JSON.")
    if metadata.get("variant") != TRAINED_VARIANT or not metadata.get("evaluation_report"):
        raise ValueError("El modelo ajustado necesita un manifiesto y evaluación registrados.")
    path = (TRAINED_MANIFEST.parent / metadata["filename"]).resolve()
    if not path.is_relative_to(TRAINED_MANIFEST.parent.resolve()):
        raise ValueError("La ruta de los pesos debe estar dentro de su carpeta registrada.")
    return {**metadata, "path": str(path)}


def model_choices() -> dict[str, str]:
    choices = {ORIGINAL_VARIANT: "Original · GeoAI USA"}
    if TRAINED_MANIFEST.is_file():
        try:
            metadata = trained_metadata()
            if Path(metadata["path"]).is_file():
                choices[TRAINED_VARIANT] = "Ajustado · SpaceNet (experimental)"
        except (OSError, ValueError, KeyError, TypeError):
            logging.getLogger(__name__).warning("Modelo opcional no disponible: manifiesto inválido.")
    return choices
