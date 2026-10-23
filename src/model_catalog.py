"""Explicit model choices and trusted local checkpoint metadata; no ML imports."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.settings import PROJECT_ROOT

ORIGINAL_VARIANT = "usa"
TRAINED_VARIANT = "spacenet2_khartoum_v1"
LOCAL_VARIANT = "chullo_local_v1"
TRAINED_VARIANTS = (TRAINED_VARIANT, LOCAL_VARIANT)
REGISTERED_VARIANTS = (ORIGINAL_VARIANT, *TRAINED_VARIANTS)
TRAINED_MANIFEST = PROJECT_ROOT / "models" / TRAINED_VARIANT / "manifest.json"
LOCAL_MANIFEST = PROJECT_ROOT / "models" / LOCAL_VARIANT / "manifest.json"

_PRESENTATION = {
    ORIGINAL_VARIANT: {
        "title": "Original · GeoAI USA",
        "description": "Se aplican los pesos preentrenados publicados por GeoAI; no se ha ajustado este modelo con Chullo.",
        "limitations": ["Puede omitir cubiertas o producir falsos positivos; requiere revisión visual local."],
    },
    TRAINED_VARIANT: {
        "title": "Ajustado · SpaceNet (experimental)",
        "description": "Modelo ajustado con imágenes y etiquetas de SpaceNet 2 (Jartum), sin etiquetas de entrenamiento de Chullo.",
        "limitations": [
            "La evaluación externa en Jartum no mide la exactitud en Chullo.",
            "En la revisión visual de Chullo se observaron varias cubiertas y patios unidos en un solo contorno; no permite contar viviendas individuales.",
        ],
    },
    LOCAL_VARIANT: {
        "title": "Ajustado · Chullo (piloto)",
        "description": (
            "Ajuste piloto con 3 recortes originales de entrenamiento (4 anotados; 1 excluido por zonas ambiguas), "
            "2 de validación y 2 de prueba "
            "de una sola captura de Chullo, Google Earth Pro / Airbus, del 2026-03-04."
        ),
        "limitations": [
            "Referencias assistant_visual_manual; human_reviewed=false. Son anotaciones visuales del asistente sin revisión humana.",
            "Los recortes pertenecen a una sola captura: la evaluación piloto no demuestra generalización a otras fechas o localidades.",
            "Se detectan cubiertas visibles; no se confirma uso residencial, número de viviendas, peligro ni permiso para construir.",
        ],
    },
}


def _trained_manifest(variant: str) -> Path:
    if variant == TRAINED_VARIANT:
        return TRAINED_MANIFEST
    if variant == LOCAL_VARIANT:
        return LOCAL_MANIFEST
    raise ValueError("Modelo desconocido; seleccione un modelo del catálogo registrado.")


def trained_metadata(variant: str = TRAINED_VARIANT) -> dict:
    """Read one registered manifest; the no-argument API still selects SpaceNet."""
    manifest = _trained_manifest(variant)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("El manifiesto del modelo debe ser un objeto JSON.")
    if metadata.get("variant") != variant or not metadata.get("evaluation_report"):
        raise ValueError("El modelo ajustado necesita un manifiesto y evaluación registrados.")
    checksum = metadata.get("sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("El manifiesto del modelo requiere un SHA256 válido de los pesos.")
    size = metadata.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("El manifiesto del modelo requiere un tamaño positivo de los pesos.")
    filename = metadata.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("El manifiesto del modelo requiere el nombre del archivo de pesos.")
    path = (manifest.parent / filename).resolve()
    if not path.is_relative_to(manifest.parent.resolve()):
        raise ValueError("La ruta de los pesos debe estar dentro de su carpeta registrada.")
    return {**metadata, "path": str(path)}


def model_presentation(metadata: dict | None = None, *, variant: str = ORIGINAL_VARIANT) -> dict:
    """Use saved model text with variant-specific fallbacks for older manifests."""
    if metadata is None:
        if variant in TRAINED_VARIANTS:
            try:
                metadata = trained_metadata(variant)
            except (OSError, ValueError, KeyError, TypeError):
                metadata = {}
        else:
            metadata = {}
    variant = metadata.get("variant", variant)
    defaults = _PRESENTATION.get(
        variant,
        {
            "title": "Modelo de edificaciones",
            "description": "Consultar la procedencia del modelo en los metadatos de la ejecución.",
            "limitations": ["No se dispone de un alcance de evaluación registrado para esta variante."],
        },
    )
    limitations = metadata.get("limitations") or defaults["limitations"]
    if isinstance(limitations, str):
        limitations = [limitations]
    notice = metadata.get("user_notice")
    if not notice:
        notice = (
            "Modelo piloto con etiquetas visuales del asistente, sin revisión humana. Recupera más cubiertas, "
            "pero genera falsos positivos y aún puede unir edificios. Revisa los contornos antes de interpretar sus distancias."
            if variant == LOCAL_VARIANT else " ".join(str(item) for item in limitations)
        )
    return {
        "title": metadata.get("title") or defaults["title"],
        "description": metadata.get("description") or defaults["description"],
        "limitations": [str(item) for item in limitations],
        "notice": notice,
    }


def model_choices() -> dict[str, str]:
    choices = {ORIGINAL_VARIANT: _PRESENTATION[ORIGINAL_VARIANT]["title"]}
    for variant in TRAINED_VARIANTS:
        if not _trained_manifest(variant).is_file():
            continue
        try:
            metadata = trained_metadata(variant)
            if Path(metadata["path"]).is_file():
                choices[variant] = model_presentation(metadata)["title"]
        except (OSError, ValueError, KeyError, TypeError):
            logging.getLogger(__name__).warning("Modelo opcional %s no disponible: manifiesto inválido.", variant)
    return choices
