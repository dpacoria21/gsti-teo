"""Exportaciones que conservan valores, referencias, fuentes y parámetros."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

from .proximity import PROXIMITY_LABELS, classify_distances, validate_thresholds, within_threshold


def _traffic_parameters(table: pd.DataFrame) -> dict | None:
    """Reject partial or stale traffic classifications before serializing them."""
    traffic_columns = {"proximity_class", "proximity_label", "red_band_m", "yellow_width_m"}
    if not traffic_columns.intersection(table.columns):
        return None
    required = traffic_columns | {"distance_m", "band_m", "within_band"}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError("Los resultados del semáforo no contienen todos sus parámetros y clases.")
    if any(table[column].nunique(dropna=False) != 1 for column in ("red_band_m", "band_m", "yellow_width_m")):
        raise ValueError("Los parámetros del semáforo deben ser uniformes en todos los resultados.")
    red, outer = validate_thresholds(table["red_band_m"].iloc[0], table["band_m"].iloc[0])
    width = table["yellow_width_m"].iloc[0]
    if not np.isfinite(width) or not np.isclose(width, outer - red, atol=1e-9, rtol=0):
        raise ValueError("El ancho amarillo no coincide con los límites del semáforo.")
    expected = classify_distances(table["distance_m"], red, outer)
    if not np.array_equal(table["proximity_class"].to_numpy(), expected):
        raise ValueError("Las clases del semáforo no coinciden con sus distancias y límites.")
    if not np.array_equal(table["proximity_label"].to_numpy(), [PROXIMITY_LABELS[value] for value in expected]):
        raise ValueError("Las etiquetas del semáforo no coinciden con sus clases.")
    if not np.array_equal(table["within_band"].to_numpy(), within_threshold(table["distance_m"], outer)):
        raise ValueError("La banda exterior no coincide con las clases del semáforo.")
    return {"red_band_m": red, "yellow_width_m": float(width), "band_m": outer}


def _safe_csv_value(value):
    """Keep genuine numeric negatives numeric while preventing spreadsheet formulas."""
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def csv_bytes(summary: pd.DataFrame) -> bytes:
    """UTF-8 BOM opens correctly in Excel; machine-readable field names remain stable."""
    _traffic_parameters(summary)
    table = pd.DataFrame(summary.drop(columns=["geometry"], errors="ignore")).copy()
    for column in table.columns:
        if pd.api.types.is_object_dtype(table[column]) or pd.api.types.is_string_dtype(table[column]):
            table[column] = table[column].map(_safe_csv_value)
    return table.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")


def geojson_bytes(summary: gpd.GeoDataFrame) -> bytes:
    _traffic_parameters(summary)
    if summary.crs is None:
        raise ValueError("La exportación GeoJSON necesita un CRS declarado.")

    def json_scalar(value):
        if isinstance(value, (date, datetime, pd.Timestamp)):
            return value.isoformat()
        raise TypeError(f"No se puede serializar {type(value).__name__} en GeoJSON.")

    return (
        summary.to_crs("EPSG:4326")
        .to_json(drop_id=True, ensure_ascii=False, na="null", default=json_scalar)
        .encode("utf-8")
    )


def analysis_bundle(
    summary: gpd.GeoDataFrame,
    relations: pd.DataFrame,
    metadata: dict,
    band_m: float,
) -> bytes:
    """Create a portable ZIP with both table grains and a reproducibility manifest."""
    if "band_m" not in summary or not summary["band_m"].eq(float(band_m)).all():
        raise ValueError("La banda del manifiesto no coincide con los resultados exportados.")
    if "band_m" not in relations or not relations["band_m"].eq(float(band_m)).all():
        raise ValueError("La banda del detalle no coincide con el manifiesto.")
    traffic_parameters = _traffic_parameters(summary)
    if traffic_parameters != _traffic_parameters(relations):
        raise ValueError("Los límites del semáforo del resumen y del detalle no coinciden.")
    now = datetime.now(timezone.utc)
    versions = {}
    for package in ("geopandas", "shapely", "pyproj", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    manifest = {
        "schema_version": 2 if traffic_parameters else 1,
        "generated_at_utc": now.isoformat(),
        "generated_at_lima": now.astimezone(timezone(timedelta(hours=-5))).isoformat(),
        "display_timezone": "America/Lima",
        "band_m": float(band_m),
        "analysis_crs": list(summary["analysis_crs"].unique()) if "analysis_crs" in summary else [],
        "export_crs": "EPSG:4326",
        "distance_definition": "Distancia mínima desde la huella completa hasta la geometría del cauce indicado; eje o polígono según distance_reference.",
        "within_band_rule": "distance_m <= band_m; tolerancia numérica de 0.000001 metros en el límite.",
        "building_count": int(summary["building_id"].nunique()),
        "relation_count": len(relations),
        "within_band_count": int(summary["within_band"].sum()),
        "limitations": [
            "La banda es un parámetro analítico, no una zona de peligro ni faja marginal oficial.",
            "Peligro y faja marginal no evaluados por ausencia de capas verificadas.",
            "Las distancias dependen de la cobertura y exactitud de los cauces y edificaciones de entrada.",
            (
                "Detecciones automáticas del modelo; no confirman uso residencial ni todas las edificaciones presentes."
                if metadata.get("analysis_mode") == "image_inference"
                else "Los contornos de edificaciones existentes no equivalen a inferencia realizada por esta aplicación."
            ),
        ],
        "source_metadata": metadata,
        "software_versions": versions,
    }
    if traffic_parameters:
        manifest.update(
            {
                "classification_mode": "proximity_traffic_light",
                "proximity_parameters": traffic_parameters,
                **traffic_parameters,
                "proximity_counts": {name: int(summary["proximity_class"].eq(name).sum()) for name in PROXIMITY_LABELS},
                "relation_proximity_counts": {
                    name: int(relations["proximity_class"].eq(name).sum()) for name in PROXIMITY_LABELS
                },
                "proximity_labels": PROXIMITY_LABELS,
                "proximity_rules": {
                    "red": "distance_m <= red_band_m",
                    "yellow": "red_band_m < distance_m <= band_m",
                    "green": "distance_m > band_m",
                    "boundary_tolerance_m": 0.000001,
                    "yellow_width_definition": "yellow_width_m = band_m - red_band_m",
                    "summary_rule": "La menor distancia entre todos los cauces determina la clase de la edificación.",
                },
                "hazard_status": "No evaluado",
                "margin_status": "No evaluado",
                "construction_status": "No evaluado",
            }
        )
        manifest["limitations"].extend(
            [
                "El semáforo representa proximidad geométrica; no es una evaluación de peligro o riesgo.",
                "Verde NO significa seguro ni permiso de construir. La posibilidad de construir no está evaluada.",
            ]
        )
    readme = (
        "RESULTADOS DE PROXIMIDAD A CAUCES\n\n"
        "resumen_edificaciones.csv: una fila por edificación, cauce cartografiado más cercano.\n"
        "relaciones_cauces.csv: una fila por par edificación y cauce, sin omitir relaciones.\n"
        "edificaciones.geojson: geometrías completas y atributos del resumen, WGS84.\n"
        "metadatos.json: parámetros, fuentes, momento del análisis y limitaciones.\n\n"
        "Las distancias están en metros. La banda no clasifica seguridad ni riesgo.\n"
        "El valor numérico exportado conserva precisión de cálculo; no expresa exactitud cartográfica.\n"
        "Los CSV usan UTF-8 con BOM. Textos que podrían ser fórmulas se preceden con apóstrofo.\n"
    )
    if traffic_parameters:
        readme += (
            "\nSEMÁFORO DE PROXIMIDAD\n"
            f"Rojo: distancia ≤ {traffic_parameters['red_band_m']:g} m.\n"
            f"Amarillo: distancia > {traffic_parameters['red_band_m']:g} y ≤ {band_m:g} m "
            f"(ancho {traffic_parameters['yellow_width_m']:g} m).\n"
            f"Verde: distancia > {band_m:g} m.\n"
            "Los límites incluyen una tolerancia numérica de 0.000001 m; no se redondea antes de clasificar.\n"
            "Se conserva la clase más próxima entre todos los cauces analizados.\n"
            "Verde NO significa seguro ni permiso de construir. Peligro, faja marginal y posibilidad de construir: "
            "No evaluado.\n"
        )
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("resumen_edificaciones.csv", csv_bytes(summary))
        archive.writestr("relaciones_cauces.csv", csv_bytes(relations))
        archive.writestr("edificaciones.geojson", geojson_bytes(summary))
        archive.writestr("metadatos.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        archive.writestr("LEEME.txt", readme)
    return output.getvalue()
