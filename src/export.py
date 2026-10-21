"""Exportaciones que conservan valores, referencias, fuentes y parámetros."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd
import pandas as pd


def _safe_csv_value(value):
    """Keep genuine numeric negatives numeric while preventing spreadsheet formulas."""
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def csv_bytes(summary: pd.DataFrame) -> bytes:
    """UTF-8 BOM opens correctly in Excel; machine-readable field names remain stable."""
    table = pd.DataFrame(summary.drop(columns=["geometry"], errors="ignore")).copy()
    for column in table.columns:
        if pd.api.types.is_object_dtype(table[column]) or pd.api.types.is_string_dtype(table[column]):
            table[column] = table[column].map(_safe_csv_value)
    return table.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")


def geojson_bytes(summary: gpd.GeoDataFrame) -> bytes:
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
    now = datetime.now(timezone.utc)
    versions = {}
    for package in ("geopandas", "shapely", "pyproj", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    manifest = {
        "schema_version": 1,
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
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("resumen_edificaciones.csv", csv_bytes(summary))
        archive.writestr("relaciones_cauces.csv", csv_bytes(relations))
        archive.writestr("edificaciones.geojson", geojson_bytes(summary))
        archive.writestr("metadatos.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        archive.writestr("LEEME.txt", readme)
    return output.getvalue()
