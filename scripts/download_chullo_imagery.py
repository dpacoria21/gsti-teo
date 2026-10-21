"""Acquire a small real IGN orthoimage when its public service is available.

Run on demand with .venv-ml/Scripts/python.exe. No recurring requests are made.
Failure never creates a synthetic image or substitutes another location.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import requests

ROOT = Path(__file__).resolve().parents[1]
SERVICE = "https://www.idep.gob.pe/geoportal/rest/services/IMAGENES/ORTOFOTO_AREQUIPA/MapServer"
OUT = ROOT / "data/imagery/chullo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    target = OUT / "chullo_rgb.tif"
    if target.exists() and not args.overwrite:
        print(f"Ya existe {target}. Use --overwrite para reemplazarlo explícitamente.")
        return
    configuration = json.loads((ROOT / "config/study_areas.json").read_text(encoding="utf8"))
    area = next(a for a in configuration["study_areas"] if a["id"] == "chullo")
    projected = transform_bounds("EPSG:4326", "EPSG:32719", *area["bbox"])
    xmin, ymin, xmax, ymax = projected
    resolution = 0.5
    width = math.ceil((xmax - xmin) / resolution)
    height = math.ceil((ymax - ymin) / resolution)
    xmax, ymax = xmin + width * resolution, ymin + height * resolution
    session = requests.Session()
    session.headers["User-Agent"] = "ChulloAcademicPrototype/0.2 (small public IGN orthoimage export)"
    metadata_response = session.get(SERVICE, params={"f": "json"}, timeout=(8, 15))
    metadata_response.raise_for_status()
    service_metadata = metadata_response.json()
    if "error" in service_metadata:
        raise ValueError(f"IGN metadata: {service_metadata['error']}")
    parameters = {"f": "json", "bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": 32719,
                  "imageSR": 32719, "size": f"{width},{height}", "format": "png24",
                  "transparent": "false", "layers": "show:0"}
    response = session.get(SERVICE + "/export", params=parameters, timeout=(8, 30))
    response.raise_for_status()
    export = response.json()
    if "error" in export or not export.get("href"):
        raise ValueError(f"El servicio no devolvió una imagen: {export}")
    if urlparse(export["href"]).hostname not in {"www.idep.gob.pe", "idep.gob.pe", "portalgeo.idep.gob.pe"}:
        raise ValueError("El enlace de imagen sale del dominio IGN esperado; revisar manualmente.")
    pixels_response = session.get(export["href"], timeout=(8, 30))
    pixels_response.raise_for_status()
    image = Image.open(io.BytesIO(pixels_response.content)).convert("RGB")
    pixels = np.asarray(image)
    if image.size != (export["width"], export["height"]):
        raise ValueError("El tamaño real no coincide con la respuesta IGN.")
    if float(pixels.std()) < 5 or float((pixels.max(axis=2) > 5).mean()) < 0.8:
        raise ValueError("La imagen es vacía o tiene cobertura insuficiente; revisar el servicio.")
    extent = export["extent"]
    crs = extent["spatialReference"].get("latestWkid", extent["spatialReference"].get("wkid"))
    if crs != 32719:
        raise ValueError("IGN no devolvió el CRS solicitado EPSG:32719.")
    bounds = [extent[k] for k in ("xmin", "ymin", "xmax", "ymax")]
    transform = from_bounds(*bounds, image.width, image.height)
    actual_bbox = transform_bounds(crs, 4326, *bounds)
    if abs(actual_bbox[0] - area["bbox"][0]) > 0.001 or abs(actual_bbox[1] - area["bbox"][1]) > 0.001:
        raise ValueError("La extensión devuelta no corresponde al sector de Chullo.")
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / "chullo_rgb.partial.tif"
    with rasterio.open(temporary, "w", driver="GTiff", width=image.width, height=image.height,
                       count=3, dtype="uint8", crs=crs, transform=transform, compress="deflate", tiled=True) as dst:
        dst.write(np.moveaxis(pixels, -1, 0))
        dst.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green, rasterio.enums.ColorInterp.blue)
    with rasterio.open(temporary) as check:
        if check.crs.to_epsg() != 32719 or check.count != 3:
            raise ValueError("Falló la validación del GeoTIFF guardado.")
    temporary.replace(target)
    image.save(OUT / "preview.png")
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1, "image_id": "ign_chullo_ortho_export", "source_id": "ign_chullo_imagery",
        "study_area_id": "chullo", "title": "Ortofoto IGN · sector Chullo",
        "source": "Instituto Geográfico Nacional del Perú", "organization": "IGN",
        "url": SERVICE, "raster_path": "data/imagery/chullo/chullo_rgb.tif",
        "image_kind": "Exportación RGB de ortofoto IGN; plataforma de captura no confirmada",
        "observed_at": None, "capture_date": None, "downloaded_at": now,
        "native_resolution_m": None, "resolution_m": [transform.a, abs(transform.e)],
        "resolution_notes": "0.5m es el muestreo solicitado al servidor. Resolución nativa y fecha no publicadas; no inferirlas de la escala 1:1000.",
        "license": "Servicio público institucional IGN; sin licencia abierta específica declarada en la metadata. Uso académico local con atribución, sin afirmar permiso comercial o de redistribución general.",
        "license_url": "https://www.gob.pe/9039-acceder-al-geoportal-institucional-del-instituto-geografico-nacional",
        "attribution": "Ortofoto de Arequipa © Instituto Geográfico Nacional del Perú (IGN). Recorte del sector Chullo.",
        "crs": "EPSG:32719", "bbox": bounds, "bbox_wgs84": list(actual_bbox),
        "width": image.width, "height": image.height, "bands": 3, "dtype": "uint8",
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "download_sha256": hashlib.sha256(pixels_response.content).hexdigest(),
        "processing": "Exportación de servicio público IGN; GeoTIFF construido con extensión exacta de la respuesta. Sin inferencia de coordenadas ni superresolución.",
        "visual_review_status": "Pendiente: inspeccionar preview.png antes de presentar resultados de IA.",
        "export_request": parameters, "export_response": export,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    (OUT / "ign_service_metadata.json").write_text(json.dumps(service_metadata, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"Imagen real descargada: {target}. Revisar preview.png; fecha y resolución nativa desconocidas.")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, ValueError, KeyError, OSError) as exc:
        print(f"No se pudo descargar la imagen IGN de Chullo: {exc}. No se ha sustituido por datos de otra localidad.", file=sys.stderr)
        sys.exit(1)
