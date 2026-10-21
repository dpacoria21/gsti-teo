"""Prepare an explicitly separate Arequipa demonstration from CENEPRED imagery.

Run with .venv-ml/Scripts/python.exe. The image is aerial (2018), not Chullo.
GISInternals GDAL is used only here to decode the official MrSID download.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import geopandas as gpd
import requests
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import LineString, box, mapping
from shapely.ops import linemerge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import prepare_buildings

IMAGE = ROOT / "data/imagery/mariano_melgar"
OUT = ROOT / "data/processed/mariano_melgar"
ZIP_URL = "https://sigrid.cenepred.gob.pe/sigridv3/documento/6798/descargar"
GDAL_URL = "https://download.gisinternals.com/sdk/downloads/release-1944-x64-gdal-3-12-1-mapserver-8-6-0.zip"
WAYS = [320098555, 1014282552, 1014282553, 1014282554, 1014282555, 1123063114]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=(15, 60))
    response.raise_for_status()
    path.write_bytes(response.content)


def extract(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("ZIP path outside destination")
        archive.extractall(destination)


def main():
    IMAGE.mkdir(parents=True, exist_ok=True)
    download(ZIP_URL, IMAGE / "source.zip")
    sid = IMAGE / "source/ORTOMOSAICO/SID/mariano_melgar.sid"
    if not sid.exists():
        extract(IMAGE / "source.zip", IMAGE / "source")
    runtime = ROOT / ".runtime/gdal_imagery"
    gdal = runtime / "tools"
    if not (gdal / "bin/gdal/apps/gdal_translate.exe").exists():
        download(GDAL_URL, runtime / "gdal.zip")
        extract(runtime / "gdal.zip", gdal)
    env = os.environ.copy()
    env.update({"PATH": str(gdal / "bin") + os.pathsep + env.get("PATH", ""),
                "GDAL_DRIVER_PATH": str(gdal / "bin/gdal/plugins"),
                "GDAL_DATA": str(gdal / "bin/gdal-data"),
                "PROJ_DATA": str(gdal / "bin/proj9/share")})
    raster = IMAGE / "mariano_melgar_rgb.tif"
    if not raster.exists():
        subprocess.run([str(gdal / "bin/gdal/apps/gdal_translate.exe"),
                        "-b", "1", "-b", "2", "-b", "3", "-mask", "4",
                        "-projwin", "233750", "8184800", "234150", "8184400",
                        "-tr", "0.5", "0.5", "-r", "average",
                        "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", str(sid), str(raster)],
                       env=env, check=True)
    with rasterio.open(raster) as src:
        if src.count != 3 or src.crs.to_epsg() != 32719 or src.dtypes[0] != "uint8":
            raise ValueError("Unexpected prepared raster format")
        bounds = list(src.bounds)
        bbox = list(transform_bounds(src.crs, "EPSG:4326", *bounds))
        valid_fraction = float((src.dataset_mask() > 0).mean())
        if valid_fraction != 1:
            raise ValueError("The selected demonstration crop contains nodata")
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1, "image_id": "cenepred_mariano_melgar_20180605_crop",
        "source_id": "cenepred_mariano_imagery", "study_area_id": "mariano_melgar",
        "title": "Ortomosaico real de Mariano Melgar · Arequipa · 2018",
        "location_name": "Mariano Melgar, Arequipa (caso alternativo; no Chullo)",
        "image_kind": "Ortomosaico aéreo de dron; no imagen satelital",
        "raster_path": "data/imagery/mariano_melgar/mariano_melgar_rgb.tif",
        "url": "https://sigrid.cenepred.gob.pe/sigridv3/documento/6798",
        "download_url": ZIP_URL, "source": "CENEPRED · SIGRID", "organization": "CENEPRED",
        "observed_at": "2018-06-05", "capture_date": "2018-06-05", "published_at": "2019",
        "downloaded_at": now,
        "license": "Acceso público SIGRID para consultar, compartir, analizar y monitorear información, con crédito a los autores. Sin licencia Creative Commons específica declarada.",
        "license_url": "https://sigrid.cenepred.gob.pe/sigridv3/documento/6798",
        "attribution": "Imagen aérea: CENEPRED / SIGRID, Ortomosaico Sector de Mariano Melgar, vuelo 05/06/2018. Recorte y remuestreo propios.",
        "platform": "DJI Inspire 2 copter", "sensor": "RGB", "camera": "Zenmuse X5S",
        "ground_control_reported": True, "native_resolution_m": 0.1093,
        "resolution_m": [0.5, 0.5], "crs": "EPSG:32719", "bbox": bounds, "bbox_wgs84": bbox,
        "width": 800, "height": 800, "bands": 3, "dtype": "uint8", "valid_pixel_fraction": valid_fraction,
        "sha256": digest(raster), "source_zip_sha256": digest(IMAGE / "source.zip"),
        "processing": "Recorte UTM [233750,8184400,234150,8184800]; promedio a 0.5m; RGB y máscara nativa. Sin superresolución.",
        "processing_tool": "GDAL 3.12.1 GISInternals con MrSID (solo preparación local)",
        "georeferencing": "Georreferencia nativa MrSID oficial conservada; no ajustada manualmente.",
        "accuracy_notes": "El catálogo indica control GPS diferencial, sin precisión numérica. Imagen de 2018; no representa un inventario actual ni validación de campo.",
    }
    dump(IMAGE / "manifest.json", metadata)
    area_utm = gpd.GeoDataFrame([{"area_id": "mariano_melgar", "name": "Mariano Melgar · imagen aérea 2018"}], geometry=[box(*bounds)], crs=32719)
    area = area_utm.to_crs(4326)
    OUT.mkdir(parents=True, exist_ok=True)
    area.to_file(OUT / "study_area.geojson", driver="GeoJSON")
    buildings, building_counts = prepare_buildings(area.geometry.iloc[0])
    dump(OUT / "buildings.geojson", {"type": "FeatureCollection", "features": buildings})
    lines, osm_provenance = [], []
    for way_id in WAYS:
        path = IMAGE / "source" / f"osm_way_{way_id}_full.xml"
        download(f"https://api.openstreetmap.org/api/0.6/way/{way_id}/full", path)
        xml = ET.parse(path).getroot()
        nodes = {n.attrib["id"]: (float(n.attrib["lon"]), float(n.attrib["lat"])) for n in xml.findall("node")}
        way = next(w for w in xml.findall("way") if w.attrib["id"] == str(way_id))
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        lines.append(LineString([nodes[n.attrib["ref"]] for n in way.findall("nd")]))
        osm_provenance.append({"way_id": way_id, "version": way.attrib["version"], "edited_at": way.attrib["timestamp"], "tags": tags, "sha256": digest(path)})
    tributary = linemerge(lines[1:])
    if tributary.geom_type != "LineString":
        raise ValueError("Expected a connected tributary")
    rivers = gpd.GeoDataFrame([
        {"river_id": "mariano_main_osm", "name": "Torrentera local · eje OSM", "source_id": "osm_mariano", "geometry_role": "axis", "review_status": "visual_ortho_reference_only", "observed_at": None, "osm_way_ids": "320098555"},
        {"river_id": "mariano_tributary_osm", "name": "Ramal local · eje OSM", "source_id": "osm_mariano", "geometry_role": "axis", "review_status": "visual_ortho_reference_only", "observed_at": None, "osm_way_ids": ",".join(map(str, WAYS[1:]))},
    ], geometry=[lines[0], tributary], crs=4326)
    if not rivers.is_valid.all():
        raise ValueError("Invalid river geometry")
    rivers.to_file(OUT / "rivers.geojson", driver="GeoJSON")
    configuration_path = ROOT / "config/study_areas.json"
    config = json.loads(configuration_path.read_text(encoding="utf8"))
    config["study_areas"] = [a for a in config["study_areas"] if a["id"] != "mariano_melgar"] + [{
        "id": "mariano_melgar", "name": "Mariano Melgar · demostración IA (imagen 2018)", "path": "data/processed/mariano_melgar", "analysis_crs": "EPSG:32719", "bbox": bbox, "max_band_m": 100,
        "description": "Caso adicional de Arequipa con ortomosaico real CENEPRED de 2018; distinto de Chullo. Inventario Microsoft y detección IA se seleccionan por separado.",
        "coverage_notes": "Recorte de 400 × 400m, ejes OSM locales sin nombre oficial verificado. No delimitación legal ni evaluación de peligro.",
        "imagery_path": metadata["raster_path"], "imagery_manifest": "data/imagery/mariano_melgar/manifest.json"}]
    dump(configuration_path, config)
    source_path = ROOT / "data/sources.json"
    sources = json.loads(source_path.read_text(encoding="utf8"))
    sources["sources"] = [s for s in sources["sources"] if s["source_id"] not in {"cenepred_mariano_imagery", "osm_mariano"}] + [metadata, {
        "source_id": "osm_mariano", "title": "OpenStreetMap · ejes de torrenteras del sector Mariano Melgar", "organization": "OpenStreetMap contributors",
        "url": "https://www.openstreetmap.org/way/320098555", "license": "ODbL-1.0", "license_url": "https://www.openstreetmap.org/copyright",
        "attribution": "© OpenStreetMap contributors; datos bajo ODbL 1.0.", "observed_at": None, "downloaded_at": now, "original_crs": "EPSG:4326",
        "review_status": "Correspondencia visual con cauces del ortomosaico CENEPRED; no exactitud topográfica ni nombre oficial verificados.",
        "notes": "No tienen name OSM. Se conservan vías completas, sin sustituir el eje por borde ni banda legal.", "ways": osm_provenance}]
    dump(source_path, sources)
    dump(OUT / "build_manifest.json", {"generated_at": now, "buildings": building_counts, "image_sha256": digest(raster), "area_m2": 160000, "river_lengths_m": list(rivers.to_crs(32719).length)})
    print(json.dumps({"raster": str(raster), "buildings": len(buildings), "bbox": bbox}, ensure_ascii=False))


if __name__ == "__main__":
    main()
