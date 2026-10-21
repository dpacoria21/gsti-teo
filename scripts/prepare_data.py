"""Prepare the real Chullo pilot from cached, attributed public vector data.

Run from any directory: .venv/Scripts/python.exe scripts/prepare_data.py
Only this preparation step uses the network. The app reads the resulting files.
The original geometries are never clipped, silently repaired, or replaced by test data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import LineString, Point, box, mapping, shape
from shapely.ops import linemerge


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/chullo"
BBOX = [-71.554, -16.403, -71.550, -16.397]
MAX_BAND_M = 100
TILE = "microsoft_peru_210031232_2026-08-13.geojsonl.gz"
TILE_URL = (
    "https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/"
    "RegionName=Peru/quadkey=210031232/"
    "part-00104-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz"
)
TILE_SHA256 = "50d3382c6225820a0da2a7553239f60e00df3b011108e252f4e1070bfcf814ad"
WAY_IDS = [570369893, 25980163, 535091453, 535091454, 781522545]
URLS = {
    TILE: TILE_URL,
    **{f"osm_way_{i}_full.xml": f"https://api.openstreetmap.org/api/0.6/way/{i}/full" for i in WAY_IDS},
    "ingemmet_A7406.pdf": "https://sigrid.cenepred.gob.pe/sigridv3/documento/16497/descargar",
    "ingemmet_OT006_2020.pdf": "https://sigrid.cenepred.gob.pe/sigridv3/documento/10009/descargar",
    "osm_north_node_1862680670_ways.xml": "https://api.openstreetmap.org/api/0.6/node/1862680670/ways",
}
# These are published infrastructure / inspection reference points, NOT channel vertices.
REFERENCE_POINTS = [
    ("A7406_cruce_1", 227534.8, 8185486.6, "ingemmet_a7406", "Cuadro 9, p.21; figura 18, p.23"),
    ("A7406_cruce_2", 227552.8, 8185248.9, "ingemmet_a7406", "Cuadro 9, p.21"),
    ("A7406_cruce_3", 227498.5, 8185057.9, "ingemmet_a7406", "Cuadro 9, p.21"),
    ("OT006_extremo_sur", 227384.0, 8184790.0, "ingemmet_ot006", "Cuadro 1, p.3"),
    ("OT006_extremo_norte", 227488.0, 8186090.0, "ingemmet_ot006", "Cuadro 1, p.3"),
]


def iso_timestamp(timestamp: float | None = None) -> str:
    when = datetime.fromtimestamp(timestamp, timezone.utc) if timestamp is not None else datetime.now(timezone.utc)
    return when.isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def acquire(*, offline: bool = False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "download_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {}
    for filename, url in URLS.items():
        target = RAW / filename
        if not target.exists():
            if offline:
                raise FileNotFoundError(f"Missing cached input {target}. Run once without --offline.")
            request = urllib.request.Request(url, headers={"User-Agent": "ChulloAcademicPrototype/0.1 (data preparation)"})
            temporary = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            temporary.replace(target)
        digest = sha256(target)
        if filename in previous and previous[filename]["sha256"] != digest:
            raise ValueError(f"Cached source changed: {filename}. Review provenance before replacing the manifest.")
        if filename == TILE and digest != TILE_SHA256:
            raise ValueError("Microsoft tile checksum differs from the verified release.")
        manifest[filename] = {
            "url": url,
            "sha256": digest,
            "bytes": target.stat().st_size,
            "downloaded_at": previous.get(filename, {}).get("downloaded_at", iso_timestamp(target.stat().st_mtime)),
        }
    write_json(manifest_path, manifest)
    return manifest


def prepare_buildings(study_area) -> tuple[list, dict]:
    features, seen = [], set()
    counts = {"tile_features_scanned": 0, "intersecting": 0, "duplicates_removed": 0, "crosses_study_boundary": 0}
    with gzip.open(RAW / TILE, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            counts["tile_features_scanned"] += 1
            geom = shape(record["geometry"])
            if not geom.intersects(study_area):
                continue
            counts["intersecting"] += 1
            if geom.is_empty or not geom.is_valid or geom.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError("An intersecting Microsoft footprint is invalid; review explicitly before proceeding.")
            canonical = geom.normalize().wkb
            identifier = "ms_210031232_" + hashlib.sha256(canonical).hexdigest()[:16]
            if identifier in seen:
                counts["duplicates_removed"] += 1
                continue
            seen.add(identifier)
            crosses = not study_area.covers(geom)
            counts["crosses_study_boundary"] += int(crosses)
            props = record.get("properties", {})
            features.append({
                "type": "Feature",
                "properties": {
                    "building_id": identifier,
                    "source_id": "microsoft",
                    "observed_at": None,
                    "review_status": "geometry_checked_not_field_verified",
                    "crosses_study_boundary": crosses,
                    "height_m": props.get("height") if props.get("height", -1) >= 0 else None,
                    "confidence": props.get("confidence") if props.get("confidence", -1) >= 0 else None,
                    "source_tile": "210031232",
                },
                "geometry": record["geometry"],
            })
    if not features:
        raise ValueError("No buildings found in the pilot area.")
    features.sort(key=lambda feature: feature["properties"]["building_id"])
    counts["selected_unique"] = len(features)
    return features, counts


def prepare_river() -> tuple[dict, dict]:
    lines, provenance, all_nodes = [], [], {}
    for way_id in WAY_IDS:
        xml = ET.parse(RAW / f"osm_way_{way_id}_full.xml").getroot()
        nodes = {node.attrib["id"]: (float(node.attrib["lon"]), float(node.attrib["lat"])) for node in xml.findall("node")}
        for node_id, coordinates in nodes.items():
            if node_id in all_nodes and all_nodes[node_id] != coordinates:
                raise ValueError(f"Inconsistent OSM shared node {node_id}")
            all_nodes[node_id] = coordinates
        way = next(way for way in xml.findall("way") if way.attrib["id"] == str(way_id))
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        if tags.get("waterway") != "stream":
            raise ValueError(f"Unexpected classification for OSM way {way_id}")
        lines.append(LineString([nodes[node.attrib["ref"]] for node in way.findall("nd")]))
        provenance.append({
            "way_id": way_id,
            "way_version": int(way.attrib["version"]),
            "way_edited_at": way.attrib["timestamp"],
            "latest_node_edit_at": max(node.attrib["timestamp"] for node in xml.findall("node")),
            "tags": tags,
        })
    merged = linemerge(lines)
    if merged.geom_type != "LineString" or not merged.is_valid or not merged.is_simple:
        raise ValueError("The OSM ways no longer form one continuous, simple channel axis.")
    feature = {
        "type": "Feature",
        "properties": {
            "river_id": "chullo_osm",
            "name": "Chullo · eje OSM",
            "geometry_role": "axis",
            "source_id": "osm",
            "review_status": "reference_identification_cross_checked",
            "osm_way_ids": ",".join(map(str, WAY_IDS)),
            "observed_at": None,
        },
        "geometry": mapping(merged),
    }
    return feature, {"ways": provenance, "unique_node_count": len(all_nodes)}


def feature_collection(features: list) -> dict:
    # RFC 7946 GeoJSON always uses WGS84 longitude/latitude; no deprecated crs member.
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require all original inputs to exist in the local cache.")
    args = parser.parse_args()
    manifest = acquire(offline=args.offline)
    area = box(*BBOX)
    buildings, building_stats = prepare_buildings(area)
    river, river_stats = prepare_river()
    metric_area = gpd.GeoSeries([area], crs="EPSG:4326").to_crs(32719).iloc[0]
    metric_river = gpd.GeoSeries([shape(river["geometry"])], crs="EPSG:4326").to_crs(32719).iloc[0]
    metrics = gpd.GeoDataFrame.from_features(buildings, crs="EPSG:4326").to_crs(32719)
    clearances = [metric_area.distance(Point(point)) for point in [metric_river.coords[0], metric_river.coords[-1]]]
    if min(clearances) <= MAX_BAND_M:
        raise ValueError("Channel axis does not extend sufficiently beyond the pilot area for the maximum band.")
    building_clearances = [float(metrics.distance(Point(point)).min()) for point in [metric_river.coords[0], metric_river.coords[-1]]]
    if min(building_clearances) <= MAX_BAND_M:
        raise ValueError("Channel axis does not extend sufficiently beyond complete selected building geometries.")
    inv = Transformer.from_crs(32719, 4326, always_xy=True)
    checks = []
    for label, east, north, source_id, location in REFERENCE_POINTS:
        point = Point(east, north)
        lon, lat = inv.transform(east, north)
        checks.append({"id": label, "easting_m": east, "northing_m": north,
                       "longitude": lon, "latitude": lat, "crs": "EPSG:32719",
                       "source_id": source_id, "reference": location,
                       "distance_to_osm_axis_m": metric_river.distance(point),
                       "inside_pilot": metric_area.covers(point)})
    # A change larger than this demands renewed correspondence review, not auto-identification.
    if any(check["distance_to_osm_axis_m"] > 10 for check in checks):
        raise ValueError("OSM axis no longer matches the reviewed INGEMMET reference locations.")
    distances = metrics.geometry.distance(metric_river)
    validation = {
        "created_at": iso_timestamp(), "analysis_crs": "EPSG:32719", "bbox": BBOX,
        "buildings": building_stats, "river": river_stats,
        "axis_length_m": metric_river.length, "axis_endpoint_distance_from_study_area_m": clearances,
        "axis_endpoint_distance_from_complete_buildings_m": building_clearances,
        "supported_max_band_m": MAX_BAND_M, "reference_checks": checks,
        "counts_by_band_m": {str(band): int((distances <= band).sum()) for band in [10, 25, 50, 100]},
        "distance_range_m": [float(distances.min()), float(distances.max())],
        "minimum_building_area_m2": float(metrics.area.min()), "maximum_building_area_m2": float(metrics.area.max()),
        "limits": ["Reference points identify the channel; they do not estimate positional accuracy.",
                   "Only the mapped Chullo axis is evaluated, not every potential drainage.",
                   "No official hazard or marginal-strip polygon is included.",
                   "Building image dates and residential uses are unknown."]}
    write_json(OUT / "buildings.geojson", feature_collection(buildings))
    write_json(OUT / "rivers.geojson", feature_collection([river]))
    write_json(OUT / "study_area.geojson", feature_collection([{
        "type": "Feature", "properties": {"area_id": "chullo", "name": "Chullo · sector piloto", "analysis_crs": "EPSG:32719"},
        "geometry": mapping(area)}]))
    write_json(OUT / "validation.json", validation)
    write_json(ROOT / "config/study_areas.json", {"study_areas": [{
        "id": "chullo", "name": "Chullo · sector piloto", "path": "data/processed/chullo",
        "analysis_crs": "EPSG:32719", "bbox": BBOX, "max_band_m": MAX_BAND_M,
        "description": "Sector de demostración de proximidad al eje cartografiado de Chullo. Fechas de captura de edificaciones desconocidas.",
        "coverage_notes": "Eje Chullo continuo de 2,1 km; área piloto de 0,284 km². No inventario de todos los cauces ni delimitación legal."}]})
    common = {"original_crs": "EPSG:4326", "observed_at": None, "study_area_id": "chullo"}
    sources = [
        {**common, "source_id": "microsoft", "title": "Microsoft Global ML Building Footprints · Perú", "organization": "Microsoft",
         "url": "https://github.com/microsoft/GlobalMLBuildingFootprints", "download_url": TILE_URL,
         "published_at": "2026-08-13", "downloaded_at": manifest[TILE]["downloaded_at"],
         "license": "CDLA-Permissive-2.0", "license_url": "https://cdla.dev/permissive-2-0/",
         "attribution": "Edificaciones: Microsoft Global ML Building Footprints (CDLA Permissive 2.0).",
         "review_status": "Controles geométricos; no validación de campo ni fecha de captura local.",
         "notes": "Fecha de publicación no equivale a captura. Selección por intersección conserva huellas completas. Altura/confianza -1 se convierten en null.",
         "raw_files": [TILE], "sha256": manifest[TILE]["sha256"]},
        {**common, "source_id": "osm", "title": "OpenStreetMap · eje referencial de Chullo", "organization": "OpenStreetMap contributors",
         "url": "https://www.openstreetmap.org/way/25980163", "published_at": None,
         "downloaded_at": manifest[f"osm_way_{WAY_IDS[0]}_full.xml"]["downloaded_at"],
         "license": "ODbL-1.0", "license_url": "https://www.openstreetmap.org/copyright",
         "attribution": "© OpenStreetMap contributors; datos bajo ODbL 1.0.",
         "review_status": "Identificación de Chullo contrastada con 5 referencias INGEMMET; exactitud del eje no medida.",
         "notes": "Vías sin etiqueta name; denominación asignada por correspondencia documental. Eje, no borde del cauce. Fechas de edición no equivalen a observación. No validación de campo.",
         "raw_files": [f"osm_way_{wid}_full.xml" for wid in WAY_IDS], "ways": river_stats["ways"]},
        {"source_id": "ingemmet_a7406", "title": "INGEMMET A7406 · Yanahuara", "organization": "INGEMMET",
         "url": URLS["ingemmet_A7406.pdf"], "published_at": "2023", "observed_at": None,
         "downloaded_at": manifest["ingemmet_A7406.pdf"]["downloaded_at"], "original_crs": "EPSG:32719",
         "license": "Documento público; conservado solo como referencia de validación con atribución.",
         "review_status": "Cuadro 9 y figuras 17-20 inspeccionados. No se extrajeron polígonos de peligro.",
         "raw_files": ["ingemmet_A7406.pdf"]},
        {"source_id": "ingemmet_ot006", "title": "INGEMMET Opinión Técnica 006-2020 · Chullo", "organization": "INGEMMET",
         "url": URLS["ingemmet_OT006_2020.pdf"], "published_at": "2020-11", "observed_at": "2020-10-29",
         "downloaded_at": manifest["ingemmet_OT006_2020.pdf"]["downloaded_at"], "original_crs": "EPSG:32719",
         "license": "CC BY-NC-SA 4.0 según registro del repositorio; uso académico con atribución.",
         "license_url": "https://alicia.concytec.gob.pe/vufind/Record/INGEMMET_1dfd183cfa07c0df2c68330d870bb030/Details",
         "review_status": "Cuadro 1 y mapa de drenaje revisados; referencia de identificación, no geometría oficial digitalizada.",
         "raw_files": ["ingemmet_OT006_2020.pdf"]},
    ]
    write_json(ROOT / "data/sources.json", {"schema_version": 1, "sources": sources})
    write_json(OUT / "build_manifest.json", {"created_at": iso_timestamp(), "script": "scripts/prepare_data.py",
        "bbox": BBOX, "sources": manifest,
        "outputs": {name: sha256(OUT / name) for name in ["buildings.geojson", "rivers.geojson", "study_area.geojson"]}})
    print(json.dumps({"buildings": building_stats, "axis_length_m": round(metric_river.length, 1),
                      "counts_by_band_m": validation["counts_by_band_m"], "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
