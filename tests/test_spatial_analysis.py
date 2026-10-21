"""Casos geométricos controlados; no usan edificios reales como verdad de terreno."""

import json
from io import BytesIO
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon, box

from src.data_loader import load_study_area
from src.export import analysis_bundle, csv_bytes, geojson_bytes
from src.spatial_analysis import analyze_proximity

CRS = "EPSG:32719"
X, Y = 220_000, 8_187_000


def buildings(geometries=None):
    shapes = geometries if geometries is not None else [box(X, Y, X + 20, Y + 20)]
    return gpd.GeoDataFrame(
        {
            "building_id": [f"b{index}" for index in range(len(shapes))],
            "source_id": ["test_buildings"] * len(shapes),
            "observed_at": [None] * len(shapes),
            "review_status": ["Caso de prueba"] * len(shapes),
        },
        geometry=shapes,
        crs=CRS,
    )


def rivers(geometries=None, roles=None):
    shapes = geometries if geometries is not None else [LineString([(X + 100, Y - 100), (X + 100, Y + 100)])]
    return gpd.GeoDataFrame(
        {
            "river_id": [f"r{index}" for index in range(len(shapes))],
            "name": [f"Cauce {index}" for index in range(len(shapes))],
            "geometry_role": roles or ["axis"] * len(shapes),
            "source_id": ["test_rivers"] * len(shapes),
        },
        geometry=shapes,
        crs=CRS,
    )


def test_known_distance_is_from_building_edge_not_centroid():
    result = analyze_proximity(buildings(), rivers(), band_m=25)
    row = result["summary"].iloc[0]
    assert row.distance_m == pytest.approx(80, abs=1e-6)
    assert not row.within_band
    assert row.distance_reference == "eje cartografiado"
    assert row.hazard_status == row.margin_status == "No evaluado"
    assert result["summary"].crs.to_epsg() == 4326
    assert result["bands"].crs.to_epsg() == 4326


def test_boundary_is_inclusive_and_neighbour_outside_remains_outside():
    shapes = [
        box(X, Y, X + 20, Y + 20),
        box(X - 0.001, Y + 30, X + 19.999, Y + 50),
        box(X + 0.001, Y - 30, X + 20.001, Y - 10),
    ]
    result = analyze_proximity(buildings(shapes).to_crs(4326), rivers().to_crs(4326), band_m=80)
    assert result["summary"].within_band.tolist() == [True, False, True]


def test_intersection_is_zero_and_zero_band_can_be_used():
    result = analyze_proximity(buildings([box(X + 95, Y, X + 105, Y + 20)]), rivers(), band_m=0)
    assert result["summary"].iloc[0].distance_m == 0
    assert result["summary"].iloc[0].within_band
    assert result["bands"].empty


def test_polygon_channel_uses_separation_and_intersection():
    channel = rivers([box(X + 100, Y - 100, X + 120, Y + 100)], ["channel_polygon"])
    shapes = [box(X, Y, X + 20, Y + 20), box(X + 105, Y, X + 115, Y + 20)]
    summary = analyze_proximity(buildings(shapes), channel, band_m=25)["summary"]
    assert summary.distance_m.tolist() == [80, 0]
    assert summary.within_band.tolist() == [False, True]
    assert set(summary.distance_reference) == {"polígono del cauce"}


def test_every_river_relation_is_retained_without_duplicate_building_counts():
    river_shapes = [
        LineString([(X + 100, Y - 100), (X + 100, Y + 100)]),
        LineString([(X + 10, Y - 100), (X + 10, Y + 100)]),
    ]
    result = analyze_proximity(buildings(), rivers(river_shapes), band_m=100)
    assert len(result["summary"]) == 1
    assert len(result["relations"]) == 2
    assert result["relations"].distance_m.tolist() == [80, 0]
    assert result["summary"].iloc[0].nearest_river_id == "r1"
    assert result["summary"].iloc[0].rivers_within_band == 2
    assert result["summary"].within_band.sum() == 1


def test_equivalent_geographic_inputs_and_nondefault_indices_produce_same_distances():
    source_buildings, source_rivers = buildings(), rivers()
    source_buildings.index = [40]
    source_rivers.index = [95]
    before = source_buildings.copy()
    projected = analyze_proximity(source_buildings, source_rivers)["summary"]
    geographic = analyze_proximity(source_buildings.to_crs(4326), source_rivers.to_crs(4326))["summary"]
    assert geographic.distance_m.iloc[0] == pytest.approx(projected.distance_m.iloc[0], abs=1e-6)
    assert source_buildings.equals(before)


@pytest.mark.parametrize("band", [-1, 1001, float("nan"), float("inf"), "25", True])
def test_invalid_band_rejected(band):
    with pytest.raises(ValueError, match="banda analítica"):
        analyze_proximity(buildings(), rivers(), band)


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:2277", "not-a-crs"])
def test_analysis_requires_valid_projected_metric_crs(crs):
    with pytest.raises(ValueError, match="CRS"):
        analyze_proximity(buildings(), rivers(), analysis_crs=crs)


def test_missing_crs_cannot_be_assumed_wgs84():
    layer = buildings().set_crs(None, allow_override=True)
    with pytest.raises(ValueError, match="falta el CRS"):
        analyze_proximity(layer, rivers())


@pytest.mark.parametrize(
    "bad_geometry",
    [
        None,
        Polygon(),
        Polygon([(X, Y), (X + 10, Y + 10), (X + 10, Y), (X, Y + 10), (X, Y)]),
        LineString([(X, Y), (X + 10, Y + 10)]),
    ],
)
def test_unusable_building_geometry_rejected(bad_geometry):
    with pytest.raises(ValueError, match="Edificaciones"):
        analyze_proximity(buildings([bad_geometry]), rivers())


def test_duplicate_ids_and_mismatched_river_geometry_role_rejected():
    layer = buildings([box(X, Y, X + 20, Y + 20)] * 2)
    layer["building_id"] = "same"
    with pytest.raises(ValueError, match="duplicados"):
        analyze_proximity(layer, rivers())
    channel = rivers()
    channel["geometry_role"] = "channel_polygon"
    with pytest.raises(ValueError, match="geometry_role"):
        analyze_proximity(buildings(), channel)


def test_empty_river_coverage_does_not_mean_outside_band():
    with pytest.raises(ValueError, match="capa está vacía"):
        analyze_proximity(buildings(), rivers().iloc[:0])


def test_export_preserves_values_sources_geometries_and_all_relations():
    result = analyze_proximity(buildings(), rivers(), band_m=80)
    source_metadata = {"sources": [{"source_id": "test_buildings", "observed_at": None}]}
    blob = analysis_bundle(result["summary"], result["relations"], source_metadata, 80)
    with ZipFile(BytesIO(blob)) as archive:
        table = pd.read_csv(BytesIO(archive.read("resumen_edificaciones.csv")))
        detail = pd.read_csv(BytesIO(archive.read("relaciones_cauces.csv")))
        features = json.loads(archive.read("edificaciones.geojson"))["features"]
        manifest = json.loads(archive.read("metadatos.json"))
    assert table.distance_m.iloc[0] == 80
    assert table.within_band.iloc[0]
    assert len(table) == len(features) == manifest["building_count"] == 1
    assert len(detail) == manifest["relation_count"] == 1
    assert manifest["source_metadata"] == source_metadata
    assert manifest["band_m"] == 80
    assert manifest["analysis_crs"] == [CRS]
    assert manifest["generated_at_lima"].endswith("-05:00")
    assert "geometry" not in table.columns
    longitude, latitude = features[0]["geometry"]["coordinates"][0][0]
    assert -180 <= longitude <= 180 and -90 <= latitude <= 90


def test_csv_neutralizes_formulas_but_not_negative_numeric_values():
    table = pd.DataFrame({"name": ["=1+1", " +1+1", "@SUM(1)", "Chullo"], "distance_m": [-1, 2, 3, 4]})
    blob = csv_bytes(table)
    assert blob.startswith(b"\xef\xbb\xbf")
    round_trip = pd.read_csv(BytesIO(blob))
    assert round_trip.name.tolist() == ["'=1+1", "' +1+1", "'@SUM(1)", "Chullo"]
    assert round_trip.distance_m.tolist() == [-1, 2, 3, 4]


def test_bundle_rejects_inconsistent_band_parameter():
    result = analyze_proximity(buildings(), rivers(), 25)
    with pytest.raises(ValueError, match="banda"):
        analysis_bundle(result["summary"], result["relations"], {}, 50)


def test_geojson_needs_declared_crs():
    with pytest.raises(ValueError, match="CRS"):
        geojson_bytes(buildings().set_crs(None, allow_override=True))


def test_geojson_and_bundle_serialize_dates_parsed_by_gdal():
    layer = buildings()
    layer["observed_at"] = pd.to_datetime(["2018-06-05"])
    summary = analyze_proximity(layer, rivers(), 25)
    feature = json.loads(geojson_bytes(summary["summary"]))["features"][0]
    assert feature["properties"]["observed_at"] == "2018-06-05T00:00:00"
    bundle = analysis_bundle(summary["summary"], summary["relations"], {"analysis_mode": "image_inference"}, 25)
    with ZipFile(BytesIO(bundle)) as archive:
        metadata = json.loads(archive.read("metadatos.json"))
    assert any("Detecciones automáticas" in text for text in metadata["limitations"])


def test_loader_preserves_provenance_and_rejects_unregistered_sources(tmp_path):
    (tmp_path / "config").mkdir()
    area_path = tmp_path / "data" / "processed" / "example"
    area_path.mkdir(parents=True)
    config = {
        "study_areas": [{"id": "example", "name": "Ejemplo", "path": "data/processed/example", "analysis_crs": CRS}]
    }
    (tmp_path / "config" / "study_areas.json").write_text(json.dumps(config), encoding="utf-8")
    source_file = tmp_path / "data" / "sources.json"
    source_file.write_text(
        json.dumps({"sources": [{"source_id": "test_buildings"}, {"source_id": "test_rivers"}]}), encoding="utf-8"
    )
    buildings().to_crs(4326).to_file(area_path / "buildings.geojson", driver="GeoJSON")
    rivers().to_crs(4326).to_file(area_path / "rivers.geojson", driver="GeoJSON")
    gpd.GeoDataFrame({"area_id": ["example"]}, geometry=[box(X - 100, Y - 100, X + 200, Y + 200)], crs=CRS).to_crs(
        4326
    ).to_file(area_path / "area.geojson", driver="GeoJSON")
    data = load_study_area("example", root=tmp_path)
    assert set(data) == {"buildings", "rivers", "area", "metadata", "config"}
    assert data["buildings"].crs.to_epsg() == 4326
    assert data["config"]["analysis_crs"] == CRS
    source_file.write_text('{"sources": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="fuente"):
        load_study_area("example", root=tmp_path)
