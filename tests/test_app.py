"""Regression of the saved Chullo workflow; never runs or changes model outputs.

Counts describe the versioned demonstration, not independently verified buildings.
The optional integration tests require the bundled image and inference artifacts.
"""

import json
import re
import shutil
import subprocess
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SAVED_RUN = ROOT / "outputs/inference/chullo_demo"
REQUIRED_ARTIFACTS = [
    ROOT / "data/imagery/chullo/chullo_rgb.tif",
    SAVED_RUN / "manifest.json",
    SAVED_RUN / "buildings.geojson",
]
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED_ARTIFACTS),
    reason="The Chullo UI regression requires the bundled demo image and saved run.",
)


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def assert_rendered(app, total=237, red=27, yellow=29, outer=50):
    assert not app.exception, [element.value for element in app.exception]
    assert not app.error, [element.value for element in app.error]
    assert [element.value for element in app.metric[:4]] == [
        str(total),
        str(red),
        str(yellow),
        str(total - red - yellow),
    ]
    assert f"{outer:g} m" in app.metric[2].label
    assert len(app.dataframe[0].value) == total


@pytest.fixture
def saved_app(monkeypatch):
    """Capture real download payloads while preserving the Streamlit render path."""
    fingerprints = {path: sha256(path.read_bytes()).hexdigest() for path in SAVED_RUN.iterdir() if path.is_file()}
    downloads = {}
    original_download = st.download_button

    def capture_download(label, data, *args, **kwargs):
        downloads[label] = {"data": data, "file_name": kwargs.get("file_name")}
        return original_download(label, data, *args, **kwargs)

    def prohibit_process(*args, **kwargs):
        raise AssertionError("Saved-result UI tests must not launch workers or downloads.")

    monkeypatch.setattr(st, "download_button", capture_download)
    monkeypatch.setattr(subprocess, "Popen", prohibit_process)
    app = streamlit_testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    yield app.run(), downloads
    assert {
        path: sha256(path.read_bytes()).hexdigest() for path in SAVED_RUN.iterdir() if path.is_file()
    } == fingerprints, "Viewing saved results must leave the inference run unchanged."


def test_traffic_width_updates_immediately_without_inference_and_mode_changes_keep_results(saved_app):
    app, downloads = saved_app
    assert_rendered(app)
    original = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))

    widget(app.slider, "Ancho de la zona amarilla (m)").set_value(50).run()
    assert_rendered(app, yellow=66, outer=75)
    updated = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))
    pd.testing.assert_series_equal(original.distance_m, updated.distance_m)
    assert updated.yellow_width_m.eq(50).all()

    mode = widget(app.radio, "Origen de las edificaciones")
    mode.set_value("Inventario Microsoft").run()
    assert_rendered(app, total=177, red=31, yellow=48, outer=75)
    assert app.metric[0].label == "Edificaciones analizadas"
    assert any("no ejecuta detección" in element.value for element in app.info)

    widget(app.radio, "Origen de las edificaciones").set_value("Detección IA desde imagen").run()
    assert_rendered(app, yellow=66, outer=75)
    assert app.metric[0].label == "Contornos detectados"


def test_sources_and_full_downloads_follow_applied_analysis_not_table_filter(saved_app):
    app, downloads = saved_app
    assert_rendered(app)
    captions = "\n".join(element.value for element in app.caption)
    assert "2026-03-04" in captions
    assert "Google Earth" in captions and "Airbus" in captions
    assert any("Georreferenciación aproximada" in element.value for element in app.warning)

    widget(app.slider, "Puntuación mínima del detector").set_value(0.60).run()
    assert_rendered(app)
    assert any("0.50" in element.value and "0.60" in element.value for element in app.info)

    red_option = next(option for option in widget(app.selectbox, "Mostrar").options if "Rojo" in option)
    widget(app.selectbox, "Mostrar").set_value(red_option).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 27

    csv_download = downloads["Descargar tabla CSV"]
    table = pd.read_csv(BytesIO(csv_download["data"]))
    assert "25" in csv_download["file_name"] and "50" in csv_download["file_name"]
    assert len(table) == 237
    assert table["within_band"].sum() == 56
    assert table["band_m"].eq(50).all()
    assert table["red_band_m"].eq(25).all()
    assert table["yellow_width_m"].eq(25).all()
    assert table["proximity_class"].value_counts().to_dict() == {"green": 181, "yellow": 29, "red": 27}
    assert table["source_id"].eq("geoai_building_inference").all()
    assert table["observed_at"].eq("2026-03-04").all()

    features = json.loads(downloads["Descargar mapa GeoJSON"]["data"])["features"]
    assert len(features) == len(table)
    assert {feature["properties"]["building_id"] for feature in features} == set(table["building_id"])
    with ZipFile(BytesIO(downloads["Descargar paquete completo"]["data"])) as archive:
        manifest = json.loads(archive.read("metadatos.json"))
        assert manifest["band_m"] == 50
        assert manifest["building_count"] == 237
        assert manifest["within_band_count"] == 56
        assert manifest["source_metadata"]["analysis_mode"] == "image_inference"
        assert manifest["analysis_crs"] == ["EPSG:32719"]
        assert len(pd.read_csv(archive.open("resumen_edificaciones.csv"))) == 237


def test_red_control_keeps_valid_yellow_width_at_sector_limit(saved_app):
    app, downloads = saved_app
    widget(app.slider, "Límite de la zona roja (m)").set_value(50).run()
    assert_rendered(app, red=56, yellow=37, outer=75)
    widget(app.slider, "Límite de la zona roja (m)").set_value(99).run()
    assert_rendered(app, red=111, yellow=0, outer=100)
    table = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))
    assert table.red_band_m.eq(99).all() and table.yellow_width_m.eq(1).all()
    assert table.band_m.le(100).all()
    widget(app.slider, "Límite de la zona roja (m)").set_value(25).run()
    assert not app.exception and not app.error
    width = widget(app.slider, "Ancho de la zona amarilla (m)").value
    table = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))
    assert table.yellow_width_m.eq(width).all()
    assert table.band_m.eq(25 + width).all()


def test_session_run_is_selected_after_success_and_retained_after_failure(saved_app, monkeypatch, tmp_path):
    """Exercise the real UI with a simulated worker; never overwrite the bundled demo."""
    from src.application.session_workspace import RESULTS_KEY, SESSION_KEY, new_detection_output
    from src.ui import sidebar, workflow

    app, _ = saved_app
    calls = []

    def completed_worker(raster_path, output_dir, **kwargs):
        calls.append(output_dir)
        assert output_dir != SAVED_RUN
        shutil.copytree(SAVED_RUN, output_dir)
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(sidebar, "ml_status", lambda: {"available": True})
    monkeypatch.setattr(workflow, "new_detection_output", lambda session: new_detection_output(session, tmp_path))
    monkeypatch.setattr(workflow, "run_detection", completed_worker)
    app.run()
    widget(app.button, "Detectar en esta imagen").click().run()
    assert_rendered(app)
    selected = app.session_state[RESULTS_KEY]["chullo:bundled"]
    assert Path(selected) == calls[0]
    assert app.session_state[SESSION_KEY] in Path(selected).parts

    app.run()
    assert app.session_state[RESULTS_KEY]["chullo:bundled"] == selected

    def failing_worker(raster_path, output_dir, **kwargs):
        assert output_dir not in {SAVED_RUN, Path(selected)}
        raise RuntimeError("Fallo simulado de prueba")

    monkeypatch.setattr(workflow, "run_detection", failing_worker)
    widget(app.button, "Detectar en esta imagen").click().run()
    assert not app.exception
    assert any("Fallo simulado de prueba" in element.value for element in app.error)
    assert app.session_state[RESULTS_KEY]["chullo:bundled"] == selected
    assert [element.value for element in app.metric[:4]] == ["237", "27", "29", "181"]
    assert any("Última detección completa conservada" in element.value for element in app.success)


@pytest.mark.parametrize("deployment", ["cloud", "local"])
def test_cloud_map_contains_interactive_geometry_and_preserves_identifier_selection(saved_app, monkeypatch, deployment):
    """Use real rendered Leaflet HTML without relying on a custom-component handshake."""
    from src.ui import map_panel

    app, _ = saved_app
    monkeypatch.setenv("CHULLO_DEPLOYMENT", deployment)
    assert not hasattr(map_panel, "st_folium"), "The map must not depend on a custom-component handshake."
    app.run()
    assert_rendered(app)
    frames = app.get("iframe")
    assert len(frames) == 1
    rendered_map = frames[0].proto.srcdoc
    assert '"FeatureCollection"' in rendered_map and '"coordinates"' in rendered_map
    assert "building_id" in rendered_map and "bindPopup" in rendered_map
    assert "L.control.layers" in rendered_map
    assert "L.imageOverlay" in rendered_map and "data:image/png;base64," in rendered_map
    assert all(color in rendered_map for color in ("#dc2626", "#eab308", "#16a34a"))
    assert "proximity_class" in rendered_map

    selector = widget(app.selectbox, "Identificador")
    selected_id = selector.options[1]
    selector.set_value(selected_id).run()
    assert_rendered(app)
    selected_map = app.get("iframe")[0].proto.srcdoc
    assert selected_id in selected_map
    collections = [
        json.JSONDecoder().raw_decode(selected_map[match.end() :])[0]
        for match in re.finditer(r"geo_json_[0-9a-f]+_add\(", selected_map)
    ]
    selected_geometry = next(
        feature["geometry"]
        for collection in collections
        for feature in collection["features"]
        if feature["properties"].get("building_id") == selected_id
    )
    assert any(
        len(collection["features"]) == 1
        and collection["features"][0]["properties"] == {}
        and collection["features"][0]["geometry"] == selected_geometry
        for collection in collections
    ), "Selecting an identifier must add the exact building geometry as its highlight layer."
    assert ".fitBounds(" in selected_map


@pytest.mark.parametrize("variant", ["spacenet2_khartoum_v1", "chullo_local_v1"])
def test_model_switch_keeps_actual_provenance_and_separates_completed_session_results(
    saved_app, monkeypatch, tmp_path, variant
):
    """Candidate fixtures live in tmp_path; both the original demo and real weights remain untouched."""
    from src.application.session_workspace import RESULTS_KEY, new_detection_output
    from src.inference.model import MODEL_ID
    from src.model_catalog import LOCAL_VARIANT, model_presentation
    from src.ui import sidebar, workflow

    app, downloads = saved_app
    demo_copy = tmp_path / "outputs" / "inference" / "chullo_demo"
    shutil.copytree(SAVED_RUN, demo_copy)
    monkeypatch.setattr(sidebar, "ROOT", tmp_path)
    monkeypatch.setattr(
        sidebar,
        "model_choices",
        lambda: {
            "usa": "Original · GeoAI USA",
            variant: model_presentation({"variant": variant})["title"],
        },
    )
    monkeypatch.setattr(sidebar, "ml_status", lambda: {"available": True})
    monkeypatch.setattr(workflow, "new_detection_output", lambda session: new_detection_output(session, tmp_path))
    app.run()
    assert widget(app.selectbox, "Modelo de edificaciones").value == "usa"
    widget(app.selectbox, "Modelo de edificaciones").set_value(variant).run()
    assert_rendered(app)
    assert any("otro modelo" in element.value for element in app.info)
    before_run = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))
    assert before_run.model_id.eq(MODEL_ID).all(), "Changing the selector must not relabel original detections"
    with ZipFile(BytesIO(downloads["Descargar paquete completo"]["data"])) as archive:
        before_metadata = json.loads(archive.read("metadatos.json"))["source_metadata"]["detection"]
    assert before_metadata["model_id"] == MODEL_ID
    assert before_metadata["model"]["trained_locally"] is False
    before_footer = next(element.value for element in app.markdown if 'class="foot"' in element.value)
    assert "Edificaciones: Original · GeoAI USA" in before_footer
    assert model_presentation({"variant": variant})["title"] not in before_footer

    candidate_id = "synthetic-test/candidate"
    candidate_hash = "2" * 64
    calls = []

    def completed_candidate(raster_path, output_dir, **kwargs):
        assert kwargs["model_variant"] == variant
        calls.append(output_dir)
        shutil.copytree(SAVED_RUN, output_dir)
        geojson_path = output_dir / "buildings.geojson"
        features = json.loads(geojson_path.read_bytes())
        for feature in features["features"]:
            feature["properties"]["model_id"] = candidate_id
        geojson_path.write_text(json.dumps(features), encoding="utf-8")
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model_id"] = candidate_id
        manifest["model"].update(
            {
                "id": candidate_id,
                "name": "Synthetic adjusted detector fixture",
                "variant": variant,
                "trained_locally": True,
                "sha256": candidate_hash,
                "evaluation_summary": {"scope": "Synthetic test fixture, no real evaluation"},
            }
        )
        manifest["outputs"]["buildings_sha256"] = sha256(geojson_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr(workflow, "run_detection", completed_candidate)
    widget(app.button, "Detectar en esta imagen").click().run()
    assert_rendered(app)
    assert not any("otro modelo" in element.value for element in app.info)
    candidate_key = f"chullo:bundled:{variant}"
    assert Path(app.session_state[RESULTS_KEY][candidate_key]) == calls[0]
    candidate = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))
    assert candidate.model_id.eq(candidate_id).all()
    with ZipFile(BytesIO(downloads["Descargar paquete completo"]["data"])) as archive:
        source_metadata = json.loads(archive.read("metadatos.json"))["source_metadata"]
    assert source_metadata["building_mode"] == "finetuned_model_inference"
    assert source_metadata["detection"]["model"]["sha256"] == candidate_hash
    assert source_metadata["detection"]["model"]["trained_locally"] is True
    candidate_footer = next(element.value for element in app.markdown if 'class="foot"' in element.value)
    assert f"Edificaciones: {model_presentation({'variant': variant})['title']}" in candidate_footer
    detection_source = next(source for source in source_metadata["sources"] if source.get("model_id") == candidate_id)
    assert detection_source["organization"] == "Proyecto Chullo; ajuste supervisado sobre pesos de giswqs/GeoAI"
    assert detection_source["model_sha256"] == candidate_hash
    if variant == LOCAL_VARIANT:
        captions = "\n".join(element.value for element in app.caption)
        warnings = "\n".join(element.value for element in app.warning)
        assert "3 recortes originales" in captions and "Google Earth Pro / Airbus" in captions
        assert "sin revisión humana" in warnings
        assert "Jartum" not in captions and "Jartum" not in warnings

    widget(app.selectbox, "Modelo de edificaciones").set_value("usa").run()
    assert_rendered(app)
    original_again = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))
    assert original_again.model_id.eq(MODEL_ID).all()
    original_footer = next(element.value for element in app.markdown if 'class="foot"' in element.value)
    assert "Edificaciones: Original · GeoAI USA" in original_footer
    assert Path(app.session_state[RESULTS_KEY][candidate_key]) == calls[0]

    widget(app.selectbox, "Modelo de edificaciones").set_value(variant).run()
    assert_rendered(app)
    candidate_again = pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"]))
    assert candidate_again.model_id.eq(candidate_id).all()
    assert len(calls) == 1, "Switching saved model results must not run inference"

    def failing_candidate(*args, **kwargs):
        assert kwargs["model_variant"] == variant
        raise RuntimeError("Synthetic candidate failure")

    monkeypatch.setattr(workflow, "run_detection", failing_candidate)
    widget(app.button, "Detectar en esta imagen").click().run()
    assert not app.exception
    assert any("Synthetic candidate failure" in element.value for element in app.error)
    assert Path(app.session_state[RESULTS_KEY][candidate_key]) == calls[0]
    assert pd.read_csv(BytesIO(downloads["Descargar tabla CSV"]["data"])).model_id.eq(candidate_id).all()
