"""Regression of the saved Chullo workflow; never runs or changes model outputs.

Counts describe the versioned demonstration, not independently verified buildings.
The optional integration tests require the bundled image and inference artifacts.
"""

import json
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


def assert_rendered(app, total, inside, band):
    assert not app.exception, [element.value for element in app.exception]
    assert not app.error, [element.value for element in app.error]
    assert [element.value for element in app.metric[:3]] == [str(total), str(inside), str(total - inside)]
    assert f"{band} m" in app.metric[1].label
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


def test_band_changes_require_apply_and_mode_changes_keep_correct_results(saved_app):
    app, _ = saved_app
    assert_rendered(app, total=237, inside=27, band=25)

    widget(app.select_slider, "Banda de proximidad").set_value(50).run()
    assert_rendered(app, total=237, inside=27, band=25)
    assert any("50 m" in element.value for element in app.info)
    widget(app.button, "Analizar sector").click().run()
    assert_rendered(app, total=237, inside=56, band=50)

    mode = widget(app.radio, "Origen de las edificaciones")
    mode.set_value("Inventario Microsoft").run()
    assert_rendered(app, total=177, inside=60, band=50)
    assert app.metric[0].label == "Edificaciones analizadas"
    assert any("no ejecuta detección" in element.value for element in app.info)

    widget(app.radio, "Origen de las edificaciones").set_value("Detección IA desde imagen").run()
    assert_rendered(app, total=237, inside=56, band=50)
    assert app.metric[0].label == "Contornos detectados"


def test_sources_and_full_downloads_follow_applied_analysis_not_table_filter(saved_app):
    app, downloads = saved_app
    assert_rendered(app, total=237, inside=27, band=25)
    captions = "\n".join(element.value for element in app.caption)
    assert "2026-03-04" in captions
    assert "Google Earth" in captions and "Airbus" in captions
    assert any("Georreferenciación aproximada" in element.value for element in app.warning)

    widget(app.slider, "Puntuación mínima del detector").set_value(0.60).run()
    assert_rendered(app, total=237, inside=27, band=25)
    assert any("0.50" in element.value and "0.60" in element.value for element in app.info)

    widget(app.selectbox, "Mostrar").set_value("Dentro de la banda").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 27
    assert app.dataframe[0].value["Dentro de banda"].all()

    csv_download = downloads["Descargar tabla CSV"]
    table = pd.read_csv(BytesIO(csv_download["data"]))
    assert csv_download["file_name"] == "chullo_deteccion_banda_25m.csv"
    assert len(table) == 237
    assert table["within_band"].sum() == 27
    assert table["band_m"].eq(25).all()
    assert table["source_id"].eq("geoai_building_inference").all()
    assert table["observed_at"].eq("2026-03-04").all()

    features = json.loads(downloads["Descargar mapa GeoJSON"]["data"])["features"]
    assert len(features) == len(table)
    assert {feature["properties"]["building_id"] for feature in features} == set(table["building_id"])
    with ZipFile(BytesIO(downloads["Descargar paquete completo"]["data"])) as archive:
        manifest = json.loads(archive.read("metadatos.json"))
        assert manifest["band_m"] == 25
        assert manifest["building_count"] == 237
        assert manifest["within_band_count"] == 27
        assert manifest["source_metadata"]["analysis_mode"] == "image_inference"
        assert manifest["analysis_crs"] == ["EPSG:32719"]
        assert len(pd.read_csv(archive.open("resumen_edificaciones.csv"))) == 237


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
    assert_rendered(app, total=237, inside=27, band=25)
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
    assert [element.value for element in app.metric[:3]] == ["237", "27", "210"]
    assert any("Última detección completa conservada" in element.value for element in app.success)
