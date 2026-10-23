"""Demo explanations preserve numerical rules and never infer safety or permission."""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from src.map_view import _building_popup, build_map
from src.risk_explanation import DemoRuleProvider, with_explanations


def test_exported_explanations_follow_active_thresholds_and_do_not_mutate_data():
    import pandas as pd

    from src.export import csv_bytes, geojson_bytes
    from src.proximity import PROXIMITY_LABELS

    table = gpd.GeoDataFrame({
        "building_id": ["a"], "distance_m": [65.0], "red_band_m": [25.0],
        "band_m": [75.0], "yellow_width_m": [50.0], "proximity_class": ["yellow"],
        "proximity_label": [PROXIMITY_LABELS["yellow"]], "within_band": [True],
    }, geometry=[box(-71.55, -16.4, -71.549, -16.399)], crs="EPSG:4326")
    exported = with_explanations(table)
    assert "demo_criterion" not in table
    assert "hasta 75" in exported.iloc[0]["demo_criterion"]
    assert exported.iloc[0]["demo_priority"] == "Revisión preventiva"
    properties = json.loads(geojson_bytes(exported))["features"][0]["properties"]
    assert properties["demo_rule_version"] == DemoRuleProvider().descriptor()["rule_version"]
    from io import BytesIO

    csv = pd.read_csv(BytesIO(csv_bytes(exported)))
    assert csv.iloc[0]["demo_criterion"] == properties["demo_criterion"]


@pytest.mark.parametrize(
    ("distance", "color"),
    [(0, "red"), (25, "red"), (25 + 0.0000005, "red"), (25.01, "yellow"),
     (50, "yellow"), (50.01, "green"), (None, "unknown"), (float("nan"), "unknown"),
     (-1, "unknown"), (float("inf"), "unknown"), (True, "unknown")],
)
def test_demo_explanation_respects_core_boundaries_and_missing_data(distance, color):
    explanation = DemoRuleProvider().explain(distance, 25, 50)
    assert explanation.classification == color
    assert explanation.red_band_m == 25 and explanation.yellow_width_m == 25
    assert explanation.outer_band_m == 50
    assert explanation.is_demonstration is True
    assert explanation.rule_version == "2026.10.demo.1"
    assert "verde no implica seguridad" in explanation.notice
    assert len(explanation.pending_factors) == 3 and len(explanation.actions) == 2


def test_changed_width_changes_explanation_without_changing_distance():
    provider = DemoRuleProvider()
    original = provider.explain(65, 25, 50)
    updated = provider.explain(65, 25, 75)
    assert original.classification == "green" and updated.classification == "yellow"
    assert original.distance_m == updated.distance_m == 65
    assert updated.yellow_width_m == 50
    assert provider.descriptor()["mode"] == "demo_rules"
    assert "no son umbrales oficiales" in provider.descriptor()["source"]


def test_explanation_rejects_contradictory_color_and_invalid_thresholds():
    provider = DemoRuleProvider()
    with pytest.raises(ValueError, match="no coinciden"):
        provider.explain(12, 25, 50, expected_class="green")
    with pytest.raises(ValueError):
        provider.explain(12, 50, 25)


def test_green_never_claims_construction_is_permitted_or_safety_is_established():
    green = DemoRuleProvider().explain(100, 25, 50)
    assert "seguridad sin evaluar" in green.priority
    assert "Verde no significa seguro ni edificable" in green.potential_exposure
    assert any("no autoriza construir" in action for action in green.actions)
    red = DemoRuleProvider().explain(10, 25, 50)
    assert "el modelo no determina si ocurre" in red.potential_exposure


def test_invalid_rule_file_is_rejected(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"schema_version": 1, "mode": "official_hazard"}), encoding="utf-8")
    with pytest.raises(ValueError, match="demostración"):
        DemoRuleProvider(path)


def test_self_contained_popup_escapes_input_and_exposes_pending_factors():
    row = {"building_id": '<img src=x onerror="alert(1)">', "nearest_river_name": "<script>bad()</script>"}
    result = _building_popup(row, DemoRuleProvider().explain(12.3, 25, 50))
    assert "<img" not in result and "<script>bad" not in result
    assert "&lt;img" in result and "&lt;script&gt;" in result
    for text in ("¿Por qué este color?", "12.30", "Límite rojo", "Ancho amarillo", "Faja marginal",
                 "Caudal", "Terreno", "Qué verificar", "2026.10.demo.1"):
        assert text in result
    assert "<details>" in result


@pytest.mark.parametrize("thresholds", [{}, {"band_m": 50, "red_band_m": 25}])
def test_map_provides_color_layers_local_popup_selection_and_visibility_counter(thresholds):
    summary = gpd.GeoDataFrame(
        {"building_id": ["red-house", "yellow-house", "green-house"],
         "distance_m": [12, 40, 80], "proximity_class": ["red", "yellow", "green"],
         "proximity_label": ["Rojo", "Amarillo", "Verde"]},
        geometry=[box(-71.55 + i * 0.001, -16.4, -71.5498 + i * 0.001, -16.3998) for i in range(3)],
        crs="EPSG:4326",
    )
    rivers = gpd.GeoDataFrame({"name": ["Chullo"]}, geometry=[LineString([(-71.55, -16.4), (-71.55, -16.39)])],
                              crs="EPSG:4326")
    area = gpd.GeoDataFrame(geometry=[box(-71.56, -16.41, -71.54, -16.38)], crs="EPSG:4326")
    bands = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    rendered = build_map(summary, rivers, area, bands, basemap=False, **thresholds).get_root().render()
    assert "building-visibility-counter" in rendered
    assert "overlayadd overlayremove" in rendered
    assert "feature.on('click'" in rendered and "feature.openPopup()" in rendered
    assert "feature.getBounds()" in rendered and "selected.group.resetStyle" in rendered
    assert all(color in rendered for color in ("#dc2626", "#eab308", "#16a34a"))
    assert "local_proximity_demo" not in rendered  # The user-facing popup shows version, not API implementation IDs.
    assert "2026.10.demo.1" in rendered
    assert rendered.count('"building_id":') >= 3


def test_map_color_filter_preserves_full_results_and_clears_hidden_selection(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    from src.ui import sidebar

    root = Path(__file__).resolve().parents[1]
    if not (root / "outputs/inference/chullo_demo/buildings.geojson").is_file():
        pytest.skip("Requires the saved Chullo demonstration.")
    monkeypatch.setenv("CHULLO_DEPLOYMENT", "cloud")
    monkeypatch.setattr(sidebar, "ml_status", lambda: {"available": False})
    app = testing.AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert not app.exception
    total = len(app.dataframe[0].value)
    color_filter = next(item for item in app.multiselect if item.label == "Colores visibles en el mapa")
    color_filter.set_value(["Rojo"]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == total
    selector = next(item for item in app.selectbox if item.label == "Identificador")
    assert len(selector.options) == int(app.metric[1].value) + 1
    selected_id = selector.options[1]
    selector.set_value(selected_id).run()
    assert any("Regla activa" in item.value for item in app.markdown)
    next(item for item in app.multiselect if item.label == "Colores visibles en el mapa").set_value([]).run()
    assert not app.exception and len(app.dataframe[0].value) == total
    selector = next(item for item in app.selectbox if item.label == "Identificador")
    assert selector.value == "" and len(selector.options) == 1
    assert "building-visibility-counter" in app.get("iframe")[0].proto.srcdoc
