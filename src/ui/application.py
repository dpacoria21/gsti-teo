"""Compose the Streamlit application from independent presentation components."""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from src.settings import PROJECT_ROOT
from src.ui.header import render_header
from src.ui.map_panel import render_map_panel
from src.ui.overview import render_overview
from src.ui.sidebar import render_sidebar
from src.ui.sources_panel import render_sources_panel
from src.ui.table_panel import render_table_panel
from src.ui.workflow import applied_band, execute_requested_detection, prepare_analysis


def _study_areas() -> list[dict]:
    """Report configuration errors at the UI boundary."""
    try:
        configuration = json.loads((PROJECT_ROOT / "config" / "study_areas.json").read_text(encoding="utf-8-sig"))
        areas = configuration.get("study_areas", []) if isinstance(configuration, dict) else configuration
        if not areas:
            raise ValueError("No hay áreas de estudio preparadas en la configuración.")
        return areas
    except (OSError, ValueError, TypeError) as error:
        st.title("Análisis de proximidad")
        st.error(f"No se pudo abrir la configuración local: {error}")
        st.info("Los archivos de configuración y datos deben estar disponibles antes de iniciar la presentación.")
        st.stop()


def main() -> None:
    """Render one rerun; imported modules never render or execute inference."""
    st.set_page_config(
        page_title="Chullo · Análisis territorial", page_icon="🌐", layout="wide", initial_sidebar_state="expanded"
    )
    styles = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>\n{styles}\n</style>", unsafe_allow_html=True)

    selection = render_sidebar(_study_areas())
    band_m = applied_band(selection)
    render_header(selection)
    selection, band_m, detection_failed = execute_requested_detection(selection, band_m)
    view = prepare_analysis(selection, band_m, detection_failed)
    render_overview(selection, view)

    map_tab, table_tab, sources_tab = st.tabs(["Mapa del sector", "Resultados y descargas", "Fuentes y alcance"])
    with map_tab:
        render_map_panel(selection, view)
    with table_tab:
        render_table_panel(selection, view)
    with sources_tab:
        render_sources_panel(selection, view)

    building_credit = (
        "Edificaciones: inferencia de modelo preentrenado" if view.detection else "Edificaciones: Microsoft"
    )
    st.markdown(
        f'<div class="foot">{html.escape(selection.area_name.upper())} · Prototipo académico de análisis territorial · {building_credit} · Procedencia de las capas en «Fuentes y alcance» · La proximidad es un indicador espacial.</div>',
        unsafe_allow_html=True,
    )
