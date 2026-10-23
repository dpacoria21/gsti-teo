"""Filter the displayed table and download complete analysis outputs."""

import pandas as pd
import streamlit as st

from src.export import analysis_bundle, csv_bytes, geojson_bytes
from src.risk_explanation import DemoRuleProvider, with_explanations
from src.ui.formatting import number
from src.ui.models import AnalysisView, SidebarSelection


def render_table_panel(selection: SidebarSelection, view: AnalysisView) -> None:
    summary = with_explanations(view.summary)
    detection = view.detection
    active_area = selection.area_id
    band_m = view.band_m
    red_band_m = view.red_band_m
    analysis = view.analysis
    metadata = view.metadata
    provider = DemoRuleProvider()
    st.markdown("### Resultados por edificación")
    st.caption(
        "La tabla resume el cauce más cercano. El paquete de análisis conserva también las relaciones con todos los cauces incluidos."
    )
    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        table_filter = st.selectbox(
            "Mostrar", ["Todas las edificaciones", "Rojo", "Amarillo", "Verde", "Sin evaluación"]
        )
    with search_col:
        query = st.text_input("Buscar identificador", placeholder="Escribe parte del identificador…")
    filtered = summary.copy()
    color_filters = {"Rojo": "red", "Amarillo": "yellow", "Verde": "green"}
    if table_filter in color_filters:
        filtered = filtered.loc[filtered["proximity_class"].eq(color_filters[table_filter]).fillna(False)]
    elif table_filter == "Sin evaluación":
        filtered = filtered.loc[filtered["distance_m"].isna()]
    if query:
        filtered = filtered.loc[filtered["building_id"].astype(str).str.contains(query, case=False, regex=False)]
    explanations = [
        provider.explain(row["distance_m"], red_band_m, band_m, row.get("proximity_class"))
        for _, row in filtered.iterrows()
    ]
    filtered["demo_priority"] = [item.priority for item in explanations]
    filtered["demo_criterion"] = [item.criterion for item in explanations]
    table_columns = ["building_id", "distance_m", "proximity_label", "demo_priority", "demo_criterion", "nearest_river_name"]
    if detection and "confidence" in filtered.columns:
        table_columns.append("confidence")
    visible = pd.DataFrame(filtered[table_columns]).rename(
        columns={
            "building_id": "Edificación",
            "distance_m": "Distancia al eje (m)",
            "proximity_label": "Semáforo de proximidad",
            "demo_priority": "Prioridad de revisión · demo",
            "demo_criterion": "Por qué este color · demo",
            "nearest_river_name": "Cauce más cercano",
            "confidence": "Puntuación del detector",
        }
    )
    st.dataframe(
        visible,
        hide_index=True,
        width="stretch",
        height=430,
        column_config={
            "Distancia al eje (m)": st.column_config.NumberColumn(format="%.1f m"),
            "Puntuación del detector": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        f"{number(len(filtered))} filas visibles. Las descargas incluyen todas las edificaciones del análisis aplicado, sin el filtro de la tabla."
    )
    st.caption(
        f"La prioridad y la explicación usan reglas de demostración {provider.descriptor()['rule_version']}; "
        "no constituyen una clasificación oficial de peligro ni un permiso de construcción."
    )
    st.markdown("### Descargar el análisis")
    download_cols = st.columns(3)
    basename = (
        f"{active_area}_{'deteccion_' if detection else ''}semaforo_rojo_{red_band_m:g}m_amarillo_hasta_{band_m:g}m"
    )
    with download_cols[0]:
        st.download_button(
            "Descargar tabla CSV",
            data=csv_bytes(summary),
            file_name=f"{basename}.csv",
            mime="text/csv",
            width="stretch",
        )
    with download_cols[1]:
        st.download_button(
            "Descargar mapa GeoJSON",
            data=geojson_bytes(summary),
            file_name=f"{basename}.geojson",
            mime="application/geo+json",
            width="stretch",
        )
    with download_cols[2]:
        st.download_button(
            "Descargar paquete completo",
            data=analysis_bundle(summary, with_explanations(analysis["relations"]), metadata, band_m),
            file_name=f"{basename}.zip",
            mime="application/zip",
            width="stretch",
        )
    st.caption(
        "CSV: resumen de edificaciones · GeoJSON: geometrías y resultados · ZIP: resumen, relaciones, parámetros y procedencia."
    )
