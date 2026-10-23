"""Map and building exploration panel."""

import hashlib
import html
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium

from src.map_view import PROXIMITY_NOTICE, build_map
from src.ui.models import AnalysisView, SidebarSelection


def render_map_panel(selection: SidebarSelection, view: AnalysisView) -> None:
    summary = view.summary
    active_area = selection.area_id
    use_detection = selection.use_detection
    detection_dir = selection.detection_dir
    detection = view.detection
    model_name = view.description.model_name if view.description else ""
    inside = view.inside_count
    evaluated = view.evaluated_count
    band_m = view.band_m
    red_band_m = view.red_band_m
    payload = view.payload
    analysis = view.analysis
    basemap = selection.basemap
    show_imagery = selection.show_imagery
    image_overlay = view.image_overlay
    detection_fingerprint = view.detection_version
    st.markdown("**Semáforo de proximidad**")
    st.markdown(
        '<div class="legend">'
        f'<span><i class="dot" style="background:#dc2626"></i>Rojo: 0 a {red_band_m:g} m</span>'
        f'<span><i class="dot" style="background:#eab308"></i>Amarillo: más de {red_band_m:g} a {band_m:g} m</span>'
        f'<span><i class="dot" style="background:#16a34a"></i>Verde: más de {band_m:g} m</span>'
        '<span><i class="dot" style="background:#66788b"></i>Sin evaluación</span>'
        '<span><i class="dot" style="background:#07838a"></i>Eje de la torrentera</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Ancho amarillo: {view.yellow_width_m:g} m, desde el límite rojo hasta {band_m:g} m del eje. "
        "Las zonas transparentes muestran los dos intervalos sobre el terreno."
    )
    map_column, detail_column = st.columns([3.65, 1], gap="large")
    with detail_column:
        st.markdown("### Explorar una edificación")
        st.caption("Busca su identificador para acercarte. También puedes pulsar cualquier contorno en el mapa.")
        identifiers = sorted(summary["building_id"].astype(str).tolist())
        selection_key = f"building_{active_area}_{'inference_' + detection_dir.name if use_detection else 'inventory'}"
        selected_id = st.selectbox(
            "Identificador",
            ["", *identifiers],
            format_func=lambda value: value or "Vista general del sector",
            key=selection_key,
        )
        if selected_id:
            selected = summary.loc[summary["building_id"].astype(str) == selected_id].iloc[0]
            distance = selected["distance_m"]
            classification = selected.get("proximity_class")
            if not isinstance(classification, str):
                classification = "unknown"
            label = selected.get("proximity_label")
            if pd.isna(label):
                label = "Sin evaluación"
            chip_class = f"chip-{classification}" if classification in ("red", "yellow", "green") else ""
            st.markdown(
                f'<div class="distance">{distance:.1f} m</div>'
                if pd.notna(distance)
                else '<div class="distance">Sin datos</div>',
                unsafe_allow_html=True,
            )
            st.caption("Distancia mínima al eje del cauce")
            st.markdown(f'<span class="chip {chip_class}">{html.escape(str(label))}</span>', unsafe_allow_html=True)
            st.caption(PROXIMITY_NOTICE)
            st.markdown(
                f'<div class="fact">Cauce más cercano</div><div class="fact-value">{html.escape(str(selected.get("nearest_river_name", "Cauce cartografiado")))}</div>',
                unsafe_allow_html=True,
            )
            building_source = (
                f"Inferencia propia · {model_name}"
                if detection
                else str(selected.get("source_id", "Microsoft Building Footprints"))
            )
            st.markdown(
                f'<div class="fact">Fuente de la edificación</div><div class="fact-value">{html.escape(building_source)}</div>',
                unsafe_allow_html=True,
            )
            if detection and pd.notna(selected.get("confidence")):
                st.markdown(
                    f'<div class="fact">Puntuación del detector</div><div class="fact-value">{float(selected["confidence"]):.3f}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Puntuación interna del modelo; no es una probabilidad de peligro ni una exactitud validada en este sector."
                )
            st.markdown(
                '<div class="fact">Zona de peligro / faja marginal</div><div class="fact-value">Sin evaluación: no se incorporaron capas oficiales verificadas.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="note"><strong>Una medida, una referencia.</strong><br>La distancia une el punto más próximo de la edificación con el eje disponible del cauce. No se mide desde el centro del edificio.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Lectura del sector**")
            ratio = 100 * inside / evaluated if evaluated else 0
            st.write(
                f"El {ratio:.1f} % de las edificaciones evaluadas está en rojo o amarillo: "
                f"a {band_m:g} m o menos del eje cartografiado."
            )
            st.caption(PROXIMITY_NOTICE)
            st.caption(
                "Los contornos pueden representar viviendas, comercios u otros usos. El conjunto no identifica su ocupación actual."
            )
    with map_column:
        map_object = build_map(
            summary,
            payload["rivers"],
            payload["area"],
            analysis["bands"],
            band_m=band_m,
            selected_id=selected_id or None,
            basemap=basemap,
            image_overlay=image_overlay,
            buildings_label="Edificaciones · detector preentrenado" if detection else "Edificaciones · Microsoft",
            traffic_zones=analysis.get("traffic_zones"),
            red_band_m=red_band_m,
        )
        if os.environ.get("CHULLO_DEPLOYMENT") == "cloud":
            # Embed inline to avoid the observed Cloud component-loading timeout.
            # Leaflet popups and controls need no Python callbacks in this UI.
            components.html(map_object.get_root().render(), height=590, scrolling=False)
        else:
            map_identity = str(detection["manifest"].get("run_id", detection_fingerprint)) if detection else "inventory"
            st_folium(
                map_object,
                height=590,
                use_container_width=True,
                returned_objects=[],
                key=f"map_{active_area}_{red_band_m}_{band_m}_{selected_id}_{basemap}_{show_imagery}_{hashlib.sha256(map_identity.encode()).hexdigest()[:12]}",
            )
        st.caption(
            "Datos vectoriales locales · El mapa base y los recursos del visor requieren Internet. Las distancias se calculan en metros sobre las geometrías originales."
        )
