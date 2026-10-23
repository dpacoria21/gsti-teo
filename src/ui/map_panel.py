"""Map and building exploration panel."""

import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.map_view import PROXIMITY_NOTICE, build_map
from src.risk_explanation import DemoRuleProvider
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
    image_overlay = view.image_overlay
    provider = DemoRuleProvider()
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
    color_options = {"Rojo": "red", "Amarillo": "yellow", "Verde": "green", "Sin evaluación": "unknown"}
    visible_labels = st.multiselect(
        "Colores visibles en el mapa", list(color_options), default=list(color_options), key="map_visible_colors",
        help="Filtra los contornos del mapa. Las tablas y descargas conservan el análisis completo.",
    )
    selected_colors = [color_options[label] for label in visible_labels]
    normalized_colors = summary["proximity_class"].fillna("unknown")
    visible_summary = summary.loc[normalized_colors.isin(selected_colors)]
    counts = " · ".join(
        f"{label}: {int(normalized_colors.eq(color).sum())}" for label, color in color_options.items()
        if label in visible_labels
    )
    st.caption(f"{len(visible_summary)} de {len(summary)} edificaciones visibles. {counts}")
    st.caption(
        f"Reglas de demostración {provider.descriptor()['rule_version']}. "
        "Las distancias se calculan sobre las geometrías; las prioridades son criterios ilustrativos del proyecto."
    )
    map_column, detail_column = st.columns([3.65, 1], gap="large")
    with detail_column:
        st.markdown("### Explorar una edificación")
        st.caption("Selecciona un identificador para esta ficha. Al pulsar un contorno, se abre su ficha dentro del mapa.")
        identifiers = sorted(visible_summary["building_id"].astype(str).tolist())
        selection_key = f"building_{active_area}_{'inference_' + detection_dir.name if use_detection else 'inventory'}"
        if st.session_state.get(selection_key, "") not in {"", *identifiers}:
            st.session_state[selection_key] = ""
        selected_id = st.selectbox(
            "Identificador",
            ["", *identifiers],
            format_func=lambda value: value or "Vista general del sector",
            key=selection_key,
        )
        if selected_id:
            selected = summary.loc[summary["building_id"].astype(str) == selected_id].iloc[0]
            distance = selected["distance_m"]
            explanation = provider.explain(distance, red_band_m, band_m, selected.get("proximity_class"))
            classification = explanation.classification
            label = explanation.label
            chip_class = f"chip-{classification}" if classification in ("red", "yellow", "green") else ""
            st.markdown(
                f'<div class="distance">{distance:.1f} m</div>'
                if pd.notna(distance)
                else '<div class="distance">Sin datos</div>',
                unsafe_allow_html=True,
            )
            st.caption("Distancia mínima al eje del cauce")
            st.markdown(f'<span class="chip {chip_class}">{html.escape(str(label))}</span>', unsafe_allow_html=True)
            st.markdown("**¿Por qué este color?**")
            st.write(explanation.criterion)
            st.caption(
                f"Límite rojo: {red_band_m:g} m · Ancho amarillo: {view.yellow_width_m:g} m · "
                f"Límite exterior: {band_m:g} m"
            )
            st.write(explanation.potential_exposure)
            st.caption(explanation.notice)
            with st.expander("Factores pendientes de evaluar"):
                for factor in explanation.pending_factors:
                    st.write(f"• {factor}")
            with st.expander("Qué verificar"):
                for action in explanation.actions:
                    st.write(f"• {action}")
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
            visible_summary,
            payload["rivers"],
            payload["area"],
            analysis["bands"],
            band_m=band_m,
            selected_id=selected_id or None,
            basemap=basemap,
            image_overlay=image_overlay,
            buildings_label="Edificaciones · detector IA" if detection else "Edificaciones · Microsoft",
            traffic_zones=analysis.get("traffic_zones"),
            red_band_m=red_band_m,
            rule_provider=provider,
        )
        # Leaflet owns selection/popups locally. No Python callback is needed;
        # inline HTML also avoids component-loading failures observed locally.
        components.html(map_object.get_root().render(), height=590, scrolling=False)
        st.caption(
            "Datos vectoriales locales · El mapa base y los recursos del visor requieren Internet. Las distancias se calculan en metros sobre las geometrías originales."
        )
