"""Render provenance notices and the overview metrics."""

import streamlit as st

from src.application.provenance import approximate_georeferencing
from src.ui.formatting import number
from src.ui.models import AnalysisView, SidebarSelection


def render_overview(selection: SidebarSelection, view: AnalysisView) -> None:
    detection = view.detection
    detection_failed = view.detection_failed
    summary = view.summary
    threshold = selection.threshold
    band_m = view.band_m
    red_band_m = view.red_band_m
    area_config = selection.area_config
    active_area = selection.area_id
    if detection:
        model_name = view.description.model_name
        image_info = view.description.image_info
        observed_at = view.description.observed_at
        applied_threshold = view.description.applied_threshold
        result_label = "Última detección completa conservada" if detection_failed else "Detección real guardada"
        st.success(f"{result_label} · {model_name} · {number(len(summary))} contornos seleccionados en el sector")
        st.caption(
            "Detecciones automáticas provisionales: pueden existir omisiones y falsos positivos. Revisa los contornos sobre la imagen antes de interpretar las distancias."
        )
        model_info = detection["manifest"].get("model", {})
        training_note = (
            "Modelo ajustado con SpaceNet (Jartum), sin etiquetas de entrenamiento de Chullo."
            if model_info.get("trained_locally")
            else "No se ha entrenado el modelo con estas edificaciones; se aplican pesos preentrenados."
        )
        st.caption(f"Captura de imagen: {observed_at} · {training_note}")
        if model_info.get("trained_locally") and active_area == "chullo":
            st.warning(
                "Modelo experimental: en la revisión visual de Chullo se observaron varias cubiertas y patios "
                "unidos en un solo contorno. Este resultado no permite contar viviendas individuales; "
                "compara con el modelo original y revisa los polígonos."
            )
        if model_info.get("variant", "usa") != selection.model_variant:
            st.info(
                "Los contornos visibles corresponden al otro modelo. Pulsa «Detectar en esta imagen» para aplicar el modelo seleccionado."
            )
        st.caption(
            f"{image_info.get('title') or 'Imagen georreferenciada del sector'} · {image_info.get('attribution') or 'Créditos en Fuentes y alcance'}"
        )
        if approximate_georeferencing(image_info):
            capture_source = (
                " de una captura de Google Earth Pro" if "google" in str(image_info).lower() else " de la imagen"
            )
            st.warning(
                f"Georreferenciación aproximada{capture_source}. Las distancias indican proximidad al eje cartografiado; no evalúan peligro ni sustituyen mediciones de campo."
            )
        if applied_threshold is not None:
            st.caption(
                f"Puntuación mínima aplicada: {float(applied_threshold):.2f}. Cambiarla requiere volver a pulsar «Detectar en esta imagen»; ajustar el semáforo reclasifica las distancias sin repetir la IA."
            )
            if abs(float(applied_threshold) - threshold) > 1e-6:
                st.info(
                    f"Los resultados guardados usan una puntuación mínima de {float(applied_threshold):.2f}. Pulsa «Detectar en esta imagen» para ejecutar con {threshold:.2f}."
                )
        if detection.get("raw_count", len(summary)) != len(summary):
            st.caption(
                f"El modelo produjo {number(detection['raw_count'])} contornos en toda la imagen. Se analizan {number(len(summary))} que intersectan el sector piloto, conservando sus geometrías completas."
            )
    else:
        st.info(
            "Modo inventario: se utilizan contornos existentes de Microsoft. Este análisis no ejecuta detección sobre una imagen."
        )

    metrics = st.columns(4)
    metrics[0].metric("Contornos detectados" if detection else "Edificaciones analizadas", number(len(summary)))
    metrics[1].metric(f"🔴 Rojo · 0–{red_band_m:g} m", number(view.color_count("red")))
    metrics[2].metric(f"🟡 Amarillo · >{red_band_m:g}–{band_m:g} m", number(view.color_count("yellow")))
    metrics[3].metric(f"🟢 Verde · >{band_m:g} m", number(view.color_count("green")))
    st.caption(
        f"{area_config.get('name', active_area)} · Límite rojo: {red_band_m:g} m · "
        f"Ancho amarillo: {view.yellow_width_m:g} m · Distancias mínimas desde el contorno al eje de la torrentera"
    )
    unknown_count = len(summary) - view.evaluated_count
    if unknown_count:
        st.caption(
            f"Sin evaluación: {number(unknown_count)} contornos; aparecen en gris y no reciben un color del semáforo."
        )
    st.info(
        "Semáforo de proximidad con límites ajustables. No determina peligro ni permiso para construir; "
        "verde no implica seguridad. Para esas decisiones se requieren estudios y delimitaciones oficiales."
    )
