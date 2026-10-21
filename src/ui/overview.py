"""Render provenance notices and the overview metrics."""

import pandas as pd
import streamlit as st

from src.application.provenance import approximate_georeferencing
from src.ui.formatting import number
from src.ui.models import AnalysisView, SidebarSelection


def render_overview(selection: SidebarSelection, view: AnalysisView) -> None:
    detection = view.detection
    detection_failed = view.detection_failed
    summary = view.summary
    threshold = selection.threshold
    chosen_band = selection.chosen_band
    band_m = view.band_m
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
        st.caption(
            f"Captura de imagen: {observed_at} · No se ha entrenado el modelo con estas edificaciones; se aplican pesos preentrenados."
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
                f"Puntuación mínima aplicada: {float(applied_threshold):.2f}. Cambiarla requiere volver a pulsar «Detectar en esta imagen»; cambiar la banda sólo recalcula distancias."
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

    if chosen_band != band_m:
        st.info(
            f"Se muestran los resultados de {band_m} m. Pulsa «Analizar sector» para aplicar la banda de {chosen_band} m."
        )

    inside = int(summary["within_band"].fillna(False).sum())
    evaluated = int(summary["distance_m"].notna().sum())
    outside = evaluated - inside
    minimum = summary["distance_m"].min() if len(summary) else float("nan")
    metrics = st.columns(4)
    metrics[0].metric("Contornos detectados" if detection else "Edificaciones analizadas", number(evaluated))
    metrics[1].metric(f"Dentro de la banda · {band_m} m", number(inside))
    metrics[2].metric("Fuera de la banda", number(outside))
    metrics[3].metric("Distancia mínima al eje", f"{minimum:.1f} m" if pd.notna(minimum) else "Sin datos")
    st.caption(
        f"{area_config.get('name', active_area)} · Banda aplicada: {band_m} m · {number(len(summary))} contornos en el área de estudio"
    )
