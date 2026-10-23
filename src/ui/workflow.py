"""Orchestrate the UI's progress, cache reads, and prepared analysis view."""

from dataclasses import replace
from pathlib import Path

import streamlit as st

from src.application.images import data_fingerprint, detection_fingerprint
from src.application.provenance import describe_detection, detection_metadata, image_details
from src.application.session_workspace import new_detection_output, remember_result
from src.ui.cache import cached_analysis, cached_detection, cached_detection_analysis, cached_load
from src.ui.engine import load_detection, run_detection
from src.ui.models import AnalysisView, SidebarSelection


def applied_band(selection: SidebarSelection) -> float:
    """Apply spatial thresholds on each rerun; inference remains an explicit action."""
    return selection.chosen_band


def execute_requested_detection(selection: SidebarSelection, band_m: float) -> tuple[SidebarSelection, float, bool]:
    use_detection = selection.use_detection
    detect_clicked = selection.detect_clicked
    raster_path = selection.raster_path
    threshold = selection.threshold
    image_manifest_path = selection.image_manifest_path
    area_id = selection.area_id
    chosen_band = selection.chosen_band
    detection_failed = False
    if use_detection and detect_clicked:
        detection_dir = new_detection_output(selection.session_id)
        with st.status("Ejecutando el detector preentrenado…", expanded=True) as status:
            progress = st.progress(0)
            progress_text = st.empty()

            def on_progress(event):
                if not isinstance(event, dict):
                    return
                message = event.get("message") or event.get("stage") or "Procesando imagen…"
                progress_text.write(str(message))
                value = event.get("progress")
                if isinstance(value, (int, float)):
                    progress.progress(min(1.0, max(0.0, float(value) / 100 if value > 1 else float(value))))

            try:
                run_detection(
                    raster_path,
                    detection_dir,
                    confidence=threshold,
                    image_manifest=image_manifest_path,
                    on_progress=on_progress,
                    area_id=area_id,
                )
                remember_result(st.session_state, selection.input_id, detection_dir)
                selection = replace(selection, detection_dir=detection_dir)
                progress.progress(1.0)
                status.update(label="Detección completada · resultados guardados", state="complete", expanded=False)
                band_m = chosen_band
            except Exception as error:
                detection_failed = True
                status.update(label="No se pudo completar la detección", state="error", expanded=True)
                st.error(str(error))

    return selection, band_m, detection_failed


def prepare_analysis(selection: SidebarSelection, band_m: float, detection_failed: bool) -> AnalysisView:
    active_area = selection.area_id
    area_name = selection.area_name
    area_config = selection.area_config
    use_detection = selection.use_detection
    detection_dir = selection.detection_dir
    show_imagery = selection.show_imagery
    detection = None
    image_overlay = None
    detection_version = ()
    try:
        fingerprint = data_fingerprint()
        with st.spinner("Leyendo datos y calculando distancias…"):
            payload = cached_load(active_area, fingerprint)
            if use_detection:
                detection_manifest = detection_dir / "manifest.json"
                if not detection_manifest.exists() or load_detection is None:
                    st.info(
                        f"Selecciona la imagen incluida o sube un GeoTIFF de {area_name} y pulsa «Detectar en esta imagen». Los contornos se producirán a partir de sus píxeles."
                    )
                    st.caption(
                        "El análisis espacial utiliza los cauces del sector seleccionado. Para analizar otra localidad hace falta incorporar su cartografía de cauces."
                    )
                    st.stop()
                detection_version = detection_fingerprint(detection_dir)
                detection = cached_detection(str(detection_dir), detection_version, active_area)
                manifest = detection["manifest"]
                preview_path = detection.get("preview_path")
                preview_bounds = detection.get("preview_bounds") or manifest.get("preview_bounds_wgs84")
                if show_imagery and preview_path and preview_bounds and Path(preview_path).is_file():
                    image_overlay = {
                        "path": preview_path,
                        "bounds": preview_bounds,
                        "name": "Imagen RGB analizada",
                        "attribution": image_details(manifest.get("image", {})).get(
                            "attribution", "Imagen analizada · consultar fuentes"
                        ),
                    }
                if not len(detection["buildings"]):
                    st.warning(
                        "El detector no produjo contornos dentro del sector de estudio con estos parámetros. Esto no demuestra que no existan edificaciones."
                    )
                    if preview_path and Path(preview_path).is_file():
                        st.image(str(preview_path), caption="Imagen que se ha procesado", width="stretch")
                    st.caption("Prueba un umbral menor o revisa la resolución, el encuadre y la calidad de la imagen.")
                    st.stop()
                analysis = cached_detection_analysis(
                    str(detection_dir),
                    detection_version,
                    active_area,
                    fingerprint,
                    float(band_m),
                    area_config.get("analysis_crs", "EPSG:32719"),
                    red_band_m=selection.chosen_red_band,
                )
            else:
                analysis = cached_analysis(
                    active_area,
                    fingerprint,
                    float(band_m),
                    area_config.get("analysis_crs", "EPSG:32719"),
                    red_band_m=selection.chosen_red_band,
                )
        metadata = (
            detection_metadata(payload.get("metadata", {}), detection) if detection else payload.get("metadata", {})
        )
    except Exception as error:
        st.title("Análisis de proximidad")
        st.error(f"No se pudo preparar el análisis: {error}")
        st.info(
            "Revisa que los archivos locales del área estén completos y que sus geometrías y coordenadas sean válidas."
        )
        st.stop()

    return AnalysisView(
        payload=payload,
        analysis=analysis,
        metadata=metadata,
        detection=detection,
        image_overlay=image_overlay,
        band_m=band_m,
        detection_version=detection_version,
        detection_failed=detection_failed,
        description=describe_detection(detection["manifest"]) if detection else None,
        red_band_m=selection.chosen_red_band,
    )
