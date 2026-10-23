"""Source attribution, inference manifests, and georeferencing evidence."""

import json
from pathlib import Path

import streamlit as st

from src.application.provenance import image_details, source_records
from src.model_catalog import model_presentation
from src.settings import PROJECT_ROOT as ROOT
from src.ui.models import AnalysisView, SidebarSelection


def georeferencing_evidence(image):
    """Make the georeferencing record inspectable without exposing arbitrary files."""
    details = image_details(image)
    evidence = details.get("georeferencing")
    if not evidence:
        return
    with st.expander("Georreferenciación y precisión de la imagen"):
        if isinstance(evidence, dict):
            st.json(evidence)
            references = [
                (label, evidence.get(field))
                for label, field in (
                    ("Informe original", "report_path"),
                    ("Archivo de georreferenciación", "path"),
                    ("Registro original", "url"),
                    ("Residuos de georreferenciación", "residuals_path"),
                    ("Puntos de control", "control_points_path"),
                )
            ]
        else:
            st.write(str(evidence))
            references = [("Registro original", evidence if isinstance(evidence, str) else None)]
        st.download_button(
            "Descargar registro de georreferenciación",
            data=json.dumps(evidence, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="georreferenciacion_imagen.json",
            mime="application/json",
            key="download_georeferencing",
        )
        for index, (label, reference) in enumerate(references):
            if isinstance(reference, str) and reference.startswith(("https://", "http://")):
                st.markdown(f"[{label}]({reference})")
            elif isinstance(reference, str) and len(reference) < 300:
                candidate = Path(reference)
                candidate = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
                if (
                    candidate.is_relative_to(ROOT)
                    and candidate.is_file()
                    and candidate.suffix.lower() in {".json", ".csv", ".txt"}
                ):
                    st.download_button(
                        f"Descargar: {label.lower()}",
                        data=candidate.read_bytes(),
                        file_name=candidate.name,
                        key=f"download_georeferencing_file_{index}",
                    )


def render_sources_panel(selection: SidebarSelection, view: AnalysisView) -> None:
    detection = view.detection
    metadata = view.metadata
    area_config = selection.area_config
    if detection:
        manifest = detection["manifest"]
        model_name = view.description.model_name
        image_info = view.description.image_info
        observed_at = view.description.observed_at
        st.markdown("### Datos que sostienen el mapa")
        if detection:
            st.write(
                "Estos contornos son predicciones de un modelo preentrenado ejecutado sobre los píxeles de la imagen indicada. Se conservan los que intersectan el sector piloto y se calcula la distancia mínima de cada contorno a los cauces de ese sector. El inventario Microsoft no interviene en esta detección."
            )
            st.markdown("### Imagen y ejecución del modelo")
            provenance_left, provenance_right = st.columns([1, 1])
            with provenance_left:
                st.write(f"**Modelo:** {model_name}")
                st.write(
                    f"**Imagen:** {image_info.get('title') or image_info.get('image_id') or manifest.get('image_id', 'GeoTIFF de entrada')}"
                )
                st.write(f"**Fecha de captura:** {observed_at}")
                st.write(f"**Tiempo de ejecución:** {float(manifest.get('elapsed_seconds', 0)):.1f} segundos")
                st.write(f"**Ejecución guardada (UTC):** {manifest.get('completed_at', 'Consultar metadatos')}")
                image_source_url = image_info.get("url") or image_info.get("source_url")
                if isinstance(image_source_url, str) and image_source_url.startswith(("https://", "http://")):
                    st.markdown(f"[Fuente original de la imagen]({image_source_url})")
                st.caption(image_info.get("attribution") or "Consultar procedencia de la imagen en los metadatos.")
                st.caption(
                    "Un modelo preentrenado puede omitir techos, unir edificios o marcar superficies que no son construcciones. La puntuación del detector no sustituye una revisión visual local."
                )
                with st.expander("Metadatos completos de la inferencia"):
                    st.json(manifest)
                if manifest.get("model", {}).get("trained_locally"):
                    presentation = model_presentation(manifest["model"])
                    st.write(f"**Ajuste del modelo:** {presentation['title']}")
                    st.caption(presentation["description"])
                    with st.expander("Evaluación y límites del ajuste"):
                        for limitation in presentation["limitations"]:
                            st.caption(limitation)
                        st.json(manifest["model"].get("evaluation_summary", {}))
                    st.caption("Los límites del semáforo son independientes del modelo.")
                georeferencing_evidence(image_info)
            with provenance_right:
                if detection.get("preview_path") and Path(detection["preview_path"]).is_file():
                    st.image(
                        str(detection["preview_path"]),
                        caption="Imagen RGB procesada, sin contornos superpuestos",
                        width="stretch",
                    )
        else:
            st.write(
                "Los contornos de edificios proceden de un inventario ya calculado por Microsoft. Este modo analiza su proximidad a un eje de torrentera obtenido de OpenStreetMap; no ejecuta detección desde una imagen nueva."
            )
        source_list = source_records(metadata)
        for source in source_list:
            if not isinstance(source, dict):
                continue
            title = source.get("title") or source.get("name") or source.get("source_id", "Fuente")
            with st.expander(str(title), expanded=True):
                organization = source.get("organization") or source.get("provider")
                if organization:
                    st.write(f"**Entidad:** {organization}")
                url = source.get("url")
                if isinstance(url, str) and url.startswith(("https://", "http://")):
                    st.markdown(f"[Consultar fuente original]({url})")
                st.write(
                    f"**Fecha del dato o imagen:** {source.get('observed_at') or 'No especificada para cada objeto'}"
                )
                st.write(f"**Descarga:** {source.get('downloaded_at') or 'Consultar metadatos del paquete'}")
                st.write(f"**Licencia / condiciones:** {source.get('license') or 'Consultar fuente original'}")
                if source.get("review_status"):
                    st.write(f"**Estado de revisión:** {source['review_status']}")
                for key in ["notes", "limitations"]:
                    if source.get(key):
                        value = source[key]
                        st.caption(" · ".join(str(item) for item in value) if isinstance(value, list) else str(value))
        st.markdown("### Alcance de esta primera versión")
        st.markdown("""
    - **Qué se calcula:** la menor distancia entre cada contorno de edificación y los cauces incluidos. El semáforo usa la distancia sin redondear: rojo hasta el primer límite, amarillo hasta ese límite más el ancho amarillo, y verde más allá.
    - **Qué significan los colores:** proximidad configurable, con límites elegidos para explorar el sector. Verde no significa seguridad ni autorización para construir; rojo tampoco demuestra por sí solo peligro real.
    - **Qué queda sin evaluar:** peligro, faja marginal y autorización de construcción, porque aún no se incorporaron capas oficiales verificadas ni estudios hidráulicos. Tampoco se identifica el uso residencial o la ocupación de cada edificio.
    - **Fechas y exactitud:** la fecha de descarga o publicación no acredita la fecha de captura de la imagen. Los decimales del cálculo no mejoran la exactitud de sus fuentes.
    - **Cobertura:** el análisis se limita al sector preparado y a los cauces disponibles. La arquitectura permite añadir áreas con sus propios datos y coordenadas métricas.
    """)
        st.caption(
            f"Sistema de referencia de cálculo: {area_config.get('analysis_crs', 'EPSG:32719')} · Coordenadas del mapa y GeoJSON: WGS 84 (EPSG:4326)."
        )
