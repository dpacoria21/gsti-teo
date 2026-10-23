"""Collect the user's inputs without running the detector or spatial analysis."""

import streamlit as st

from src.application.images import bundled_image, stored_upload
from src.application.session_workspace import selected_result, session_identifier, session_output_root
from src.settings import PROJECT_ROOT as ROOT
from src.ui.engine import ml_status
from src.ui.models import SidebarSelection


def render_sidebar(areas: list[dict]) -> SidebarSelection:
    session_id = session_identifier(st.session_state)
    with st.sidebar:
        st.markdown(
            '<div class="eyebrow">OBSERVATORIO TERRITORIAL</div><div class="brand">Chullo</div><div class="brand-sub">Edificaciones y torrenteras<br>Arequipa, Perú</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Preparar análisis")
        area_lookup = {area["id"]: area for area in areas}
        area_id = st.selectbox(
            "Área de estudio", list(area_lookup), format_func=lambda key: area_lookup[key].get("name", key)
        )
        area_name = (
            area_lookup[area_id].get("short_name") or area_lookup[area_id].get("name", area_id).split("·")[0].strip()
        )
        included_raster, included_manifest, included_record = bundled_image(area_id)
        availability_key = f"bundled_available_{area_id}"
        previous_availability = st.session_state.get(availability_key)
        if included_raster is not None and (
            previous_availability is False or (area_id == "chullo" and previous_availability is None)
        ):
            st.session_state[f"mode_{area_id}"] = "Detección IA desde imagen"
        st.session_state[availability_key] = included_raster is not None
        building_mode = st.radio(
            "Origen de las edificaciones",
            ["Detección IA desde imagen", "Inventario Microsoft"],
            index=0 if included_raster else 1,
            key=f"mode_{area_id}",
            help="La detección ejecuta un modelo preentrenado. El inventario contiene contornos ya calculados por otra entidad.",
        )
        use_detection = building_mode == "Detección IA desde imagen"
        detect_clicked = False
        threshold = 0.50
        raster_path = image_manifest_path = None
        image_record = {}
        input_id = f"{area_id}:bundled"
        detection_dir = ROOT / "outputs" / "inference" / f"{area_id}_demo"
        if use_detection:
            image_choice = st.selectbox(
                "Imagen de entrada",
                [f"{area_name} · imagen incluida", f"Subir GeoTIFF de {area_name}"],
                key=f"image_choice_{area_id}",
            )
            if image_choice == f"{area_name} · imagen incluida":
                raster_path, image_manifest_path, image_record = included_raster, included_manifest, included_record
                if raster_path is None:
                    st.caption("La imagen incluida todavía no está disponible en este equipo.")
                else:
                    st.caption(
                        f"{image_record.get('title', 'Imagen georreferenciada del sector')} · Fecha: {image_record.get('observed_at') or image_record.get('capture_date') or 'No informada'}"
                    )
            else:
                uploaded = st.file_uploader(
                    "Imagen RGB georreferenciada",
                    type=["tif", "tiff"],
                    key=f"upload_{area_id}",
                    help=f"GeoTIFF RGB de alta resolución, con coordenadas y cobertura del sector piloto de {area_name}. Una captura JPG/PNG sin coordenadas no es suficiente.",
                )
                if uploaded is not None:
                    raster_path, image_manifest_path, image_record, detection_dir = stored_upload(
                        uploaded.getvalue(), uploaded.name, area_id, session_id=session_id
                    )
                    input_id = f"{area_id}:upload:{image_record['sha256']}"
                else:
                    input_id = f"{area_id}:awaiting_upload"
                    detection_dir = session_output_root(session_id) / f"{area_id}_awaiting_upload"
            detection_dir = selected_result(st.session_state, input_id, detection_dir)
            threshold = st.slider(
                "Puntuación mínima del detector",
                min_value=0.20,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Una puntuación más baja admite más candidatos y puede aumentar los falsos positivos. No mide peligro ni garantiza exactitud.",
            )
            engine = ml_status()
            detect_clicked = st.button(
                "Detectar en esta imagen",
                type="primary",
                width="stretch",
                disabled=raster_path is None or not engine.get("available", False),
            )
            if not engine.get("available", False):
                st.caption(engine.get("reason", "El motor de IA no está disponible."))
            st.caption(
                "Modelo preentrenado de segmentación de edificios. La ejecución puede tardar varios minutos. "
                "Los nuevos resultados se mantienen separados por sesión; descárgalos antes de cerrar la aplicación."
            )
            st.divider()
        st.markdown("### Semáforo de proximidad")
        maximum_band = int(area_lookup[area_id].get("max_band_m", 100))
        if maximum_band < 2:
            st.error("La cobertura del sector debe admitir al menos 2 metros para configurar ambas zonas.")
            st.stop()
        chosen_red_band = st.slider(
            "Límite de la zona roja (m)",
            min_value=1,
            max_value=maximum_band - 1,
            value=min(25, maximum_band - 1),
            step=1,
            key=f"red_band_{area_id}",
            help="Distancia desde el contorno al cauce más cercano. Hasta este límite la edificación se colorea de rojo.",
        )
        maximum_width = maximum_band - chosen_red_band
        width_key = f"yellow_width_{area_id}"
        if width_key in st.session_state:
            st.session_state[width_key] = max(1, min(int(st.session_state[width_key]), maximum_width))
        yellow_width = (
            st.slider(
                "Ancho de la zona amarilla (m)",
                min_value=1,
                max_value=maximum_width,
                value=min(25, maximum_width),
                step=1,
                key=width_key,
                disabled=maximum_width == 1,
                help="Metros adicionales a partir del límite rojo, a cada lado del cauce. El exterior es rojo + ancho amarillo.",
            )
            if maximum_width > 1
            else 1
        )
        if maximum_width == 1:
            st.caption("Ancho de la zona amarilla: 1 m (máximo disponible en este sector).")
        chosen_band = float(chosen_red_band + yellow_width)
        st.caption(
            f"Rojo: hasta {chosen_red_band:g} m · Amarillo: más de {chosen_red_band:g} y hasta {chosen_band:g} m · "
            f"Verde: más de {chosen_band:g} m."
        )
        st.caption("Los colores se actualizan al mover los controles; no es necesario volver a ejecutar el detector.")
        st.caption("Distancia mínima desde el contorno de cada edificación al eje cartografiado de la torrentera.")
        analyze = st.button("Analizar sector", type="primary", width="stretch")
        st.divider()
        st.markdown("### Visualización")
        basemap = st.toggle(
            "Mostrar mapa base",
            value=True,
            help="OpenStreetMap. El fondo requiere acceso a Internet; el cálculo usa los archivos locales.",
        )
        show_imagery = st.toggle("Mostrar imagen analizada", value=True) if use_detection else False
        st.caption("Abre el control de capas del mapa para mostrar u ocultar cada capa.")
        st.markdown(
            '<div class="note"><strong>Cómo leer el semáforo</strong><br>Los colores indican proximidad. Verde no significa que sea seguro ni que se pueda construir. El peligro, la faja marginal y la autorización de construcción no están evaluados.</div>',
            unsafe_allow_html=True,
        )

    return SidebarSelection(
        area_id=area_id,
        area_name=area_name,
        area_lookup=area_lookup,
        building_mode=building_mode,
        included_raster=included_raster,
        raster_path=raster_path,
        image_manifest_path=image_manifest_path,
        image_record=image_record,
        detection_dir=detection_dir,
        threshold=threshold,
        detect_clicked=detect_clicked,
        chosen_band=chosen_band,
        chosen_red_band=float(chosen_red_band),
        analyze=analyze,
        basemap=basemap,
        show_imagery=show_imagery,
        session_id=session_id,
        input_id=input_id,
    )
