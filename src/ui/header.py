"""Page heading and area-specific data availability notices."""

import html
import subprocess

import streamlit as st

from src.application.images import bundled_image, imagery_download_available, retry_chullo_imagery
from src.settings import PROJECT_ROOT as ROOT
from src.ui.models import SidebarSelection


def render_header(selection: SidebarSelection) -> None:
    area_name = selection.area_name
    active_area = selection.area_id
    use_detection = selection.use_detection
    included_raster = selection.included_raster
    area_lookup = selection.area_lookup
    st.markdown(
        f'<div class="eyebrow">{html.escape(area_name.upper())} / AREQUIPA · PROTOTIPO DE INVESTIGACIÓN</div>',
        unsafe_allow_html=True,
    )
    st.title("Edificaciones junto a la torrentera")
    st.markdown(
        '<div class="lede">Detecta contornos en una imagen georreferenciada, mide su proximidad al cauce y descarga resultados con su procedencia.</div>'
        if use_detection
        else '<div class="lede">Explora contornos de un inventario existente, consulta su proximidad al cauce y descarga resultados con fuentes identificadas.</div>',
        unsafe_allow_html=True,
    )
    if active_area == "mariano_melgar":
        st.warning(
            "Demostración alternativa en Mariano Melgar, Arequipa: ortofoto de dron del 5 de junio de 2018, no imagen satelital. Esta imagen y sus resultados no corresponden a Chullo ni describen el estado actual de las edificaciones."
        )
    elif active_area == "chullo" and included_raster is None:
        fallback_notice = (
            " La demostración con imagen real está disponible por separado en Mariano Melgar."
            if "mariano_melgar" in area_lookup
            else ""
        )
        original_image_exists = (ROOT / "data/imagery/chullo/chullo_google_earth_original.jpg").exists()
        missing_image_notice = (
            "La captura real de Chullo está disponible; falta terminar su georreferenciación para usarla en el detector."
            if original_image_exists
            else "Chullo conserva su análisis con inventario Microsoft. Aún falta una imagen de este sector utilizable para inferencia."
        )
        st.info(missing_image_notice + fallback_notice)
        with st.expander("Imagen de Chullo: descarga o archivo propio"):
            st.write(
                "El servicio del IGN no respondió durante la preparación. Puedes reintentar la descarga o aportar un GeoTIFF propio, con coordenadas y resolución suficiente para distinguir edificaciones."
            )
            st.markdown(
                "[Servicio de ortofotos de Arequipa del IGN](https://www.idep.gob.pe/geoportal/rest/services/IMAGENES/ORTOFOTO_AREQUIPA/MapServer)"
            )
            retry_image = st.button("Reintentar descarga del IGN", disabled=not imagery_download_available())
            if retry_image:
                try:
                    with st.spinner("Consultando el IGN; espera máxima de 45 segundos…"):
                        downloaded = retry_chullo_imagery()
                    if downloaded.returncode == 0 and bundled_image("chullo")[0]:
                        st.rerun()
                    st.warning(
                        "El servicio no devolvió una imagen utilizable. El inventario de Chullo y la demostración de Mariano Melgar siguen disponibles."
                    )
                    st.caption((downloaded.stderr or downloaded.stdout)[-1500:])
                except subprocess.TimeoutExpired:
                    st.warning(
                        "El IGN no respondió dentro de 45 segundos. Puedes probar más tarde o subir una imagen propia; no se han sustituido los datos de Chullo."
                    )
                except OSError as error:
                    st.error(f"No se pudo iniciar la descarga: {error}")
            st.caption(
                "El reintento depende de la disponibilidad del servicio oficial. No se ejecuta de forma automática ni periódica."
            )
