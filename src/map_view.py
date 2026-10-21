"""Folium presentation of the local proximity analysis (no spatial calculations)."""

from __future__ import annotations

import json

import folium

COLORS = {"river": "#07838a", "inside": "#d79024", "outside": "#66788b", "selected": "#182d43"}


def build_map(
    summary,
    rivers,
    area,
    bands,
    band_m=25,
    selected_id=None,
    basemap=True,
    image_overlay=None,
    buildings_label="Edificaciones · Microsoft",
):
    """Build a map in WGS84 from already evaluated, full-resolution geometries."""
    bounds = area.total_bounds if len(area) else summary.total_bounds
    west, south, east, north = (float(value) for value in bounds)
    map_view = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        tiles=None,
        zoom_start=16,
        control_scale=True,
        prefer_canvas=True,
        zoom_control=True,
        attribution_control=True,
    )
    if basemap:
        folium.TileLayer("OpenStreetMap", name="Mapa base · OpenStreetMap", control=True).add_to(map_view)
        map_view.get_root().header.add_child(
            folium.Element(
                "<style>.leaflet-tile-pane {filter: grayscale(.9) saturate(.25) contrast(.85) brightness(1.09);}</style>"
            )
        )

    if image_overlay:
        folium.raster_layers.ImageOverlay(
            image=str(image_overlay["path"]),
            bounds=image_overlay["bounds"],
            name=image_overlay.get("name", "Imagen analizada"),
            opacity=1,
            interactive=False,
            cross_origin=False,
            zindex=1,
            attribution=image_overlay.get("attribution", "Imagen analizada · consultar fuentes"),
        ).add_to(map_view)

    if len(area):
        folium.GeoJson(
            json.loads(area[["geometry"]].to_json()),
            name="Área de estudio",
            style_function=lambda _: {"color": "#7c8b97", "weight": 1.2, "dashArray": "5 6", "fillOpacity": 0},
            interactive=False,
        ).add_to(map_view)

    if len(bands):
        folium.GeoJson(
            json.loads(bands[["geometry"]].to_json()),
            name=f"Banda analítica · {band_m:g} m",
            style_function=lambda _: {"color": "#ddb267", "weight": 0.6, "fillColor": "#f0bb62", "fillOpacity": 0.22},
            interactive=False,
        ).add_to(map_view)

    if len(summary):
        columns = [
            "building_id",
            "distance_m",
            "within_band",
            "nearest_river_name",
            "distance_reference",
            "confidence",
            "geometry",
        ]
        display = summary[[column for column in columns if column in summary.columns]].copy()
        display["Distancia"] = display["distance_m"].map(
            lambda value: f"{value:.1f} m" if value is not None and value == value else "Sin evaluación"
        )
        display["Resultado"] = (
            display["within_band"]
            .map({True: "Dentro de la banda", False: "Fuera de la banda"})
            .fillna("Sin evaluación")
        )
        display["Referencia"] = display.get("distance_reference", "axis").map(_reference_label)
        popup_fields = ["building_id", "Distancia", "Resultado", "Referencia"]
        popup_aliases = ["Edificación", "Distancia mínima", "Banda analítica", "Medición"]
        if "confidence" in display.columns:
            display["Puntuación IA"] = display["confidence"].map(
                lambda value: f"{float(value):.3f}" if value is not None and value == value else "No disponible"
            )
            popup_fields.append("Puntuación IA")
            popup_aliases.append("Puntuación del detector")

        def building_style(feature):
            within = feature["properties"].get("within_band")
            color = COLORS["inside"] if within is True else COLORS["outside"]
            return {
                "color": color,
                "weight": 1.5 if image_overlay else 0.75,
                "fillColor": color,
                "fillOpacity": 0.22 if image_overlay else 0.68 if within is True else 0.38,
            }

        folium.GeoJson(
            json.loads(display.to_json()),
            name=buildings_label,
            style_function=building_style,
            highlight_function=lambda _: {"weight": 2.5, "color": COLORS["selected"], "fillOpacity": 0.85},
            tooltip=folium.GeoJsonTooltip(
                fields=["building_id", "Distancia", "Resultado"],
                aliases=["Edificación", "Distancia", "Resultado"],
                sticky=False,
                localize=False,
            ),
            popup=folium.GeoJsonPopup(fields=popup_fields, aliases=popup_aliases, max_width=330, localize=False),
            smooth_factor=0,
        ).add_to(map_view)

    if len(rivers):
        fields = [field for field in ["name", "river_id"] if field in rivers.columns]
        folium.GeoJson(
            json.loads(rivers[[*fields, "geometry"]].to_json()),
            name="Torrentera · eje cartografiado",
            style_function=lambda _: {"color": COLORS["river"], "weight": 3.5, "opacity": 0.95},
            tooltip=folium.GeoJsonTooltip(
                fields=fields, aliases=["Cauce" if field == "name" else "Identificador" for field in fields]
            )
            if fields
            else None,
        ).add_to(map_view)

    if selected_id:
        selected = summary.loc[summary["building_id"].astype(str) == str(selected_id)]
        if len(selected):
            folium.GeoJson(
                json.loads(selected[["geometry"]].to_json()),
                name="Edificación seleccionada",
                style_function=lambda _: {"color": COLORS["selected"], "weight": 3, "fillOpacity": 0.12},
                control=False,
                interactive=False,
            ).add_to(map_view)
            box = selected.total_bounds
            map_view.fit_bounds(
                [[float(box[1]) - 0.00045, float(box[0]) - 0.00045], [float(box[3]) + 0.00045, float(box[2]) + 0.00045]]
            )
        else:
            map_view.fit_bounds([[south, west], [north, east]])
    else:
        map_view.fit_bounds([[south, west], [north, east]], padding=(24, 24))

    folium.LayerControl(position="topright", collapsed=True).add_to(map_view)
    return map_view


def _reference_label(value):
    if value in ("axis", "river_axis", "eje", "eje_del_cauce"):
        return "Desde el contorno al eje del cauce"
    if value in ("channel_polygon", "channel_boundary", "polygon", "borde"):
        return "Entre la edificación y el polígono del cauce"
    return str(value)
