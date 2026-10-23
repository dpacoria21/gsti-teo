"""Folium presentation of the local proximity analysis (no spatial calculations)."""

from __future__ import annotations

import html
import json
import math

import folium
from branca.element import MacroElement, Template

from src.risk_explanation import DemoRuleProvider, ExplanationProvider, ProximityExplanation

COLORS = {
    "river": "#07838a",
    "red": "#dc2626",
    "yellow": "#eab308",
    "green": "#16a34a",
    "unknown": "#66788b",
    "selected": "#182d43",
}
PROXIMITY_NOTICE = "No determina peligro ni permiso para construir; verde no implica seguridad."


def build_map(
    summary,
    rivers,
    area,
    bands,
    band_m=50,
    selected_id=None,
    basemap=True,
    image_overlay=None,
    buildings_label="Edificaciones · Microsoft",
    traffic_zones=None,
    red_band_m=25,
    rule_provider: ExplanationProvider | None = None,
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

    if traffic_zones is not None and len(traffic_zones):
        for color, name in (
            ("red", f"Zona roja · 0 a {red_band_m:g} m"),
            ("yellow", f"Zona amarilla · más de {red_band_m:g} a {band_m:g} m"),
        ):
            zone = traffic_zones.loc[traffic_zones["proximity_class"].eq(color)]
            if len(zone):
                folium.GeoJson(
                    json.loads(zone[["geometry"]].to_json()),
                    name=name,
                    style_function=lambda _, zone_color=COLORS[color]: {
                        "color": zone_color,
                        "weight": 0.8,
                        "fillColor": zone_color,
                        "fillOpacity": 0.18,
                    },
                    interactive=False,
                ).add_to(map_view)
    elif len(bands):
        folium.GeoJson(
            json.loads(bands[["geometry"]].to_json()),
            name=f"Límite exterior amarillo · {band_m:g} m",
            style_function=lambda _: {
                "color": COLORS["yellow"],
                "weight": 0.6,
                "fillColor": COLORS["yellow"],
                "fillOpacity": 0.18,
            },
            interactive=False,
        ).add_to(map_view)

    building_layers = []
    if len(summary):
        provider = rule_provider or DemoRuleProvider()
        columns = [
            "building_id",
            "distance_m",
            "within_band",
            "proximity_class",
            "proximity_label",
            "nearest_river_name",
            "distance_reference",
            "confidence",
            "geometry",
        ]
        display = summary[[column for column in columns if column in summary.columns]].copy()
        explanations = [
            provider.explain(row["distance_m"], red_band_m, band_m, row.get("proximity_class"))
            for _, row in display.iterrows()
        ]
        display["proximity_class"] = [item.classification for item in explanations]
        display["Resultado"] = [item.label for item in explanations]
        display["Distancia"] = [
            f"{item.distance_m:.2f} m" if item.distance_m is not None else "Sin evaluación" for item in explanations
        ]
        display["Ficha"] = [
            _building_popup(row, explanation) for (_, row), explanation in zip(display.iterrows(), explanations)
        ]
        # The tooltip inserts text into HTML; retain the raw ID separately for selection.
        display["Identificador"] = display["building_id"].astype(str).map(html.escape)

        def building_style(feature):
            classification = feature["properties"].get("proximity_class")
            color = COLORS.get(classification, COLORS["unknown"])
            return {
                "color": color,
                "weight": 1.5 if image_overlay else 0.75,
                "fillColor": color,
                "fillOpacity": 0.24 if image_overlay else 0.58,
            }

        for color, label in (("red", "Rojo"), ("yellow", "Amarillo"), ("green", "Verde"), ("unknown", "Sin evaluación")):
            subset = display.loc[display["proximity_class"].eq(color)]
            if subset.empty:
                continue
            layer = folium.GeoJson(
                json.loads(subset.to_json()),
                name=f"{label} · {len(subset)} · {buildings_label}",
                style_function=building_style,
                highlight_function=lambda _: {"weight": 2.5, "color": COLORS["selected"], "fillOpacity": 0.65},
                tooltip=folium.GeoJsonTooltip(
                    fields=["Identificador", "Distancia", "Resultado"],
                    aliases=["Edificación", "Distancia", "Semáforo demo"],
                    sticky=False, localize=False,
                ),
                popup=folium.GeoJsonPopup(fields=["Ficha"], labels=False, max_width=410, localize=False),
                smooth_factor=0,
            ).add_to(map_view)
            building_layers.append({"variable": layer.get_name(), "count": len(subset), "label": label})

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
    _BuildingInteractions(building_layers).add_to(map_view)
    return map_view


def _building_popup(row, explanation: ProximityExplanation) -> str:
    """Self-contained escaped HTML works in Cloud without callbacks to Python."""
    def escape(value):
        return html.escape(str(value))

    color = COLORS[explanation.classification]
    distance = f"{explanation.distance_m:.2f} m" if explanation.distance_m is not None else "Sin evaluación"
    factors = "".join(f"<li>{escape(item)}</li>" for item in explanation.pending_factors)
    actions = "".join(f"<li>{escape(item)}</li>" for item in explanation.actions)
    try:
        confidence = float(row.get("confidence"))
    except (TypeError, ValueError):
        confidence = float("nan")
    confidence_html = (
        f'<div>Puntuación del detector: <b>{confidence:.3f}</b> · no representa probabilidad de peligro.</div>'
        if math.isfinite(confidence) else ""
    )
    return (
        '<section class="building-explanation" style="max-width:360px;max-height:390px;overflow-y:auto;'
        'font:13px/1.45 system-ui,sans-serif;white-space:normal;padding:3px 5px">'
        f'<strong style="color:{color};font-size:15px">{escape(explanation.label)}</strong>'
        f'<div style="color:#526878;margin:5px 0">Edificación {escape(row["building_id"])}</div>'
        f'<div style="font-size:22px;font-weight:700">{distance}</div>'
        f'<div>{escape(_reference_label(row.get("distance_reference", "axis")))}</div>'
        f'<div>Cauce: {escape(row.get("nearest_river_name", "Cauce cartografiado"))}</div>'
        f'{confidence_html}'
        f'<p><strong>¿Por qué este color?</strong><br>{escape(explanation.criterion)} '
        f'{escape(explanation.reason)}</p>'
        f'<div>Límite rojo: <b>{explanation.red_band_m:g} m</b> · '
        f'Ancho amarillo: <b>{explanation.yellow_width_m:g} m</b> · '
        f'Límite exterior: <b>{explanation.outer_band_m:g} m</b></div>'
        f'<p>{escape(explanation.potential_exposure)}</p>'
        f'<details><summary><strong>Factores pendientes de evaluar</strong></summary><ul>{factors}</ul></details>'
        f'<details><summary><strong>Qué verificar</strong></summary><ul>{actions}</ul></details>'
        f'<p style="background:#f1f5f9;padding:8px;border-radius:5px">{escape(explanation.notice)}</p>'
        f'<small>Regla de demostración {escape(explanation.rule_version)} · '
        'criterios ilustrativos del proyecto, no una evaluación oficial.</small></section>'
    )


class _BuildingInteractions(MacroElement):
    """Local Leaflet selection, zoom and live visibility counts within the iframe."""

    _template = Template("""
    {% macro script(this, kwargs) %}
    (function () {
        const map = {{ this._parent.get_name() }};
        const groups = [
            {% for item in this.layers %}
            {layer: {{ item.variable }}, count: {{ item.count }}, label: {{ item.label | tojson }}},
            {% endfor %}
        ];
        let selected = null;
        const counter = L.control({position: 'bottomleft'});
        counter.onAdd = function () {
            this.node = L.DomUtil.create('div', 'building-visibility-counter');
            this.node.style.cssText = 'background:white;padding:8px 10px;border-radius:6px;'
                + 'box-shadow:0 1px 5px #0003;font:12px/1.5 system-ui;max-width:300px;';
            L.DomEvent.disableClickPropagation(this.node);
            return this.node;
        };
        counter.addTo(map);
        function updateCounter() {
            let count = 0;
            const parts = [];
            groups.forEach(function (item) {
                if (map.hasLayer(item.layer)) { count += item.count; parts.push(item.label + ': ' + item.count); }
            });
            counter.node.textContent = count + ' edificaciones visibles · ' + parts.join(' · ');
        }
        groups.forEach(function (item) {
            item.layer.eachLayer(function (feature) {
                feature.on('click', function () {
                    if (selected) { selected.group.resetStyle(selected.feature); }
                    selected = {group: item.layer, feature: feature};
                    feature.setStyle({color: '#182d43', weight: 3.5, fillOpacity: 0.42});
                    map.fitBounds(feature.getBounds(), {padding: [65, 65], maxZoom: 19});
                    feature.openPopup();
                });
                feature.on('mouseout', function () {
                    if (selected && selected.feature === feature) {
                        feature.setStyle({color: '#182d43', weight: 3.5, fillOpacity: 0.42});
                    }
                });
            });
        });
        map.on('overlayadd overlayremove', updateCounter);
        updateCounter();
    })();
    {% endmacro %}
    """)

    def __init__(self, layers):
        super().__init__()
        self._name = "BuildingInteractions"
        self.layers = layers


def _reference_label(value):
    if value in ("axis", "river_axis", "eje", "eje_del_cauce"):
        return "Desde el contorno al eje del cauce"
    if value in ("channel_polygon", "channel_boundary", "polygon", "borde"):
        return "Entre la edificación y el polígono del cauce"
    return str(value)
