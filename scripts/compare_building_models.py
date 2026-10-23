"""Render two detector runs on identical georeferenced Chullo image extents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/inference/chullo_demo")
    parser.add_argument("--candidate", type=Path, default=ROOT / "outputs/inference/chullo_spacenet2_khartoum_v1_demo")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training_validation/chullo_comparison.png")
    args = parser.parse_args()
    manifests = [
        json.loads((folder / "manifest.json").read_text(encoding="utf-8")) for folder in (args.baseline, args.candidate)
    ]
    if manifests[0]["image_id"] != manifests[1]["image_id"] or manifests[0]["parameters"] != manifests[1]["parameters"]:
        raise ValueError("The comparison requires the same image and inference parameters.")
    with rasterio.open(ROOT / "data/imagery/chullo/chullo_rgb.tif") as raster:
        area = gpd.read_file(ROOT / "data/processed/chullo/study_area.geojson").to_crs(raster.crs)
        rivers = gpd.read_file(ROOT / "data/processed/chullo/rivers.geojson").to_crs(raster.crs)
        sector = area.geometry.union_all()
        buildings = [
            gpd.read_file(folder / "buildings.geojson").to_crs(raster.crs) for folder in (args.baseline, args.candidate)
        ]
        buildings = [layer.loc[layer.intersects(sector)] for layer in buildings]
        river_inside = rivers.geometry.union_all().intersection(sector)
        segments = list(river_inside.geoms) if hasattr(river_inside, "geoms") else [river_inside]
        longest = max(segments, key=lambda segment: segment.length)
        center = longest.interpolate(0.5, normalized=True)
        zoom = (center.x - 90, center.y - 90, center.x + 90, center.y + 90)
        extents = [tuple(area.total_bounds), zoom]
        fig, axes = plt.subplots(2, 2, figsize=(12, 14), gridspec_kw={"height_ratios": [1.45, 1]})
        fig.patch.set_facecolor("#f1f5f9")
        headings = ["Original · GeoAI USA", "Ajustado · SpaceNet 2 Khartoum"]
        for row, extent in enumerate(extents):
            window = from_bounds(*extent, transform=raster.transform).round_offsets().round_lengths()
            rgb = np.moveaxis(raster.read((1, 2, 3), window=window), 0, -1)
            west, south, east, north = window_bounds(window, raster.transform)
            for column, layer in enumerate(buildings):
                axis = axes[row, column]
                axis.imshow(rgb, extent=[west, east, south, north], interpolation="nearest")
                visible = layer.loc[layer.intersects(box(*extent))]
                visible.boundary.plot(ax=axis, color="#a3ff12", linewidth=0.65 if row == 0 else 1.05)
                rivers.plot(ax=axis, color="#00d8ff", linewidth=1.8, alpha=0.95)
                if row == 0:
                    area.boundary.plot(ax=axis, color="white", linewidth=1, linestyle="--")
                    gpd.GeoSeries([box(*zoom)], crs=raster.crs).boundary.plot(
                        ax=axis, color="#f472b6", linewidth=1.4, linestyle="--"
                    )
                axis.set_xlim(extent[0], extent[2])
                axis.set_ylim(extent[1], extent[3])
                axis.set_aspect("equal")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(
                    headings[column] if row == 0 else "Mismo acercamiento junto al cauce", fontsize=12, pad=10
                )
                length = 100 if row == 0 else 25
                start_x = extent[0] + (extent[2] - extent[0]) * 0.06
                start_y = extent[1] + (extent[3] - extent[1]) * 0.045
                axis.plot([start_x, start_x + length], [start_y, start_y], color="white", linewidth=3)
                axis.text(
                    start_x,
                    start_y + (extent[3] - extent[1]) * 0.015,
                    f"{length} m",
                    color="white",
                    fontsize=9,
                    bbox={"facecolor": "#0f172a", "alpha": 0.75, "edgecolor": "none", "pad": 2},
                )
        fig.suptitle("Chullo · comparación visual de contornos", fontsize=19, fontweight="bold", y=0.985)
        fig.text(
            0.5,
            0.955,
            "Misma imagen y parámetros · sector completo arriba · acercamiento de 180 × 180 m abajo",
            ha="center",
            fontsize=10,
            color="#334155",
        )
        legend = [
            Line2D([0], [0], color="#65a30d", lw=2, label="Contornos detectados"),
            Line2D([0], [0], color="#0891b2", lw=2, label="Eje OSM de Chullo"),
            Line2D([0], [0], color="#db2777", lw=1.5, ls="--", label="Acercamiento seleccionado por geometría"),
        ]
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.067), ncol=3, fontsize=9, frameon=False)
        fig.text(
            0.5,
            0.055,
            "El zoom se centra en el punto medio del tramo de cauce más largo dentro del sector; no se eligió por las detecciones.",
            ha="center",
            fontsize=8,
            color="#475569",
        )
        fig.text(
            0.5,
            0.037,
            "Menos o más contornos no implica mayor precisión. No hay etiquetas locales independientes para evaluar esta imagen.",
            ha="center",
            fontsize=9,
            color="#334155",
        )
        fig.text(
            0.5,
            0.018,
            "Imagen: Google Earth · Image © 2026 Airbus · fecha del visor 04/03/2026. Eje: © OpenStreetMap contributors.\n"
            "Georreferencia aproximada · EPSG:32719 · comparación exploratoria de cubiertas, sin evaluación de peligro.",
            ha="center",
            fontsize=8,
            color="#475569",
        )
        fig.subplots_adjust(top=0.93, bottom=0.11, left=0.05, right=0.95, hspace=0.12, wspace=0.08)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)
        evidence = {
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "sector_bounds_metric": list(extents[0]),
            "zoom_bounds_metric": list(zoom),
            "zoom_selection": "midpoint of longest river segment inside sector, 180 m square",
            "crs": raster.crs.to_string(),
            "parameters": manifests[0]["parameters"],
            "sector_detection_counts": {"baseline": len(buildings[0]), "candidate": len(buildings[1])},
            "image_id": manifests[0]["image_id"],
            "accuracy_evaluation": "not available: no local reference labels",
        }
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
