"""Generate reproducible exports and a static presentation map from local inputs."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.data_loader import load_study_area
from src.export import analysis_bundle, csv_bytes, geojson_bytes
from src.spatial_analysis import analyze_proximity


def generate(area_id: str, band_m: float, destination: Path) -> dict:
    data = load_study_area(area_id, root=ROOT)
    if band_m > data["config"].get("max_band_m", 100):
        raise ValueError("La banda excede la cobertura validada de este sector.")
    result = analyze_proximity(data["buildings"], data["rivers"], band_m, data["config"]["analysis_crs"])
    summary = result["summary"]
    destination.mkdir(parents=True, exist_ok=True)
    suffix = f"{area_id}_{band_m:g}m"
    (destination / f"edificaciones_{suffix}.csv").write_bytes(csv_bytes(summary))
    (destination / f"edificaciones_{suffix}.geojson").write_bytes(geojson_bytes(summary))
    (destination / f"analisis_{suffix}.zip").write_bytes(analysis_bundle(summary, result["relations"], data["metadata"], band_m))

    projected = summary.to_crs(data["config"]["analysis_crs"])
    rivers = data["rivers"].to_crs(projected.crs)
    area = data["area"].to_crs(projected.crs)
    bands = result["bands"].to_crs(projected.crs)
    within = int(summary["within_band"].sum())
    fig = plt.figure(figsize=(15, 9), facecolor="#f6f8fa")
    ax = fig.add_axes([0.05, 0.14, 0.62, 0.74], facecolor="#edf2f5")
    if not bands.empty:
        bands.plot(ax=ax, color="#f6b44b", alpha=0.25, edgecolor="none")
    outside = projected.loc[~projected.within_band]
    inside = projected.loc[projected.within_band]
    if not outside.empty:
        outside.plot(ax=ax, color="#94a3b8", edgecolor="#64748b", linewidth=0.35)
    if not inside.empty:
        inside.plot(ax=ax, color="#d97706", edgecolor="#92400e", linewidth=0.45)
    rivers.plot(ax=ax, color="#0891b2", linewidth=2.0)
    area.boundary.plot(ax=ax, color="#334155", linewidth=1.0, linestyle="--")
    minx, miny, maxx, maxy = area.total_bounds
    ax.set_xlim(minx-25, maxx+25)
    ax.set_ylim(miny-25, maxy+25)
    ax.set_aspect("equal")
    ax.set_axis_off()
    scale_y = miny+20
    ax.plot([minx+20, minx+120], [scale_y, scale_y], color="#0f172a", linewidth=3)
    ax.text(minx+70, scale_y+9, "100 m", ha="center", fontsize=9, color="#0f172a")
    ax.annotate("N", xy=(0.94, 0.98), xytext=(0.94, 0.90), xycoords="axes fraction", ha="center", fontsize=11,
                arrowprops={"arrowstyle":"-|>", "color":"#0f172a"}, color="#0f172a")
    fig.text(.06, .945, "CHULLO", fontsize=23, fontweight="bold", color="#0f172a")
    fig.text(.06, .91, "Edificaciones y proximidad al eje de la torrentera", fontsize=14, color="#475569")
    fig.text(.72, .81, f"{len(summary):,}".replace(",", " "), fontsize=36, fontweight="bold", color="#0f172a")
    fig.text(.72, .77, "edificaciones analizadas", fontsize=12, color="#475569")
    fig.text(.72, .67, str(within), fontsize=36, fontweight="bold", color="#b45309")
    fig.text(.72, .63, f"a {band_m:g} m o menos del eje", fontsize=12, color="#475569")
    fig.text(.72, .54, f"{100*within/len(summary):.1f}% dentro de la banda", fontsize=12, color="#0f172a")
    handles = [Line2D([0],[0],color="#0891b2",lw=2,label="Eje cartografiado · OSM"),
               Patch(facecolor="#d97706",label="Dentro de banda analítica"),
               Patch(facecolor="#94a3b8",label="Fuera de banda analítica"),
               Patch(facecolor="#f6b44b",alpha=.4,label=f"Banda analítica · {band_m:g} m")]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.71,.48),frameon=False,labelspacing=1.1,fontsize=11)
    fig.text(.72,.25,"Indicador de proximidad.\nNo determina seguridad ni peligro.\nFaja marginal: no evaluada.",fontsize=11,color="#475569",linespacing=1.5)
    fig.text(.06,.073,"Fuentes: Microsoft Global ML Building Footprints · © OpenStreetMap contributors (ODbL).",fontsize=10,color="#475569")
    generated_date = datetime.now(timezone(timedelta(hours=-5))).date().isoformat()
    fig.text(.06,.042,f"Contornos existentes; fecha de captura no conocida. Cálculos en WGS 84 / UTM 19S. Elaboración: {generated_date}.",fontsize=9,color="#64748b")
    image_path = destination / f"mapa_{suffix}.png"
    fig.savefig(image_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    report = {"area_id":area_id,"band_m":band_m,"buildings":len(summary),"within_band":within,
              "outside_band":len(summary)-within,"minimum_distance_m":float(summary.distance_m.min()),
              "maximum_distance_m":float(summary.distance_m.max()),"map":str(image_path)}
    (destination / f"resumen_{suffix}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--area",default="chullo")
    parser.add_argument("--band",type=float,default=25)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"presentacion")
    args = parser.parse_args()
    print(json.dumps(generate(args.area,args.band,args.output),ensure_ascii=False,indent=2))
