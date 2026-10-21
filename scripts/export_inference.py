"""Export actual image detections and an offline map for the presentation."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.data_loader import load_study_area
from src.detection_client import load_detection
from src.spatial_analysis import analyze_proximity
from src.export import analysis_bundle, csv_bytes, geojson_bytes


def generate(run_directory, destination, band=25, area_id="chullo"):
    data = load_detection(run_directory, area_id=area_id)
    area = load_study_area(area_id)
    if data["buildings"].empty:
        raise ValueError("La ejecución no tiene detecciones dentro del sector piloto.")
    if not 0 <= band <= area["config"]["max_band_m"]:
        raise ValueError("La banda está fuera de la cobertura verificada.")
    result = analyze_proximity(data["buildings"], area["rivers"], band, area["config"]["analysis_crs"])
    summary = result["summary"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = f"ia_{area_id}_{band:g}m"
    (destination / f"edificaciones_{suffix}.csv").write_bytes(csv_bytes(summary))
    (destination / f"edificaciones_{suffix}.geojson").write_bytes(geojson_bytes(summary))
    (destination / f"analisis_{suffix}.zip").write_bytes(analysis_bundle(summary, result["relations"], data["metadata"], band))
    within = int(summary.within_band.sum())
    fig = plt.figure(figsize=(15, 9), facecolor="#f6f8fa")
    ax = fig.add_axes([0.05, 0.15, 0.62, 0.73])
    if data["preview_path"]:
        bounds = data["preview_bounds"]
        ax.imshow(plt.imread(data["preview_path"]), extent=[bounds[0][1], bounds[1][1], bounds[0][0], bounds[1][0]])
    result["bands"].plot(ax=ax, color="#fbbf24", alpha=.13, edgecolor="none")
    for flag, color in [(False, "#80e7ff"), (True, "#ffb829")]:
        subset = summary.loc[summary.within_band == flag]
        if not subset.empty:
            subset.plot(ax=ax, facecolor="none", edgecolor=color, linewidth=.9)
    area["rivers"].plot(ax=ax, color="#17e7e7", linewidth=2)
    w, s, e, n = area["area"].total_bounds
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1 / math.cos(math.radians((s+n)/2)))
    ax.set_axis_off()
    fig.text(.055,.94,f"{area_id.replace('_', ' ').upper()} · DETECCIÓN CON IA",fontsize=22,fontweight="bold",color="#0f172a")
    fig.text(.055,.905,"Imagen real → Mask R-CNN → contornos → distancias",fontsize=14,color="#475569")
    fig.text(.72,.79,str(len(summary)),fontsize=38,fontweight="bold",color="#0f172a")
    fig.text(.72,.75,"contornos detectados en el sector",fontsize=11,color="#475569")
    fig.text(.72,.65,str(within),fontsize=38,fontweight="bold",color="#b45309")
    fig.text(.72,.61,f"a {band:g} m o menos del eje",fontsize=12,color="#475569")
    confidence = data["manifest"].get("parameters", {}).get("confidence_threshold")
    fig.text(.72,.53,f"Modelo: Mask R-CNN ResNet50-FPN\nPesos preentrenados · GeoAI USA\nUmbral de confianza: {confidence}",fontsize=10,color="#475569",linespacing=1.6)
    fig.legend(handles=[Line2D([0],[0],color="#0891b2",lw=2,label="Eje cartografiado · OSM"),
                       Patch(facecolor="none",edgecolor="#b45309",label="Dentro de banda"),
                       Patch(facecolor="none",edgecolor="#0891b2",label="Fuera de banda")],
               loc="upper left",bbox_to_anchor=(.71,.41),frameon=False,fontsize=11,labelspacing=1.1)
    fig.text(.72,.2,"Detecciones automáticas.\nPueden existir omisiones y errores.\nProximidad no equivale a peligro.",fontsize=10,color="#475569",linespacing=1.5)
    image = data["manifest"].get("image", {})
    source = image.get("provenance", {}).get("attribution") or image.get("source_id", "Imagen documentada en manifiesto")
    fig.text(.055,.084,str(source)[:155],fontsize=8,color="#475569")
    fig.text(.055,.054,f"Cauce: © OpenStreetMap contributors. Captura: {image.get('observed_at') or 'fecha no confirmada'}. Distancias UTM 19S. No validación de campo.",fontsize=8,color="#64748b")
    image_file = destination / f"mapa_{suffix}.png"
    fig.savefig(image_file, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    report = {"run_directory":str(run_directory),"buildings":len(summary),"within_band":within,
              "outside_band":len(summary)-within,"band_m":band,"minimum_distance_m":float(summary.distance_m.min()),
              "maximum_distance_m":float(summary.distance_m.max()),"image_sha256":image.get("sha256"),
              "model_sha256":data["manifest"]["model"]["sha256"],"map":str(image_file)}
    (destination / f"resumen_{suffix}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=ROOT/"outputs/inference/chullo_demo")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/presentacion")
    parser.add_argument("--band",type=float,default=25)
    parser.add_argument("--area",default="chullo")
    args = parser.parse_args()
    print(json.dumps(generate(args.run,args.output,args.band,args.area),ensure_ascii=False,indent=2))
