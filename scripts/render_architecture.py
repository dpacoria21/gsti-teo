"""Render the audited architecture as editable SVG and slide-ready PNG figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "arquitectura"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
INK, MUTED, LINE = "#172E44", "#50677A", "#7A8D9B"
TEAL, BLUE, AMBER = "#087E84", "#315CA8", "#A66A17"


def page(title, subtitle, state, color=TEAL):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    ax.text(.6, 8.48, "CHULLO  /  ARQUITECTURA DEL PROYECTO", size=10, color=color, weight="bold")
    ax.text(.6, 7.94, title, size=25, color=INK, weight="bold")
    ax.text(.6, 7.52, subtitle, size=11.5, color=MUTED)
    ax.text(15.4, 8.48, state, size=10, color=color, ha="right", weight="bold")
    ax.plot([.6, 15.4], [.62, .62], color="#DDE5EA", lw=1)
    ax.text(.6, .3, "Diseño basado en el código y los manifiestos locales · 21 oct 2026", size=9, color=MUTED)
    ax.text(15.4, .3, "Proyecto académico · Arequipa", size=9, color=MUTED, ha="right")
    return fig, ax


def box(ax, x, y, w, h, title, body, color=TEAL, fill="#EFF7F7", fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.09", fc=fill, ec="#CCDADD", lw=1))
    ax.plot([x+.14, x+.14], [y+.17, y+h-.17], color=color, lw=3)
    ax.text(x+.3, y+h-.26, title, size=12.5, color=INK, weight="bold", va="top")
    ax.text(x+.3, y+h-.66, body, size=fs, color=MUTED, va="top", linespacing=1.55)


def arrow(ax, pts, label=None, label_xy=None, color=LINE, dashed=False, bidirectional=False):
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.4, linestyle="--" if dashed else "-")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="<->" if bidirectional else "-|>", mutation_scale=13,
                                lw=1.4, color=color, linestyle="--" if dashed else "-", shrinkA=3, shrinkB=3))
    if label:
        ax.text(*label_xy, label, size=9.5, color=MUTED, ha="center", va="center", bbox={"facecolor":"white", "edgecolor":"none", "pad":2})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


fig, ax = page("De una imagen a un análisis de proximidad", "Flujo funcional implementado · la preparación geográfica ocurre antes de cargar la imagen en la aplicación.", "IMPLEMENTADO")
xs = [.55, 3.05, 5.55, 8.05, 10.55, 13.05]
titles = ["1. Imagen real", "2. Georreferencia", "3. Detección IA", "4. Contornos", "5. Distancias", "6. Resultados"]
bodies = ["Google Earth Pro\nCrédito: Airbus\n04 mar 2026",
          "JPG + controles KML\nGeoTIFF RGB\nUTM 19 Sur",
          "Mask R-CNN\nResNet-50 + FPN\nPyTorch / Torchvision",
          "Máscaras → polígonos\nEliminar duplicados\nSeleccionar el sector",
          "Huella → eje OSM\nMínimo en metros\nComparar con banda",
          "Mapa y tabla\nCSV / GeoJSON / ZIP\nFuentes y parámetros"]
for x, title, body in zip(xs, titles, bodies):
    box(ax, x, 4.72, 2.25, 1.72, title, body, fs=9.7)
for x in xs[:-1]:
    arrow(ax, [(x+2.25,5.57),(x+2.5,5.57)])
ax.text(.65, 6.79, "PREPARACIÓN PREVIA", size=10, color=MUTED, weight="bold")
ax.text(5.65, 6.79, "PROCESAMIENTO EN LA APLICACIÓN", size=10, color=TEAL, weight="bold")
box(ax, 5.55, 2.35, 2.25, 1.58, "Pesos guardados", "giswqs / GeoAI\nPublicado para USA\nSin ajuste local", BLUE, "#F0F4FA", fs=9.7)
box(ax, 8.05, 2.35, 2.25, 1.58, "Área de estudio", "Polígono del sector\nConfiguración local\nHuellas completas", BLUE, "#F0F4FA", fs=9.7)
box(ax, 10.55, 2.35, 2.25, 1.58, "Cauce conocido", "Eje de OpenStreetMap\nReferencia: INGEMMET\nBanda elegida: 25 m", BLUE, "#F0F4FA", fs=9.5)
for x in [5.55,8.05,10.55]:
    arrow(ax, [(x+1.125,3.93),(x+1.125,4.72)])
ax.text(.65,3.55,"Un solo modelo de IA", size=17,color=INK,weight="bold")
ax.text(.65,3.06,"Reconoce posibles edificaciones.\nLa torrentera se carga desde cartografía;\nlas distancias se calculan con geometría.",size=11.2,color=MUTED,va="top",linespacing=1.65)
ax.text(13.15,3.55,"Trazabilidad",size=14,color=INK,weight="bold")
ax.text(13.15,3.1,"Imagen y fecha\nModelo y versión\nHashes y parámetros",size=10.6,color=MUTED,va="top",linespacing=1.65)
ax.text(.65,1.5,"1 006 contornos en la imagen  →  237 en el sector piloto  →  27 dentro de 25 m",size=17,color=INK,weight="bold")
ax.text(.65,1.03,"Conteos de candidatos de la corrida guardada. La cercanía al eje no determina peligro ni una faja marginal oficial.",size=10.8,color=MUTED)
save(fig,"01-arquitectura-funcional")

fig, ax = page("Despliegue actual: todo se procesa en una PC", "Aplicación modular en Python, con un proceso de IA separado y almacenamiento en archivos.", "IMPLEMENTADO")
ax.add_patch(FancyBboxPatch((.55,1.04),14.9,6.02,boxstyle="round,pad=.02,rounding_size=.12",fc="none",ec="#B4C3CD",lw=1.4,linestyle="--"))
ax.text(.8,6.72,"EQUIPO WINDOWS DEL PROYECTO",size=10,color=MUTED,weight="bold")
box(ax,.9,4.24,2.6,1.65,"Navegador", "Interfaz Streamlit\nMapa Folium / Leaflet\nSelección y descargas",fs=10.5)
box(ax,4.35,4.24,4.55,1.65,"Servidor de la aplicación · .venv", "app.py + módulos GIS\nGeoPandas / Shapely / PyProj\nValida, analiza y construye el mapa",fs=11)
box(ax,10.05,4.24,4.95,1.65,"Proceso de inferencia · .venv-ml", "scripts/run_detection.py\nPyTorch / Torchvision / Rasterio\nGPU CUDA disponible; alternativa CPU", BLUE, "#F0F4FA", fs=11)
arrow(ax,[(3.5,5.07),(4.35,5.07)],bidirectional=True)
ax.text(3.91,3.78,"HTTP / WebSocket\n127.0.0.1:8501",ha="center",size=9.5,color=MUTED,linespacing=1.5)
arrow(ax,[(8.9,5.38),(10.05,5.38)])
arrow(ax,[(10.05,4.76),(8.9,4.76)],dashed=True)
ax.text(9.47,5.77,"Lanza CLI",ha="center",size=9.5,color=MUTED)
ax.text(9.47,4.04,"Progreso JSON",ha="center",size=9.5,color=MUTED)
box(ax,4.35,1.56,10.65,1.46,"Disco local · entradas y resultados reproducibles", "GeoTIFF + cauces GeoJSON + configuración + pesos .pth\nDetecciones GeoJSON + preview PNG + manifiestos JSON + exportaciones", BLUE,"#F7F9FB",fs=11)
arrow(ax,[(6.6,4.24),(6.6,3.02)],bidirectional=True)
arrow(ax,[(12.5,4.24),(12.5,3.02)],bidirectional=True)
ax.text(6.6,3.62,"Lee / guarda",size=9.5,color=MUTED,ha="center",bbox={"facecolor":"white","edgecolor":"none"})
ax.text(12.5,3.62,"Lee / guarda",size=9.5,color=MUTED,ha="center",bbox={"facecolor":"white","edgecolor":"none"})
ax.text(.95,2.85,"Preparación previa",size=12,color=INK,weight="bold")
ax.text(.95,2.4,"Google Earth Pro\n+ georreferenciación\n+ descarga de capas",size=10.5,color=MUTED,va="top",linespacing=1.55)
ax.text(.95,6.2,"Internet: adquisición inicial, mapa base y recursos web. La inferencia y las distancias usan datos locales.",size=10.4,color=MUTED)
save(fig,"02-despliegue-actual")

fig, ax = page("Despliegue propuesto para varios usuarios", "Reutiliza el detector y el motor GIS; incorpora API, trabajos en cola y persistencia central.", "PROPUESTA · NO IMPLEMENTADO",AMBER)
box(ax,.65,4.65,2.25,1.45,"Usuarios", "Navegador\nAcceso por HTTPS",fs=10.8)
box(ax,3.48,4.65,2.2,1.45,"Entrada web", "Proxy HTTPS\nNginx",fs=10.8)
box(ax,6.38,5.55,3.1,1.25,"Interfaz Streamlit", "Mapa, carga y resultados",fs=10.6)
box(ax,6.38,3.55,3.1,1.35,"API · FastAPI", "Validación, trabajos\ny consulta del análisis",fs=10.6)
box(ax,10.05,3.55,2.05,1.35,"Cola · Redis", "Trabajos pendientes\nIdentificador de tarea",BLUE,"#F0F4FA",fs=9.6)
box(ax,12.7,3.55,2.7,1.35,"Worker · Celery", "IA + motor GIS\nGPU o CPU",BLUE,"#F0F4FA",fs=10.6)
box(ax,6.38,1.13,3.1,1.4,"PostgreSQL / PostGIS", "Sectores, cauces, huellas\nDistancias y estados",BLUE,"#F0F4FA",fs=10.3)
box(ax,12.05,1.13,3.35,1.4,"Almacenamiento S3", "Imágenes y pesos versionados\nExportaciones y manifiestos",BLUE,"#F0F4FA",fs=10.2)
arrow(ax,[(2.9,5.37),(3.48,5.37)],bidirectional=True)
arrow(ax,[(5.68,5.37),(5.98,5.37),(5.98,6.16),(6.38,6.16)])
arrow(ax,[(7.93,5.55),(7.93,4.9)],bidirectional=True)
arrow(ax,[(9.48,4.23),(10.05,4.23)])
arrow(ax,[(12.1,4.23),(12.7,4.23)])
arrow(ax,[(7.93,3.55),(7.93,2.53)],bidirectional=True)
arrow(ax,[(14.05,3.55),(14.05,2.53)],bidirectional=True)
arrow(ax,[(12.9,3.55),(12.9,3.03),(9.05,3.03),(9.05,2.53)],label="Guarda análisis y estado",label_xy=(10.9,3.03))
arrow(ax,[(9.48,3.8),(9.73,3.8),(9.73,1.85),(12.05,1.85)],label="Acceso a archivos",label_xy=(10.9,1.85),dashed=True)
ax.text(10.05,5.8,"Procesamiento asíncrono",size=13,color=INK,weight="bold")
ax.text(10.05,5.35,"La API devuelve un ID; la interfaz consulta\nel progreso hasta que el trabajo termina.",size=10.5,color=MUTED,va="top",linespacing=1.6)
ax.text(.7,3.5,"Una VM al inicio",size=15,color=INK,weight="bold")
ax.text(.7,3.02,"Servicios en Docker Compose.\nLa cola permite limitar concurrencia\ny mover el worker a otra máquina\ncuando aumente la carga.",size=10.8,color=MUTED,va="top",linespacing=1.75)
ax.text(.7,1.65,"La base guarda geometrías;\nel almacén guarda archivos grandes.\nSi se usa GPU, se asigna al worker.",size=10.2,color=MUTED,va="top",linespacing=1.6)
save(fig,"03-despliegue-propuesto")

print(f"Arquitectura exportada: {OUT}")
for path in sorted(OUT.glob("0*.*")):
    print(f"{path.name}: {path.stat().st_size:,} bytes")
