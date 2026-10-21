"""Presentation diagrams with sourced SVG logos.
Run: uv run --with resvg-py python scripts/render_architecture_logos.py
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from html import escape
import hashlib
import json
import resvg_py

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/arquitectura/visual"
ASSETS = OUT / "logos"
ASSETS.mkdir(parents=True, exist_ok=True)
SLUGS = ["python", "streamlit", "pytorch", "geopandas", "leaflet", "googleearth", "openstreetmap", "fastapi", "redis", "celery", "postgresql", "docker", "nginx"]
INK, MUTED, BLUE, GREEN, PURPLE = "#18354A", "#617384", "#169BC0", "#20936B", "#8B65BD"


def fetch(slug):
    path = ASSETS / f"{slug}.svg"
    url = f"https://cdn.simpleicons.org/{slug}"
    if not path.exists():
        with urlopen(Request(url, headers={"User-Agent": "ChulloAcademicDiagram/1.0"}), timeout=30) as response:
            path.write_bytes(response.read())
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    assert root.tag.endswith("svg") and root.get("viewBox") == "0 0 24 24"
    return {"name":slug, "url":url, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "color":root.get("fill")}


with ThreadPoolExecutor(max_workers=6) as pool:
    SOURCES = list(pool.map(fetch, SLUGS))
(OUT / "fuentes-logos.json").write_text(json.dumps({"provider":"Simple Icons", "website":"https://simpleicons.org", "notice":"Logos de sus respectivos titulares; no implican patrocinio.", "logos":SOURCES}, ensure_ascii=False, indent=2),encoding="utf-8")


class Diagram:
    def __init__(self, title, subtitle, proposed=False):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="1125" viewBox="0 0 2000 1125" role="img">
<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>
<defs>
<pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#DEE6EC"/></pattern>
<filter id="shadow" x="-25%" y="-25%" width="150%" height="150%"><feDropShadow dx="0" dy="4" stdDeviation="9" flood-color="#17364A" flood-opacity="0.07"/></filter>
''']
        for name, color in [("blue",BLUE),("green",GREEN),("purple",PURPLE),("gray","#8A9EAD")]:
            self.parts.append(f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 Z" fill="{color}"/></marker>')
        self.parts.append('</defs><rect width="2000" height="1125" fill="#FBFCFE"/><rect x="35" y="175" width="1930" height="855" fill="url(#dots)" opacity=".65"/>')
        self.text(65,61,"CHULLO / AREQUIPA",18,BLUE,weight=700)
        self.text(65,114,title,42,INK,weight=700)
        self.text(65,153,subtitle,21,MUTED)
        self.rect(1570,35,365,43,12,"#FFF1DB" if proposed else "#E6F5EF","none")
        self.text(1752,64,"PROPUESTA · NO IMPLEMENTADO" if proposed else "IMPLEMENTADO · LOCAL",17,"#A36617" if proposed else GREEN,anchor="middle",weight=700)

    def text(self,x,y,value,size=22,color=INK,anchor="start",weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')

    def rect(self,x,y,w,h,r=16,fill="white",stroke="#D6E1E8",dash=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')

    def group(self,x,y,w,h,label,fill="#F9FBFF",color="#CCDDE8",logo=None):
        self.rect(x,y,w,h,20,fill,color,"8 7")
        if logo:
            self.brand(logo,x+20,y+17,30)
        self.text(x+(65 if logo else 24),y+42,label,18,MUTED,weight=600)

    def brand(self,name,x,y,size=76):
        root = ET.fromstring((ASSETS/f"{name}.svg").read_text(encoding="utf-8"))
        color=root.get("fill","#333")
        self.parts.append(f'<g transform="translate({x} {y}) scale({size/24})" fill="{color}">')
        for elem in root:
            if elem.tag.endswith("path"):
                self.parts.append(f'<path d="{escape(elem.attrib["d"],quote=True)}"/>')
        self.parts.append('</g>')

    def generic(self,name,x,y,size=76):
        self.parts.append(f'<g transform="translate({x} {y}) scale({size/80})" stroke="#4F89B4" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round">')
        drawings = {
            "browser": '<rect x="3" y="9" width="74" height="54" rx="5" fill="#EEF6FB"/><path d="M3 23H77 M29 73H51 M40 63V73"/><circle cx="12" cy="16" r="1.6" fill="#4F89B4"/><circle cx="20" cy="16" r="1.6" fill="#4F89B4"/><path d="M30 35L23 43L30 51 M50 35L57 43L50 51"/>',
            "file": '<path d="M16 4H49L66 21V75H16Z" fill="#EEF6FB"/><path d="M49 4V21H66 M26 35H55 M26 46H55 M26 57H47"/>',
            "polygons": '<path d="M6 13L31 5L37 31L10 34Z M47 12L71 20L65 42L43 35Z M16 48L43 42L51 69L23 75Z M59 53L74 58L70 74L55 70Z" fill="#DBF4E9" stroke="#20936B"/>',
            "download": '<path d="M40 6V48 M25 33L40 48L55 33 M10 48V70H70V48 M22 59H58"/>',
            "storage": '<path d="M8 18V64C8 79 72 79 72 64V18" fill="#DDECF8"/><ellipse cx="40" cy="18" rx="32" ry="12" fill="#B7D6EC"/><path d="M8 40C8 55 72 55 72 40 M8 59C8 74 72 74 72 59"/>',
            "gpu": '<rect x="16" y="16" width="48" height="48" rx="5" fill="#EEF6FB"/><rect x="25" y="25" width="30" height="30" rx="3"/><path d="M25 6V16 M40 6V16 M55 6V16 M25 64V74 M40 64V74 M55 64V74 M6 25H16 M6 40H16 M6 55H16 M64 25H74 M64 40H74 M64 55H74"/>'
        }
        self.parts.append(drawings[name])
        self.parts.append('</g>')

    def node(self,x,y,title,subtitle,logo=None,icon=None,w=270,h=180,tint="white",border="#D9E4EA"):
        self.parts.append('<g filter="url(#shadow)">')
        self.rect(x-w/2,y-h/2,w,h,18,tint,border)
        self.parts.append('</g>')
        if logo:
            self.brand(logo,x-34,y-h/2+22,68)
        elif icon:
            self.generic(icon,x-34,y-h/2+18,68)
        self.text(x,y+h/2-48,title,24,INK,"middle",600)
        self.text(x,y+h/2-18,subtitle,18,MUTED,"middle")

    def edge(self,pts,color="blue",label=None,at=None,dash=False,both=False):
        colors={"blue":BLUE,"green":GREEN,"purple":PURPLE,"gray":"#8A9EAD"}
        path="M "+" L ".join(f"{x} {y}" for x,y in pts)
        self.parts.append(f'<path d="{path}" fill="none" stroke="{colors[color]}" stroke-width="2.6" stroke-linejoin="round" marker-end="url(#{color})"'+(f' marker-start="url(#{color})"' if both else '')+(' stroke-dasharray="8 7"' if dash else '')+'/>')
        if label:
            x,y=at
            width=len(label)*10+24
            self.rect(x-width/2,y-18,width,30,6,"#FBFCFE","none")
            self.text(x,y+4,label,18,colors[color],"middle")

    def finish(self,name,note):
        self.text(65,1072,note,18,MUTED)
        self.text(65,1104,"Logos: respectivos titulares · SVG de Simple Icons · Diseño técnico del proyecto",15,"#8998A5")
        self.text(1935,1104,"21 OCT 2026",15,"#8998A5","end")
        self.parts.append('</svg>')
        data="\n".join(self.parts)
        ET.fromstring(data)
        (OUT/f"{name}.svg").write_text(data,encoding="utf-8")
        (OUT/f"{name}.png").write_bytes(resvg_py.svg_to_bytes(svg_string=data,width=4000,font_family="Segoe UI"))
        print(name)


d=Diagram("Arquitectura del prototipo", "Una imagen real → detección de edificaciones → distancias a la torrentera")
d.group(430,215,1505,805,"PC DEL PROYECTO · PYTHON",logo="python")
d.text(205,249,"FUENTES",18,MUTED,"middle",600)
d.edge([(340,360),(490,360)],label="RGB",at=(415,342))
d.edge([(760,360),(910,360)],label="GeoTIFF",at=(835,342))
d.edge([(1180,360),(1340,360)],"green","Máscaras",(1260,342))
d.edge([(1475,450),(1475,595)],"green","Polígonos",(1552,525))
d.edge([(340,685),(490,685)],label="Eje OSM",at=(416,667))
d.edge([(625,595),(625,525),(1430,525),(1430,595)],"green","Cauces",(815,525))
d.edge([(1340,685),(1180,685)],"green","Distancias",(1260,667))
d.edge([(1045,595),(1045,450)],"purple","Ejecutar",(978,488),dash=True)
d.edge([(340,935),(820,935),(820,685),(910,685)],"purple","HTTP local",(560,935),dash=True)
d.edge([(1110,775),(1110,880),(1215,880)],label="Mapa",at=(1133,830))
d.edge([(1045,775),(1045,990),(1725,990),(1725,960)],label="Exportar",at=(1420,990))
d.node(205,360,"Google Earth Pro","Imagen de Chullo",logo="googleearth")
d.node(205,685,"OpenStreetMap","Eje de la torrentera",logo="openstreetmap")
d.node(205,935,"Usuario","Navegador",icon="browser",h=160)
d.node(625,360,"Georreferenciación","Python · Rasterio",logo="python",tint="#F5F9FF")
d.node(1045,360,"Detector de edificios","PyTorch · Mask R-CNN",logo="pytorch",tint="#FFF7F4")
d.node(1475,360,"Contornos","Máscaras → GeoJSON",icon="polygons",tint="#F1FBF6")
d.node(625,685,"Capa de cauces","GeoJSON local",icon="file")
d.node(1045,685,"Aplicación","Streamlit",logo="streamlit",tint="#FFF7F7")
d.node(1475,685,"Motor geográfico","GeoPandas · Shapely",logo="geopandas",tint="#F1FBF6")
d.node(1350,880,"Mapa interactivo","Folium · Leaflet",logo="leaflet",h=160)
d.node(1725,880,"Exportación","CSV · GeoJSON · ZIP",icon="download",h=160)
d.brand("python",1740,305,34)
d.text(1700,375,".venv-ml",18,MUTED)
d.generic("gpu",1738,398,40)
d.text(1700,474,"GPU / CPU",18,MUTED)
d.generic("storage",1728,590,56)
d.text(1755,682,"Archivos locales",19,INK,"middle",600)
d.text(1755,711,"Imágenes · pesos · resultados",15,MUTED,"middle")
d.finish("01-arquitectura-actual-con-logos","Cálculo local · UTM 19S · Distancia a eje cartografiado · Proximidad, no evaluación de peligro")

d=Diagram("Arquitectura de despliegue propuesta", "Interfaz web + API + trabajos en cola + procesamiento IA / GIS",True)
d.group(385,215,1550,805,"SERVIDOR / VM · DOCKER COMPOSE",logo="docker",fill="#FAFCFF")
d.group(1210,365,690,345,"PROCESAMIENTO",fill="#FFF9F5",color="#EEDAC7")
d.edge([(320,600),(455,600)],label="HTTPS",at=(385,581),both=True)
d.edge([(695,600),(750,600),(750,355),(810,355)])
d.edge([(945,445),(945,540)],label="HTTP",at=(1002,497),both=True)
d.edge([(1080,630),(1145,630),(1145,515),(1240,515)],"purple","Trabajo",(1153,578),dash=True)
d.edge([(1460,515),(1560,515)],"purple","Tarea",(1510,497),dash=True)
d.edge([(1830,550),(1868,550),(1868,750),(1695,750),(1695,810)],"green","Imágenes / resultados",(1790,750),both=True)
d.edge([(945,720),(945,810)],"green",both=True)
d.edge([(1570,605),(1570,765),(1010,765),(1010,810)],"green","Estado + geometrías",(1285,765))
d.edge([(1080,670),(1110,670),(1110,900),(1560,900)],label="Archivos",at=(1320,881),dash=True)
d.node(185,600,"Usuarios","Navegador",icon="browser")
d.node(575,600,"NGINX","Entrada HTTPS",logo="nginx",w=240)
d.node(945,355,"Interfaz web","Streamlit · Folium",logo="streamlit",tint="#FFF7F7")
d.node(945,630,"API","FastAPI",logo="fastapi",tint="#EFFAF7")
d.node(1350,515,"Cola","Redis",logo="redis",w=220,tint="#FFF5F4")
d.node(1695,515,"Worker","Celery",logo="celery",tint="#F7FBF3")
d.brand("pytorch",1608,627,37)
d.brand("geopandas",1675,627,37)
d.generic("gpu",1742,624,42)
d.text(1695,694,"PyTorch · GIS · GPU/CPU",16,MUTED,"middle")
d.node(945,900,"Base geográfica","PostgreSQL · PostGIS",logo="postgresql",tint="#F3F6FF")
d.node(1695,900,"Archivos","Almacén compatible con S3",icon="storage",w=300)
d.brand("docker",532,840,84)
d.text(575,953,"Contenedores",22,INK,"middle",600)
d.text(575,984,"Una VM al inicio",18,MUTED,"middle")
d.finish("02-despliegue-propuesto-con-logos","La interfaz consulta el estado del trabajo · El worker procesa · La base guarda geometrías y el almacén guarda imágenes")

(OUT/"LEEME.txt").write_text("ARQUITECTURA VISUAL CHULLO\n\n01: arquitectura implementada del prototipo local.\n02: propuesta de despliegue multiusuario, aún no implementada.\n\nPNG: 4000 x 2250, listos para diapositivas 16:9.\nSVG: diagramas vectoriales editables, con logos incorporados; no necesitan Internet.\n\nAzul = datos o solicitudes; verde = geometrías/resultados; violeta discontinuo = ejecución de tareas. Las etiquetas especifican cada conexión. Los pictogramas genéricos representan usuario, archivos, contornos y GPU; no son logos de proveedores.\n\nLogos obtenidos de Simple Icons; pertenecen a sus respectivos titulares y no implican patrocinio. Fuentes y hashes en fuentes-logos.json.\n\nEl mapa base y la adquisición de imágenes pueden requerir Internet. El procesamiento actual usa datos locales. Google Earth Pro se usó en la preparación; no se llama su API durante cada inferencia. No hay base SQL, API FastAPI ni cola Celery en la versión actual.\n\nGenerador: scripts/render_architecture_logos.py\nComando: uv run --with resvg-py python scripts/render_architecture_logos.py\n",encoding="utf-8")
