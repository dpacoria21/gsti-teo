# Datos del piloto de Chullo y comprobaciones

Preparación y revisión: 14 de octubre de 2026 (Lima). Este documento describe los datos que realmente utiliza el prototipo. El análisis calcula proximidad a un eje cartografiado; no clasifica seguridad, peligro hidráulico ni situación legal.

## Resultado preparado

| Elemento | Resultado |
|---|---|
| Área | Rectángulo WGS84 `[-71.554, -16.403, -71.550, -16.397]`, aproximadamente 0,284 km². |
| Inventario | 177 huellas poligonales de Microsoft seleccionadas por intersección con el área. |
| Objetos en borde | 43 huellas cruzan el límite; se conserva su geometría completa. |
| Eje | Cinco vías OSM conectadas, reunidas en una línea continua de 2.065,5 m. |
| Coordenadas de intercambio | GeoJSON WGS84, longitud/latitud (EPSG:4326). |
| Coordenadas del cálculo | WGS84 / UTM zona 19S, EPSG:32719; distancias en metros. |
| Distancia | Mínima entre la huella completa y el eje de Chullo. |
| Bandas admitidas | Hasta 100 m; el parámetro es analítico. |
| Capas de peligro y faja marginal | No incluidas; resultados correspondientes no evaluados. |

Los 177 objetos son **huellas del inventario**, no 177 viviendas verificadas. Algunas huellas agrupan varias cubiertas y patios. No hay información para confirmar su uso residencial ni su fecha de construcción.

## Procedencia de las edificaciones

Fuente: [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints), tesela de Perú `210031232`, publicación del conjunto de 13 de agosto de 2026. Se conserva el [archivo original comprimido](https://bfppub.z5.web.core.windows.net/2026-08-13/global-buildings.geojsonl/RegionName=Peru/quadkey=210031232/part-00104-110f5303-ff85-4c71-a2bf-c6070024fec8.c000.csv.gz) en `data/raw/` y su SHA-256 en `download_manifest.json`. La licencia declarada es CDLA Permissive 2.0.

Se recorrieron 280.614 registros; 177 intersectan el piloto. La selección no corta polígonos. Los IDs se generan de manera determinista a partir de la geometría normalizada y la tesela. El archivo procesado mantiene las coordenadas originales.

- Ninguna de las 177 geometrías seleccionadas es vacía o inválida.
- No se encontraron duplicados geométricos exactos en la selección.
- Área mínima de huella: 13,47 m²; máxima: 4.898,30 m². La mayor superficie refuerza que algunos objetos son agrupaciones, no viviendas individuales.
- La fecha de captura no está disponible y permanece como `observed_at: null`. La fecha de publicación no se utiliza como fecha de imagen.
- Los valores desconocidos `height=-1` y `confidence=-1` pasan a `null`.
- Se conserva `review_status=geometry_checked_not_field_verified`: no es un catastro ni una revisión de campo.

## Identificación de la torrentera

El eje se obtiene de las vías OSM completas [570369893](https://www.openstreetmap.org/way/570369893), [25980163](https://www.openstreetmap.org/way/25980163), [535091453](https://www.openstreetmap.org/way/535091453), [535091454](https://www.openstreetmap.org/way/535091454) y [781522545](https://www.openstreetmap.org/way/781522545). Se ensamblan por coordenadas de nodos compartidos. No se inventan segmentos para cerrar huecos. La geometría resultante es una línea simple, válida y continua.

Estas vías tienen `waterway=stream`, pero no etiqueta `name`. Por ello, el nombre Chullo se asigna tras contrastar la ubicación con dos publicaciones de INGEMMET:

1. [Informe Técnico A7406 (2023)](https://sigrid.cenepred.gob.pe/sigridv3/documento/16497/descargar): cuadro 9, página impresa 21 (página 27 del PDF), y figuras 17 a 20, páginas impresas 23 y 24. Se inspeccionaron visualmente el cuadro, el mapa y las fotografías.
2. [Opinión Técnica 006-2020](https://sigrid.cenepred.gob.pe/sigridv3/documento/10009/descargar): cuadro 1 en la página 3 y mapa de drenaje en la página 11. El informe denomina Chullo al tramo inspeccionado entre Calle Grande y Av. Metropolitana.

Se compararon cinco ubicaciones publicadas en UTM WGS84 19S con el eje descargado:

| Referencia publicada | Este (m) | Norte (m) | Separación al eje OSM calculada |
|---|---:|---:|---:|
| A7406, cruce 1 | 227534,8 | 8185486,6 | 0,23 m |
| A7406, cruce 2 | 227552,8 | 8185248,9 | 1,52 m |
| A7406, cruce 3 | 227498,5 | 8185057,9 | 3,82 m |
| OT006-2020, extremo sur | 227384,0 | 8184790,0 | 3,37 m |
| OT006-2020, extremo norte | 227488,0 | 8186090,0 | 0,05 m |

Los cruces 2 y 3 y la referencia sur están dentro del piloto. Las otras dos referencias respaldan la continuidad del tramo cargado fuera del área. Estas separaciones permiten corroborar la identidad del cauce, **no estimar la exactitud topográfica del eje**: los puntos son ubicaciones de cruces y extremos de inspección, no controles geodésicos de un levantamiento del centro del cauce. No se ha demostrado independencia entre las fuentes cartográficas originales.

Los puntos publicados se utilizan exclusivamente para comparación. No se unieron para construir un cauce, una faja marginal o un polígono de peligro. Tampoco se transfirieron categorías de peligro del informe a las edificaciones del prototipo.

Atribución: © OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright). Fechas de edición de vías y nodos se conservan por separado; no equivalen a fecha de observación. El eje no es una delimitación oficial.

## Extensión y límites de cobertura

La primera preparación utilizó cuatro vías. La revisión de geometrías completas detectó una huella del borde norte cuya distancia llegaba al extremo cargado. Se incorporó la vía aguas arriba `570369893`, conectada al nodo `1862680670`, y se corroboró también con el extremo norte publicado en OT006-2020.

En el resultado final, los extremos del eje están a **850,87 m y 325,75 m**, respectivamente, de las huellas seleccionadas más cercanas. Ambos exceden la banda máxima de 100 m. Así se conserva margen alrededor de los edificios completos y no solo del rectángulo de estudio. El script comprueba esta condición al preparar los datos.

El sistema evalúa este eje de Chullo. La cobertura no acredita que estén cartografiados todos los drenajes, canalizaciones o peligros posibles del sector. Una huella fuera de la banda no recibe una declaración de seguridad.

## Comprobación visual de 20 huellas

Se seleccionaron 20 huellas completamente contenidas en el área, distribuidas a lo largo del orden norte-sur. Se revisó su superposición sobre una exportación de [Esri World Imagery](https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer), utilizada únicamente como referencia visual. La ortofoto pública de IGN no respondió durante la preparación.

El material de revisión queda localmente en `outputs/validacion/`:

- `muestra_20.csv`: etiquetas 1–20, identificadores y distancia al eje.
- `muestra_visual_1.png` y `muestra_visual_2.png`: los veinte recortes inspeccionados.
- `huellas_revision_visual.png`: ubicación de la muestra en el sector.
- `imagery_qa_metadata.json`: extensión y resolución de la exportación de referencia.
- `eje_referencias_ingemmet.png`: comparación de puntos publicados y eje cartografiado.

Hallazgos cualitativos:

- La muestra presenta correspondencia general con superficies construidas visibles; no se aprecia un desplazamiento general que ubique todas las huellas fuera de la zona.
- Las huellas 6, 9, 14, 15 y 17 incluyen conjuntos de cubiertas o partes de patios. Un polígono no debe interpretarse como una sola vivienda.
- La huella 3 queda parcialmente en sombra; su correspondencia requiere una imagen mejor o inspección. Los contornos pequeños de las muestras 2, 5, 12 y 13 también tienen límites visualmente inciertos.
- Hay diferencias locales de contorno y segmentación. Se conservaron las geometrías originales para que la procedencia y el cálculo sean reproducibles; no se ajustaron manualmente para hacer coincidir la imagen.

La fecha local de la imagen Esri no está verificada, y no se ha probado que sea independiente de las imágenes que originaron las huellas de Microsoft. Esta revisión es operativa: no calcula precisión, recall, IoU, exhaustividad, error de posición ni actualidad. La imagen de revisión no se utiliza para entrenamiento, no es una capa de análisis y no es necesaria para ejecutar la presentación.

## Conteos reproducibles

Sobre las 177 huellas completas y el eje final, la condición `distancia <= banda` produce:

| Banda | Dentro | Fuera |
|---|---:|---:|
| 10 m | 21 | 156 |
| 25 m | 31 | 146 |
| 50 m | 60 | 117 |
| 100 m | 93 | 84 |

Las distancias van de 0 a 331,04 m. Un cero representa intersección entre las geometrías de entrada; puede reflejar ocupación, error de delimitación o cambios entre las fechas de las fuentes. No demuestra por sí solo que una vivienda esté construida sobre el cauce actual.

## Reproducción y archivos

Con el entorno de la aplicación instalado:

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py --offline
```

`--offline` exige los originales en caché y no abre conexiones. Sin esa opción, el script descarga solo los originales faltantes. La aplicación no llama este script ni vuelve a descargar la tesela al abrir el mapa.

La preparación verifica el hash del archivo Microsoft, registra tamaño, URL y descarga de cada original, y rechaza cambios inesperados en la caché. Los XML de OSM conservan su estado en el momento de descarga. Si se desea actualizar datos, hay que revisar la nueva versión y su correspondencia; no basta con reemplazar archivos para mantener la validación anterior.

Archivos de salida:

- `data/processed/chullo/buildings.geojson`
- `data/processed/chullo/rivers.geojson`
- `data/processed/chullo/study_area.geojson`
- `data/processed/chullo/validation.json` con métricas y referencias completas.
- `data/processed/chullo/build_manifest.json` con fuentes y hashes de las tres capas.
- `data/sources.json` con atribuciones y condiciones.
- `config/study_areas.json` con CRS, extensión y banda máxima.

No se repararon ni simplificaron geometrías, no se entrenó un modelo y no se ejecutó detección sobre una imagen nueva. Los contornos son datos de terceros; el prototipo aporta el cálculo espacial, su consulta y exportación con trazabilidad.
