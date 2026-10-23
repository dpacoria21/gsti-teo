# Datos para ajustar el detector de edificaciones

Se preparó un conjunto de **280 imágenes satelitales de Khartoum con 8056 anotaciones de edificaciones de SpaceNet 2**. Las imágenes y sus polígonos de referencia permiten ajustar y evaluar Mask R-CNN con las clases fondo y edificación. Esta preparación no demuestra todavía una mejora del modelo ni su precisión en Arequipa.

## Procedencia y acceso

SpaceNet 2 publica imágenes WorldView-3 y huellas de edificaciones. La ficha de Khartoum identifica la adquisición `104001000A6A1E00`, del **13 de abril de 2015**, y el producto RGB con pansharpening de aproximadamente 0.3 m. La licencia declarada del conjunto es **CC BY-SA 4.0**. Se conservan los originales y se identifican las modificaciones realizadas. Fuentes: [SpaceNet 2](https://spacenet.ai/spacenet-buildings-dataset-v2/), [Khartoum](https://spacenet.ai/khartoum/) y [licencia](https://creativecommons.org/licenses/by-sa/4.0/).

El [registro de AWS](https://registry.opendata.aws/spacenet/) permite acceso público sin cuenta mediante `--no-sign-request`. Se verificó acceso HTTPS anónimo a los objetos; no se necesitan claves. Se descargaron archivos individuales para evitar el archivo completo, cuyo tamaño comprobado es 5071429867 bytes.

Prefijo de origen:

```text
https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/train/AOI_5_Khartoum/
```

Cada identificador `N` tiene un par:

```text
PS-RGB/SN2_buildings_train_AOI_5_Khartoum_PS-RGB_imgN.tif
geojson_buildings/SN2_buildings_train_AOI_5_Khartoum_geojson_buildings_imgN.geojson
```

Ejemplos verificados: [imagen 1](https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/train/AOI_5_Khartoum/PS-RGB/SN2_buildings_train_AOI_5_Khartoum_PS-RGB_img1.tif) y [anotaciones 1](https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/train/AOI_5_Khartoum/geojson_buildings/SN2_buildings_train_AOI_5_Khartoum_geojson_buildings_img1.geojson). Las URLs exactas de los **280 pares seleccionados**, sus tamaños, ETag y hashes constan en el manifiesto local.

El catálogo público contiene 1012 pares RGB/anotaciones. Se leyeron primero cabeceras TIFF de 4096 bytes para conocer la ubicación sin descargar toda la colección. Las imágenes inspeccionadas tienen 650 × 650 píxeles, tres bandas `uint16`, CRS EPSG:4326 y paso de cuadrícula de 0.0000027 grados. Los GeoJSON declaran CRS84. Se utilizaron los polígonos de referencia de SpaceNet, sin generar etiquetas con el propio detector ni con Microsoft Building Footprints.

## Partición geográfica

La semilla es `20261023`. La selección reserva franjas geográficas completas: validación al oeste, entrenamiento en el centro y prueba al este. La elección aleatoria ocurre únicamente dentro de esas franjas. Los aumentos o recortes que se hagan durante el entrenamiento deben conservar el conjunto de su imagen de origen.

| Conjunto | Imágenes | Anotaciones de referencia | Extensión WGS84: oeste, sur, este, norte |
|---|---:|---:|---|
| Entrenamiento | 200 | 5558 | 32.5174284, 15.5138112, 32.5595484, 15.7366962 |
| Validación | 40 | 1217 | 32.4911034, 15.5138112, 32.5121634, 15.5506662 |
| Prueba reservada | 40 | 1281 | 32.5613034, 15.5208312, 32.5665684, 15.7349412 |

Estas extensiones describen el conjunto de teselas; no implican cobertura continua de todo el rectángulo. Se proyectaron los límites completos de cada imagen a **UTM 36N, EPSG:32636**, apropiado para esta adquisición, y se exigió una separación mínima de 200 m entre conjuntos. Las líneas de división se fijaron en coordenadas este `447965.308` y `452872.926` m, con márgenes de 100 m a cada lado. La comprobación final obtuvo:

| Pares de conjuntos | Distancia mínima entre límites completos |
|---|---:|
| Entrenamiento–validación | 563.695 m |
| Entrenamiento–prueba | 375.232 m |
| Validación–prueba | 5456.804 m |

No hay solapamiento de imágenes entre conjuntos ni hashes RGB duplicados entre las 280 imágenes finales. Esta separación evita que teselas vecinas que comparten edificios entren simultáneamente en entrenamiento y evaluación.

## Calidad y preparación RGB

Se descargaron 360 pares: **80 se excluyeron** por tener menos del 50 % de píxeles válidos. Permanecen en `raw/` para mantener trazabilidad, pero no aparecen en `manifest.json.scenes`. Los reemplazos se tomaron de la misma franja geográfica, sin modificar la partición según resultados del modelo. La fracción válida mínima final es 51.1 % en entrenamiento, 58.5 % en validación y 54.6 % en prueba.

La transformación aplicada a cada imagen es:

1. Leer las tres bandas del producto `PS-RGB`. Conservar como válido el píxel admitido por la máscara original que tenga al menos una banda distinta de cero.
2. Calcular los percentiles 2 y 98 de cada canal sobre sus píxeles válidos.
3. Estirar linealmente ese intervalo a `[0,255]`, recortar los extremos y guardar `uint8`.
4. Guardar una máscara de validez interna y asignar la interpretación rojo, verde y azul a las bandas.
5. Conservar exactamente 650 × 650 píxeles, transformación espacial y CRS. **No se remuestrea ni se crea resolución adicional.**

Los valores de los percentiles y la fracción válida se guardan por imagen. El cálculo utiliza únicamente los píxeles de esa imagen, sin etiquetas ni estadísticas de otras particiones. Las zonas sin datos deben seguir excluidas de la evaluación. Los 8056 polígonos son el conteo original de referencia; la rasterización puede descartar geometrías vacías o demasiado pequeñas y debe informar esos descartes por separado. Dos imágenes de entrenamiento superan las 100 anotaciones; el máximo es 119. En validación el máximo es 75 y en prueba 85.

Se verificaron **840 hashes SHA256**: imagen original, imagen preparada y GeoJSON de cada una de las 280 escenas. También se comprobaron tamaños y ETag de S3, CRS, dimensiones, límites y máscara. Una inspección visual de cuatro escenas encontró alineación general entre imagen y polígonos; es una comprobación de preparación, no una certificación exhaustiva de las etiquetas.

## Reproducción y archivos

Desde el entorno de IA del proyecto:

```powershell
.\.venv-ml\Scripts\python.exe scripts/prepare_training_data.py `
  --output data/training/spacenet2_khartoum `
  --train 200 --val 40 --test 40 `
  --seed 20261023 --gap-m 200 --workers 8
```

El módulo `src/training/acquisition.py` realiza la adquisición. Las descargas originales acumuladas son **920236994 bytes**; los 280 pares incluidos suman 716731011 bytes. Los RGB preparados ocupan 300649212 bytes adicionales. El directorio completo ocupa aproximadamente **1.225 GB decimales (1.14 GiB)**, incluidos originales excluidos y metadatos; las cabeceras y listados añaden unos pocos megabytes de tráfico a la descarga original. La adquisición limita los pares descargados a 1000000000 bytes.

```text
data/training/spacenet2_khartoum/
├── manifest.json                 # Escenas finales, split, URLs, hashes, parámetros
├── source_index.json             # Cabeceras y origen de los 1012 pares disponibles
├── selection.json                # Selección final y comprobación geográfica
├── quality_rejections.json       # 80 exclusiones por cobertura de píxeles
├── acquisition_validation.json   # Auditoría de la adquisición realizada
├── LICENSE_AND_ATTRIBUTION.txt
├── raw/images/                   # 360 originales uint16
├── raw/labels/                   # 360 GeoJSON originales
└── rgb/                          # 280 imágenes uint8 con máscara
```

Estos archivos grandes permanecen locales y están excluidos de Git. Se conservan la atribución a SpaceNet Partners y las condiciones CC BY-SA 4.0 para los datos y sus adaptaciones. Cita del conjunto: Van Etten, Lindenbaum y Bacastow (2018), *SpaceNet: A Remote Sensing Dataset and Challenge Series*, [arXiv:1807.01232](https://arxiv.org/abs/1807.01232).

## Alcance de la evaluación posterior

El modelo original y el ajustado deben evaluarse contra los mismos polígonos reservados, con emparejamiento de instancias e IoU, precisión, sensibilidad y F1; el ajuste de parámetros y selección del mejor checkpoint utiliza únicamente validación. El [criterio de SpaceNet](https://github.com/SpaceNetChallenge/utilities#evaluation-metric) emplea correspondencia uno a uno entre propuestas y referencias alrededor de un umbral IoU de 0.5. También resulta útil informar AP de máscaras y falsos positivos por imagen.

Khartoum aporta ejemplos satelitales de una ciudad árida, pero **no constituye verdad de terreno de Chullo**. Las diferencias de materiales, densidad urbana, captura y procesamiento pueden cambiar el resultado. Tampoco se conoce el corpus completo usado para preentrenar el checkpoint original, por lo que no se puede certificar que nunca haya visto estos datos. Una mejora en esta prueba reservada deberá describirse como mejora en este conjunto de Khartoum; afirmar mejora local exige anotaciones independientes revisadas de Arequipa.
