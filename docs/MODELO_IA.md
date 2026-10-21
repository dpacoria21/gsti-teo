# Detector de edificaciones mediante IA

El motor ejecuta **Mask R-CNN con ResNet50 y Feature Pyramid Network (FPN)**. Usa los pesos preentrenados `building_footprints_usa.pth` publicados por **giswqs/geoai**. No se entrenó un modelo nuevo y no se usan los polígonos de Microsoft como respuesta del detector.

## Procedencia y reproducibilidad

| Elemento | Valor |
|---|---|
| Repositorio de pesos | [giswqs/geoai](https://huggingface.co/giswqs/geoai) |
| Revisión fijada | `43b6aa04156d7d9d2b90fe321096f15f03353cd2` |
| Archivo | `models/building_footprints_usa.pth` |
| Tamaño | 176 225 905 bytes |
| SHA256 | `3aea5d0da7803be31ff4e2e1e5ac747f15cab2a13d6bb1a9319b6a6ba8bc7bca` |
| Licencia declarada por el repositorio | MIT |
| Clases | Fondo y edificación |
| Datos de entrenamiento | Publicado para USA; no se ha verificado un inventario completo de entrenamiento en la ficha del modelo |
| Ajuste con datos de Arequipa | No realizado |

El constructor sigue la [implementación oficial de GeoAI](https://github.com/opengeos/geoai/blob/9e330c6fd0fe8f66ea4761707d753ef69b98ec37/geoai/extract.py) y su [ejemplo de edificios USA](https://opengeoai.org/examples/building_footprints_usa/). Se utiliza `torchvision` directamente para mantener aisladas las dependencias y conservar las máscaras de cada instancia y su puntuación. No se usa ArcGIS ni se depende de una cuenta de pago para ejecutar los pesos.

Se verifican el hash y tamaño antes de cargar el modelo. `torch.load(weights_only=True)` lee exclusivamente pesos; `load_state_dict(strict=True)` verifica todas las claves y dimensiones. Las 307 claves del checkpoint coincidieron con la arquitectura.

## Recorrido de la imagen a los contornos

1. Verificar GeoTIFF, CRS, RGB uint8 y resolución entre 0.05 y 2 m por píxel. El intervalo es un control de admisión; no garantiza precisión.
2. Dividir en mosaicos de 512 × 512 píxeles con 128 píxeles de solapamiento. Se cubren también los extremos de la imagen.
3. Convertir RGB a `[0,1]`; el modelo aplica normalización ImageNet y redimensionamiento de lado mínimo 800 / máximo 1333, igual al constructor oficial de GeoAI.
4. Ejecutar la red en modo evaluación, sin gradientes. Se utiliza CUDA cuando está disponible, con CPU como alternativa.
5. Filtrar instancias por puntuación del detector, por defecto 0.50. Convertir cada máscara con umbral 0.50 en contornos geográficos; **los rectángulos de detección no se convierten en huellas**. Se preservan huecos e irregularidades.
6. Filtrar áreas menores de 10 m². Depurar duplicados entre mosaicos por IoU ≥ 0.35 o intersección / área menor ≥ 0.70, dando preferencia a máscaras que no quedan cortadas por un borde interno.
7. Exportar `buildings.geojson` en EPSG:4326 y una vista RGB reproyectada para superponerla en el mapa. La aplicación calcula después distancias en metros con su motor espacial.

Las imágenes deben llegar ya georreferenciadas. No se inventan coordenadas para capturas de pantalla, fotos oblicuas o archivos JPEG sin información espacial. La resolución de exportación no demuestra por sí sola la resolución nativa del sensor.

## Ejecución

El entorno de IA está separado en `.venv-ml`; `.venv` mantiene la aplicación de mapa. La máquina de demostración tiene una NVIDIA RTX 5070 Ti y PyTorch CUDA 12.8. El lock guarda las versiones exactas instaladas.

```powershell
uv venv --python 3.12 .venv-ml
uv pip install --python .venv-ml\Scripts\python.exe torch==2.11.0 torchvision==0.26.0 --index https://download.pytorch.org/whl/cu128
uv pip install --python .venv-ml\Scripts\python.exe -r requirements-ml.in

.\.venv-ml\Scripts\python.exe scripts/run_detection.py `
  --image data/imagery/mariano_melgar/mariano_melgar_rgb.tif `
  --image-manifest data/imagery/mariano_melgar/manifest.json `
  --output-dir outputs/inference/mariano_melgar_demo `
  --threshold 0.5
```

El ejemplo utiliza la ortofoto incluida de Mariano Melgar (2018). Chullo también dispone de imagen real y resultados en `data/imagery/chullo/chullo_rgb.tif` y `outputs/inference/chullo_demo`; cambiar ambas rutas y el manifiesto para reproducir ese caso. Para instalar con versiones fijadas también se puede ejecutar `PREPARAR_IA.ps1`.

En un equipo sin NVIDIA, instalar `torch` y `torchvision` desde `https://download.pytorch.org/whl/cpu` y agregar `--device cpu`. Las versiones del lock CUDA son específicas de ese índice; no es necesario instalar CUDA manualmente para usar las ruedas incluidas.

El CLI imprime progreso JSON por línea y termina con código 0 al completar. Un error produce código 1 y `manifest.json` con estado `error`. Una detección vacía es un resultado posible del modelo, no se reemplaza por polígonos de otra fuente.

Cada corrida produce:

- `buildings.geojson`: identificador, puntuación real del modelo, origen, imagen, modelo, área y marcas de borde.
- `preview.png`: imagen RGB real reproyectada para el mapa, sin contornos pintados.
- `manifest.json`: hash de imagen y pesos, procedencia, parámetros, dimensiones, resolución, dispositivo, versiones, fecha de ejecución y límites de superposición.

`observed_at` conserva únicamente la fecha de captura declarada por la fuente. La fecha de descarga o inferencia no se presenta como fecha de captura.

## Qué demuestra y qué requiere validación

El resultado demuestra ejecución de una red preentrenada sobre píxeles de una imagen y posterior análisis espacial. La puntuación de la red no es una probabilidad calibrada. El modelo puede omitir cubiertas, agrupar construcciones o detectar objetos incorrectos; la generalización USA → Arequipa requiere revisión local.

El algoritmo identifica posibles edificaciones y cubiertas. No determina uso residencial, ocupación, límite predial ni amenaza por inundación. La distancia al eje de la torrentera es independiente de la confianza de detección y tampoco equivale a una faja marginal oficial.

Para cuantificar calidad, crear etiquetas manuales de un sector independiente y reportar precisión/recall o IoU. Una imagen visualmente convincente y los conteos obtenidos no son una validación de exactitud.

## Comprobaciones del motor

```powershell
.\.venv-ml\Scripts\python.exe -m pytest tests/test_building_detection.py tests/test_inference_config.py tests/test_inference_pipeline.py tests/test_georeference_capture.py tests/test_raster_cli.py -q
```

Las comprobaciones cubren cobertura de bordes, conversión de máscaras con huecos, georreferenciación métrica, depuración de duplicados, rechazo de imágenes sin CRS y rechazo de píxeles demasiado grandes. El modelo real se probó además en una tesela NAIP de ejemplo del publicador: eso verifica ejecución, no precisión en Chullo ni independencia de los datos de entrenamiento.

### Corrida real de Arequipa, 14 de octubre de 2026

Se ejecutó el checkpoint sobre `data/imagery/mariano_melgar/mariano_melgar_rgb.tif`, una imagen RGB real de CENEPRED, capturada el **5 de junio de 2018**, de un sector de **Mariano Melgar, distinto de Chullo**. La adquisición y procedencia están en el manifiesto de la imagen. Este ejemplo se conserva como segundo caso.

- Imagen analizada: 800 × 800 píxeles, remuestreada a 0.5 m por píxel, EPSG:32719.
- Cuatro mosaicos procesados con el modelo real; 263 candidatos tras filtro de área, 83 duplicados suprimidos, **180 contornos finales** con umbral 0.50.
- Tiempo registrado por el motor: **2.346 s en CUDA**, una vez disponibles pesos, paquetes y cachés del sistema. El proceso completo tarda más por el inicio de Python y sus bibliotecas; no constituye un compromiso de latencia para otras imágenes o equipos.
- Cero geometrías inválidas; 12 contornos llegan al borde de imagen y 7 conservan una marca de borde interno de mosaico.
- Salida: `outputs/inference/mariano_melgar_demo/`.

La comparación visual de `qa_overlay.png` confirma contornos sobre cubiertas visibles y revela **omisiones numerosas en manzanas densas y algunos falsos positivos sobre calles o patios**. Aumentar el umbral a 0.80 deja 135 de los 180 contornos, pero no elimina todos los errores. No se eligió ese umbral como una calibración de precisión. Se mantiene la puntuación disponible al usuario y la condición de detecciones sin revisión.

La tesela NAIP de control produjo nueve contornos tanto en CPU como CUDA. Esa prueba verifica ambas rutas de ejecución. No mide calidad en Arequipa.

### Corrida de Chullo con Google Earth Pro

La captura **Google Earth; Image © 2026 Airbus**, con fecha mostrada **2026-03-04**, se exportó mediante Google Earth Pro el 14 de octubre de 2026. Se guardaron dos JPG idénticos en cámara y tamaño (8192 × 6390): uno limpio con créditos y otro con controles KML. Se ajustó una transformación afín usando cuatro esquinas de coordenadas conocidas; dos marcadores reservados dieron RMSE de 0,227 m. Este valor describe consistencia interna del visor, **no precisión absoluta ni error de alineación con OSM**.

`scripts/georeference_capture.py` transforma la imagen completa a EPSG:32719, RGB uint8 y cuadrícula de 0,5 m (2621 × 2045). Conserva originales, máscara de validez, puntos, residuos, hashes y condiciones de uso. No genera detalle ni entrena la red. El modelo procesa 35 ventanas; con umbral 0,50 conserva 1006 contornos en la imagen completa, de los que **237 intersectan el sector piloto**. El análisis obtiene **27 a 25 m o menos** y **56 a 50 m o menos** del eje OSM.

La revisión muestra cubiertas reconocidas, omisiones en manzanas densas y falsos positivos sobre una cancha, vegetación y patios. No hay etiquetas independientes para reportar precisión/recall. Esta imagen no es un dataset abierto; se conserva para la demostración académica con atribución y [condiciones de Google](https://about.google/brand-resource-center/products-and-services/geo-guidelines/).
