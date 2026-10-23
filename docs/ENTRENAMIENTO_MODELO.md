# Entrenamiento del detector de edificaciones

El 23 de octubre de 2026 se completó un ajuste supervisado de Mask R-CNN con imágenes y anotaciones de referencia de SpaceNet 2, Khartoum. El candidato mejoró el F1 de **14.24 % a 40.18 %** en las mismas 40 escenas de prueba reservadas. Es un modelo experimental: su precisión en esa prueba fue **35.45 %**, y todavía no existe una evaluación de referencia que demuestre una mejora en Chullo.

El entrenamiento guardó un checkpoint separado. No sobrescribió `models/building_footprints_usa.pth`. Se integró como opción **«Ajustado · SpaceNet (experimental)»**; el original sigue siendo la selección inicial. Los pesos exactos y los informes de evaluación se incluyen en `models/spacenet2_khartoum_v1/`. La copia del manifiesto del dataset conserva rutas relativas a su carpeta original `data/training/spacenet2_khartoum/`; no es un segundo dataset.

La comprobación funcional con la imagen real de Chullo terminó en 10.053 s en CUDA: 545 contornos en toda la imagen y 136 que intersectan el sector. Con rojo hasta 25 m y amarillo hasta 50 m, se obtuvieron 25 rojos, 14 amarillos y 97 verdes. La demo del original conserva sus 237 contornos del sector en otra carpeta. Estas cifras verifican la integración, **no una mejora cuantificada en Chullo**, porque no hay etiquetas locales de referencia. Los resultados del ajuste están en `outputs/inference/chullo_spacenet2_khartoum_v1_demo/`.

La comparación visual sobre la misma extensión encontró un problema de transferencia: el ajuste **agrupa varias cubiertas y patios en polígonos grandes**, llegando a abarcar conjuntos similares a manzanas. El original omite muchas cubiertas, pero separa algunas de ellas. Por ello **no se promueve el ajuste como reemplazo del original para Chullo**. Se mantiene accesible como experimento, con aviso visible. `scripts/compare_building_models.py` reproduce la comparación en `outputs/training_validation/chullo_comparison.png`; el zoom se fija por el punto medio del cauce y no por la calidad de las detecciones. El siguiente ajuste útil necesita referencias locales revisadas y una prueba reservada de Chullo, con derechos de uso adecuados de las imágenes para entrenamiento.

## Qué se entrenó

Se partió del checkpoint `giswqs/geoai/building_footprints_usa`, revisión `43b6aa04156d7d9d2b90fe321096f15f03353cd2`, con SHA256 `3aea5d0da7803be31ff4e2e1e5ac747f15cab2a13d6bb1a9319b6a6ba8bc7bca`.

La arquitectura permanece compatible con la aplicación: `torchvision.maskrcnn_resnet50_fpn`, dos clases —fondo y edificación—, normalización ImageNet y redimensionamiento interno de lado menor a 800 píxeles y lado mayor máximo de 1333. El ajuste actualizó FPN, RPN y las cabezas de cajas y máscaras. El cuerpo ResNet permaneció congelado durante las cinco épocas; las estadísticas de BatchNorm también se mantuvieron fijas, por el lote de una imagen.

Esto es *fine-tuning*: se adaptan pesos existentes usando ejemplos supervisados adicionales. No se entrenó desde cero y no se usaron predicciones del propio detector como etiquetas correctas.

No se reconstruyó ni auditó el corpus completo utilizado para generar los pesos originales. La separación espacial que verificamos corresponde al nuevo conjunto train/val/test; no demuestra ausencia absoluta de coincidencias con todo el preentrenamiento del modelo base.

## Datos y separación

La procedencia, adquisición, licencia y selección espacial se describen en [DATOS_ENTRENAMIENTO.md](DATOS_ENTRENAMIENTO.md). El manifiesto utilizado es `data/training/spacenet2_khartoum/manifest.json`, SHA256 `0f0c5746876f27612a56862560ca120dc65bbe9fb6ea04662ab15e75331bdede`.

| Conjunto | Escenas | Anotaciones originales | Instancias utilizables |
| --- | ---: | ---: | ---: |
| Entrenamiento | 200 | 5,558 | 5,532 |
| Validación | 40 | 1,217 | 1,212 |
| Prueba reservada | 40 | 1,281 | 1,275 |
| Total | 280 | 8,056 | 8,019 |

Cada escena contiene una imagen RGB de 650 × 650 píxeles y sus polígonos de referencia. Se preservaron los originales uint16 y se prepararon RGB uint8 con percentiles 2 y 98 sobre píxeles válidos. El cargador respeta esos uint8 sin volver a estirarlos; los convierte a valores entre 0 y 1. Su compatibilidad con uint16 permite aplicar el mismo tipo de estiramiento cuando se proporcionan datos sin preparar.

Las etiquetas se reproyectan al CRS de cada raster y se convierten en una máscara por entidad. Un MultiPolygon sigue siendo una instancia, se conservan sus huecos y se aplica la máscara de píxeles válidos. Se omiten máscaras vacías o con menos de cuatro píxeles; de ahí la diferencia entre las anotaciones originales y las instancias utilizadas. Las cajas se calculan a partir de las máscaras, con límites superiores exclusivos para evitar cajas degeneradas.

El split está fijado por escena antes de cualquier aumento. Según el manifiesto final, las distancias mínimas entre los límites completos de las imágenes son 563.69 m entre entrenamiento y validación, 375.23 m entre entrenamiento y prueba, y 5,456.80 m entre validación y prueba. No hay superposición entre splits ni hashes de imágenes duplicados. El control previo comprobó archivos, CRS, geometrías, máscaras, cajas y valores de imagen; no utilizó métricas de prueba para tomar decisiones.

## Configuración ejecutada

| Parámetro | Valor |
| --- | --- |
| Épocas | 5 |
| Lote | 1 imagen completa; sin recortes |
| Optimizador | AdamW |
| Tasa de aprendizaje | 0.0001 |
| Weight decay | 0.0001 |
| Semilla de entrenamiento | 42 |
| Aumentos, solo train | Giros de 0/90/180/270° y reflejos horizontal/vertical |
| Precisión de entrenamiento | AMP float16 en CUDA |
| Escala AMP inicial | 8192, con ajuste automático ante desbordamientos |
| Recorte de gradientes | Norma máxima 1.0 |
| Procesos de carga | 0, compatible con Windows |
| Selección de checkpoint | Mayor F1 de validación; en empate se conserva la época anterior |

Los registros por paso y época están en `events.jsonl` y `epochs.json`. El intento exitoso realizó 1,000 actualizaciones del optimizador y no omitió pasos por AMP.

Un primer intento se detuvo en el paso 6 por un desbordamiento transitorio de gradientes AMP. Se corrigió el tratamiento para permitir que GradScaler reduzca su escala y omita una actualización, registrándolo y deteniéndose si ocurren diez desbordamientos consecutivos. El intento fallido se conserva en `outputs/training/spacenet2_khartoum_v1_failed_initial_amp`; no llegó a evaluar test. Se reinició desde los mismos pesos originales y el mismo split. El resultado documentado aquí pertenece exclusivamente al intento posterior completo.

## Cómo se midió

La evaluación usa máscaras de instancia, con puntuación del detector ≥ 0.5, probabilidad de máscara ≥ 0.5 e IoU ≥ 0.5. Se emparejan predicciones y referencias uno a uno, empezando por la mayor IoU y sin reutilizar instancias. Una detección duplicada cuenta como falso positivo. Se recortan ambas clases de máscaras a los mismos píxeles válidos; se ignoran predicciones que quedan completamente vacías.

Se agregan verdaderos positivos (TP), falsos positivos (FP) y falsos negativos (FN) de todas las imágenes para calcular precisión, recall y F1. También se informa la IoU media de los emparejamientos correctos. **Estas métricas no son COCO AP ni mAP.**

El protocolo procesa escenas completas de 650 píxeles y mantiene el límite original de 100 detecciones por imagen. Ninguna escena de validación o prueba supera 100 instancias de referencia utilizables. La aplicación, en cambio, usa mosaicos de 512 píxeles con 128 de solapamiento y filtros de área y duplicados. Por eso los resultados siguientes describen el protocolo supervisado de SpaceNet, y no una medición de exactitud de todo el procesamiento desplegado en Chullo.

El test reservado solo se evaluó después de seleccionar la mejor época por validación. Se comparó una vez el modelo original con el candidato seleccionado, sobre exactamente las mismas escenas y con los mismos umbrales. No se ajustaron umbrales ni se eligió otra época a partir de los resultados de test.

## Resultados

| Época | Pérdida media de entrenamiento | F1 de validación |
| --- | ---: | ---: |
| Modelo original | — | 16.76 % |
| 1 | 1.8218 | 51.96 % |
| 2 | 1.6431 | 54.00 % |
| 3 | 1.5775 | 51.94 % |
| 4 | 1.5608 | 51.78 % |
| **5, seleccionada** | **1.5337** | **56.48 %** |

| Prueba reservada: 40 escenas, 1,275 referencias | Original | Candidato |
| --- | ---: | ---: |
| Precisión | 26.55 % | 35.45 % |
| Recall | 9.73 % | 46.35 % |
| F1 | 14.24 % | 40.18 % |
| IoU media de verdaderos positivos | 65.96 % | 69.33 % |
| TP | 124 | 591 |
| FP | 343 | 1,076 |
| FN | 1,151 | 684 |

El candidato recupera más edificios, pero también produce más falsos positivos. Por tanto, esta mejora no permite presentar sus contornos como un inventario verificado.

La diferencia de F1 es de **25.94 puntos porcentuales**. Un bootstrap pareado de 1,000 remuestreos por imagen, semilla 42, produjo un intervalo percentil del 95 % de **21.86 a 29.76 puntos** para esa diferencia. Es un análisis exploratorio: las imágenes cercanas pueden tener dependencia espacial que este remuestreo no modela.

## Checkpoint y evidencias

Los resultados se guardaron en `outputs/training/spacenet2_khartoum_v1/`:

- `run.json`: configuración, hashes, ejecución, época elegida y resultado general.
- `baseline_val.json` y `best_val.json`: validación usada para seleccionar.
- `best_val_stored_checkpoint.json`: validación tras recargar el archivo final.
- `baseline_test.json`, `candidate_test.json` y `comparison.json`: métricas agregadas, resultados por escena y comparación.
- `events.jsonl` y `epochs.json`: registro de pérdidas y evolución.
- `best_model.pth`: checkpoint separado, con tensores flotantes almacenados como float16 y enteros conservados.

El archivo `best_model.pth` mide **88,170,873 bytes** y tiene SHA256 **`5e929fa3cf1f657b1fa9afd2cbdc48ea475629d8c78910c98c9a99c8fffb90c6`**. Al cargarlo, los parámetros del modelo vuelven a float32 mediante carga estricta del estado. Se evaluó este archivo exacto después de guardarlo; su F1 de validación fue el mismo 56.48 %, y las cifras de test corresponden a él. No se publican estados del optimizador como parte del detector.

## Recursos y reproducción

El entrenamiento se ejecutó con una NVIDIA GeForce RTX 5070 Ti de 15.89 GiB, CPU AMD Ryzen 7 9800X3D, 61.68 GiB de RAM, PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128 y Rasterio 1.5.2.

El intento exitoso duró **206.14 segundos**, incluyendo la validación inicial, las cinco épocas y la comparación final. No incluye descarga de datos, comprobaciones previas ni el intento interrumpido. PyTorch registró un máximo de **1,267.59 MiB de memoria CUDA asignada**; esta medida excluye la memoria reservada por el controlador y otras aplicaciones. Una muestra de Windows durante el entrenamiento registró 3,496 MiB de RAM residente y un máximo observado hasta ese momento de 3,528 MiB; no es una medición del pico final completo.

```powershell
.venv-ml\Scripts\python.exe scripts/train_buildings.py `
  --manifest data/training/spacenet2_khartoum/manifest.json `
  --output-dir outputs/training/otro_experimento `
  --epochs 5 --batch-size 1 --learning-rate 0.0001 `
  --freeze-backbone-epochs 5 --seed 42 --workers 0 --device cuda
```

La carpeta de salida debe ser nueva. El script no descarga pesos automáticamente y no modifica el checkpoint inicial. Las semillas y los parámetros están registrados, pero ROIAlign en CUDA no garantiza una ejecución bit a bit idéntica: el entrenamiento utiliza algoritmos deterministas cuando están disponibles y registra esta limitación.

Las 12 pruebas en `tests/test_training.py` cubren rasterización, huecos y MultiPolygon, máscaras válidas, transformaciones de CRS, separación de entradas, emparejamiento uno a uno, bootstrap, formato del checkpoint y selección por validación antes de evaluar test. Pasaron en CPU; Ruff también pasó para los módulos nuevos.

## Alcance del candidato

Este experimento demuestra que el proyecto puede adquirir datos de referencia, adaptar un detector, guardar un modelo propio derivado y compararlo sin seleccionar por test. Para afirmar una mejora local se necesitan anotaciones revisadas de Chullo y una evaluación independiente de sus imágenes. Además, Khartoum proviene de otra ciudad, otra fecha, otro sensor y otros tipos de techos; la adaptación puede ayudar a ciertas apariencias y perjudicar otras.

El detector identifica contornos candidatos de edificaciones. No determina ocupación, peligro de inundación, estabilidad del terreno ni permiso de construcción. La clasificación de proximidad a la torrentera sigue siendo un análisis geográfico posterior e independiente del entrenamiento.
