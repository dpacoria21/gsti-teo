# Entrenamiento del detector de edificaciones

## Ajuste local completado: Chullo, piloto provisional

El 23 de octubre de 2026 se completó un ajuste local de Mask R-CNN, inicializado desde el modelo original USA. En dos recortes de prueba de Chullo, el **F1 pasó de 8.76 % a 24.88 %** y la recuperación de referencias de 5.00 % a 22.50 %. También aumentaron los falsos positivos de **11 a 70** y la precisión bajó de **35.29 % a 27.84 %**. El resultado recupera más cubiertas candidatas, pero no constituye una mejora uniforme ni un inventario verificado.

El modelo registrado es **«Ajustado · Chullo (piloto)»**, variante `chullo_local_v1`. Se conserva separado del original y del ajuste anterior SpaceNet. Sus referencias son **anotaciones visuales provisionales del asistente, sin revisión humana** (`assistant_visual_manual`, `human_reviewed=false`). La unidad anotada es una cubierta distinguible en la imagen; no identifica una vivienda, una propiedad ni su ocupación. El experimento previo con SpaceNet se documenta más abajo como historial.

### Imagen, anotaciones y separación local

La fuente es `data/imagery/chullo/chullo_rgb.tif`, procedente de Google Earth Pro / Airbus, con fecha de captura declarada **2026-03-04** y georreferenciación aproximada. Todas las áreas pertenecen a **una sola captura**. La atribución y las condiciones de la imagen se conservan en `models/chullo_local_v1/LICENSE_AND_ATTRIBUTION.txt`; este registro no declara una licencia abierta ni un permiso general para redistribuir o entrenar con la imagen.

Se planificaron y anotaron ocho áreas originales de 192 × 192 píxeles: cuatro para entrenamiento, dos para validación y dos para prueba. Contienen **464 polígonos originales**. Los límites se delinearon sobre el RGB sin usar predicciones del detector; las sombras, posibles vehículos y superficies ambiguas se marcaron como regiones ignoradas.

`train_1100_500` quedó excluida **antes de entrenar y sin consultar resultados del modelo**: ningún recorte de la cuadrícula fija de 192, 128 o 96 píxeles evitaba todas las regiones ignoradas, más un margen de un píxel, conservando al menos ocho referencias utilizables. Sus anotaciones se mantienen como evidencia de procedencia, pero no alimentaron el entrenamiento.

| Conjunto local | Áreas originales utilizadas | Imágenes entregadas al modelo | Referencias utilizables |
| --- | ---: | ---: | ---: |
| Entrenamiento | 3 de las 4 anotadas | 4 recortes: 192, 128, 96 y 96 píxeles | 136 cubiertas distintas; 151 instancias contando repetición entre recortes |
| Validación | 2 | 2 imágenes de 192 × 192 | 70 |
| Prueba reservada | 2 | 2 imágenes de 192 × 192 | 120 de 121 polígonos originales |

Los dos recortes de 96 píxeles se superponen dentro de entrenamiento; no son nuevas imágenes independientes. Las 151 instancias de entrenamiento se presentan en cada época, con aumentos aleatorios. No se deben describir como 151 edificios diferentes ni multiplicarlas por el número de épocas para afirmar que se dispone de más datos.

El manifiesto fijó una separación mínima entre áreas originales de **104 m** entre entrenamiento y validación, **354 m** entre entrenamiento y prueba, y **154 m** entre validación y prueba, sin solapes entre conjuntos. Esto evita compartir píxeles entre conjuntos, pero no demuestra independencia estadística o de adquisición: todos proceden de la misma captura. Las geometrías y sus huecos se rasterizan por instancia. Se aplican los píxeles válidos y las regiones evaluables; se excluyen máscaras de menos de cuatro píxeles. Una referencia de prueba deja de ser utilizable tras excluir su región ambigua, de ahí 121 polígonos frente a 120 instancias evaluadas.

Manifiesto congelado: `data/training/chullo_local_v1/manifest.json`, SHA256 `0631962c46b7cdb2a6b3b17cdf1cd3afd32e22f09c1d0bf36090f989b841ef4e`. Su copia en `models/chullo_local_v1/dataset_manifest.json` es de auditoría; las rutas de imágenes y etiquetas se resuelven desde la carpeta del dataset, no desde la carpeta de pesos.

### Entrenamiento local ejecutado

Se mantuvo la arquitectura `torchvision.maskrcnn_resnet50_fpn`, con dos clases, fondo y edificación, y normalización ImageNet. El ajuste partió de `models/building_footprints_usa.pth`, SHA256 `3aea5d0da7803be31ff4e2e1e5ac747f15cab2a13d6bb1a9319b6a6ba8bc7bca`; **no continuó desde el ajuste SpaceNet**. El corpus completo de preentrenamiento de ese checkpoint no fue reconstruido ni auditado.

| Parámetro | Ejecución local |
| --- | --- |
| Épocas / actualizaciones | 50 / 200; cuatro actualizaciones por época |
| Lote / procesos de carga | 1 / 0 |
| Optimizador | AdamW; learning rate 0.0001; weight decay 0.0001 |
| Componentes entrenados | FPN, RPN y cabezas ROI de cajas y máscaras |
| Componentes congelados | Cuerpo ResNet durante las 50 épocas; estadísticas de BatchNorm fijas |
| Aumentos, solo train | Giros de 0/90/180/270 grados y reflejos horizontal/vertical |
| Escala espacial | Factor fijo **1.5625**: 192 → 300, 128 → 200, 96 → 150 píxeles |
| Precisión / gradientes | AMP float16 en CUDA; recorte de norma a 1.0; cero pasos omitidos |
| Semilla / selección | 42; mayor F1 de validación, conservando la época anterior en caso de empate |
| Mejor época | **47** de 50 |
| Tiempo del proceso | **53.11 s**, incluida validación y comparación final; excluye anotación y preparación de datos |
| Equipo / memoria CUDA asignada máxima | NVIDIA GeForce RTX 5070 Ti; **653.69 MiB**, sin incluir toda la reserva del controlador |

La selección utilizó las dos imágenes de validación. El test reservado se evaluó después de elegir la época; no se ajustaron umbrales ni se eligió otra época con sus resultados. Los registros de las 50 épocas y los eventos están en `outputs/training/chullo_local_v1/epochs.json` y `events.jsonl`.

### Resultados locales y uniones de cubiertas

Se usa puntuación del detector ≥ 0.5, probabilidad de máscara ≥ 0.5 e IoU de máscara ≥ 0.5. El emparejamiento es uno a uno por mayor IoU: una detección duplicada cuenta como falso positivo. Se aplican las mismas máscaras válidas y regiones ignoradas a predicciones y referencias. Las métricas son precisión, recall y F1 de instancias con umbrales fijos; **no son COCO AP ni mAP**.

La comparación principal es original USA frente al modelo local seleccionado. El modelo SpaceNet se evaluó posteriormente sobre los mismos dos recortes y con el mismo protocolo, solo para contextualizar el problema de bloques; ese resultado no intervino en la elección de época ni de umbrales locales.

| Prueba local: 2 recortes, 120 referencias evaluables | Original USA | Ajuste local | SpaceNet previo, comparación posterior |
| --- | ---: | ---: | ---: |
| Precisión | 35.29 % | 27.84 % | 4.76 % |
| Recall | 5.00 % | 22.50 % | 0.83 % |
| F1 | 8.76 % | 24.88 % | 1.42 % |
| TP / FP / FN | 6 / 11 / 114 | 27 / 70 / 93 | 1 / 20 / 119 |
| IoU media de los emparejamientos correctos | 65.61 % | 63.00 % | 65.68 % |
| Predicciones que unen varias referencias | 6 | 12 | 15 |
| Referencias afectadas por esas uniones | 15 | 27 | 106 |

Una unión se registra cuando una predicción cubre al menos el 50 % de cada una de al menos dos referencias distintas. Es un diagnóstico adicional de máscaras, no una prueba de que se hayan fusionado propiedades o viviendas reales.

El local reduce las uniones grandes **frente al ajuste previo SpaceNet**: de 15 a 12 predicciones y de 106 a 27 referencias afectadas. **Frente al original USA, las uniones aumentan de 6 a 12 y las referencias afectadas de 15 a 27.** Por tanto, no se afirma que el problema esté resuelto ni que el local separe siempre las cubiertas mejor que el original.

La diferencia principal de F1 es de 16.13 puntos porcentuales. El JSON incluye un bootstrap pareado de 1,000 remuestreos, pero solo hay dos imágenes de prueba, procedentes de la misma captura. Su intervalo estrecho no debe interpretarse como una estimación fiable de incertidumbre o generalización a toda Chullo.

### Archivo exacto evaluado y evidencias locales

La mejor validación durante selección fue **44.19 % F1**. Al guardar los tensores flotantes en float16 y volver a cargar estrictamente el archivo en un modelo float32, el F1 observado fue **43.02 %**, una diferencia de **−1.16 puntos porcentuales**: 38 TP / 64 FP / 32 FN antes y 37 TP / 65 FP / 33 FN después. No se oculta esta diferencia ni se atribuye al archivo final el 44.19 % de la selección. El original obtuvo 22.22 % en esa validación. **Las métricas de prueba publicadas corresponden al archivo almacenado y recargado**, sin volver a seleccionar una época según test.

El checkpoint registrado `models/chullo_local_v1/weights.pth` mide **88,170,873 bytes** y tiene SHA256 **`dcc985d4b57ec6ab0c6dbc36bf48f8b6433bfe61e9d89faf2445bb7b05d5f61c`**. Es la copia del `best_model.pth` de la corrida completada. Conserva los tensores enteros en su tipo original; los parámetros de inferencia se cargan en float32. No incluye estado del optimizador ni sobrescribe el modelo original.

Las evidencias están en `outputs/training/chullo_local_v1/` y sus copias portables en `models/chullo_local_v1/`:

- `run.json` y `config.json`: parámetros, hashes, época y recursos.
- `baseline_val.json`, `best_val.json` y `best_val_stored_checkpoint.json`: validación original, selección y verificación del archivo guardado.
- `baseline_test.json`, `candidate_test.json` y `comparison.json`: evaluación principal sobre la misma prueba reservada.
- `epochs.json` y el `events.jsonl` de la corrida: evolución y actualizaciones.
- `manifest.json`: registro del modelo, procedencia, limitaciones y hashes de sus informes.

La comparación posterior SpaceNet está en `outputs/training/chullo_local_v1/prior_spacenet_test.json`. La figura `outputs/training_validation/chullo_local_v1_comparison.png` muestra **las dos áreas fijas de validación**, con referencias provisionales, original y ajuste local; **no muestra test**. Estas áreas también se usaron para elegir la época. El color morado marca regiones ambiguas excluidas. No se seleccionaron recortes nuevos según el aspecto favorable de sus predicciones.

El entrenamiento y la evaluación usan el factor 1.5625 sobre recortes completos, sin mosaicos ni filtros de área. En producción, la aplicación procesa mosaicos de **512 píxeles con 128 de solapamiento**, redimensionados por el detector, y después aplica área mínima y eliminación de duplicados. Así, las métricas anteriores evalúan el protocolo local de referencia; **no miden la exactitud de todo el procesamiento desplegado**. Tampoco verifican ocupación, peligro, estabilidad del terreno, faja marginal ni autorización para construir. El semáforo de proximidad se calcula después mediante geometría y mantiene sus limitaciones propias.

### Reproducción del ajuste local

Los comandos siguientes reconstruyen la secuencia en una **copia de trabajo limpia**, con la imagen, la selección y los ocho JSON de anotaciones ya disponibles y congelados. No generan etiquetas nuevas. Las carpetas de entrenamiento y registro deben ser nuevas; los scripts rechazan sobrescribirlas. En el proyecto existente se conservan las evidencias del experimento terminado. La imagen fuente no se distribuye como un dataset de licencia abierta.

```powershell
.venv-ml\Scripts\python.exe scripts/prepare_chullo_labels.py `
  --selection data/training/chullo_local_v1/selection.json

.venv-ml\Scripts\python.exe scripts/train_buildings.py `
  --manifest data/training/chullo_local_v1/manifest.json `
  --output-dir outputs/training/chullo_local_v1 `
  --checkpoint models/building_footprints_usa.pth `
  --epochs 50 --batch-size 1 --learning-rate 0.0001 --weight-decay 0.0001 `
  --freeze-backbone-epochs 50 --seed 42 --workers 0 --device cuda `
  --image-scale 1.5625 --score-threshold 0.5 --mask-threshold 0.5 `
  --match-iou 0.5 --gradient-clip 1.0 --bootstrap-samples 1000

.venv-ml\Scripts\python.exe scripts/register_local_model.py `
  --run-dir outputs/training/chullo_local_v1 `
  --data-manifest data/training/chullo_local_v1/manifest.json `
  --output-dir models/chullo_local_v1

.venv-ml\Scripts\python.exe scripts/run_detection.py `
  --image data/imagery/chullo/chullo_rgb.tif `
  --image-manifest data/imagery/chullo/manifest.json `
  --output-dir outputs/inference/chullo_local_v1_reproduccion `
  --model-variant chullo_local_v1 --device cuda `
  --threshold 0.5 --mask-threshold 0.5 --tile-size 512 --overlap 128 `
  --min-area-m2 10

.venv-ml\Scripts\python.exe scripts/compare_local_validation.py `
  --manifest data/training/chullo_local_v1/manifest.json `
  --run-dir outputs/training/chullo_local_v1 `
  --output outputs/training_validation/chullo_local_v1_comparison.png
```

Registrar un modelo copia el checkpoint y sus evidencias; no lo entrena, publica ni convierte automáticamente en la opción inicial. Las semillas y los parámetros permiten repetir el procedimiento, pero CUDA/ROIAlign y los distintos entornos no garantizan resultados bit a bit idénticos. Una nueva corrida conserva su propio hash y debe verificar otra vez el checkpoint empaquetado.

## Historial: ajuste previo con SpaceNet 2 Khartoum

Los apartados siguientes conservan el experimento anterior y sus métricas en **Khartoum**, no en Chullo. Su 40.18 % F1 no debe mezclarse con el 24.88 % de la prueba piloto local: usan imágenes, referencias y escalas diferentes.

En el experimento previo del 23 de octubre de 2026 se completó un ajuste supervisado de Mask R-CNN con imágenes y anotaciones de referencia de SpaceNet 2, Khartoum. El candidato mejoró el F1 de **14.24 % a 40.18 %** en las mismas 40 escenas de prueba reservadas. Es un modelo experimental: su precisión en esa prueba fue **35.45 %**, y en ese momento no se había realizado la evaluación piloto local descrita al inicio de este documento.

El entrenamiento guardó un checkpoint separado. No sobrescribió `models/building_footprints_usa.pth`. Se integró como opción **«Ajustado · SpaceNet (experimental)»**; el original sigue siendo la selección inicial. Los pesos exactos y los informes de evaluación se incluyen en `models/spacenet2_khartoum_v1/`. La copia del manifiesto del dataset conserva rutas relativas a su carpeta original `data/training/spacenet2_khartoum/`; no es un segundo dataset.

La comprobación funcional con la imagen real de Chullo terminó en 10.053 s en CUDA: 545 contornos en toda la imagen y 136 que intersectan el sector. Con rojo hasta 25 m y amarillo hasta 50 m, se obtuvieron 25 rojos, 14 amarillos y 97 verdes. La demo del original conserva sus 237 contornos del sector en otra carpeta. Estas cifras verifican la integración, **no una mejora cuantificada en Chullo**, porque esa comprobación precedió a las anotaciones visuales locales provisionales. Los resultados del ajuste están en `outputs/inference/chullo_spacenet2_khartoum_v1_demo/`.

La comparación visual sobre la misma extensión encontró un problema de transferencia: el ajuste **agrupa varias cubiertas y patios en polígonos grandes**, llegando a abarcar conjuntos similares a manzanas. El original omite muchas cubiertas, pero separa algunas de ellas. Por ello **no se promueve el ajuste como reemplazo del original para Chullo**. Se mantiene accesible como experimento, con aviso visible. `scripts/compare_building_models.py` reproduce la comparación en `outputs/training_validation/chullo_comparison.png`; el zoom se fija por el punto medio del cauce y no por la calidad de las detecciones. Esta observación motivó el ajuste local descrito arriba. Sigue pendiente una evaluación local con referencias humanas revisadas e imágenes con derechos de uso adecuados para entrenamiento.

### Qué se entrenó

Se partió del checkpoint `giswqs/geoai/building_footprints_usa`, revisión `43b6aa04156d7d9d2b90fe321096f15f03353cd2`, con SHA256 `3aea5d0da7803be31ff4e2e1e5ac747f15cab2a13d6bb1a9319b6a6ba8bc7bca`.

La arquitectura permanece compatible con la aplicación: `torchvision.maskrcnn_resnet50_fpn`, dos clases —fondo y edificación—, normalización ImageNet y redimensionamiento interno de lado menor a 800 píxeles y lado mayor máximo de 1333. El ajuste actualizó FPN, RPN y las cabezas de cajas y máscaras. El cuerpo ResNet permaneció congelado durante las cinco épocas; las estadísticas de BatchNorm también se mantuvieron fijas, por el lote de una imagen.

Esto es *fine-tuning*: se adaptan pesos existentes usando ejemplos supervisados adicionales. No se entrenó desde cero y no se usaron predicciones del propio detector como etiquetas correctas.

No se reconstruyó ni auditó el corpus completo utilizado para generar los pesos originales. La separación espacial que verificamos corresponde al nuevo conjunto train/val/test; no demuestra ausencia absoluta de coincidencias con todo el preentrenamiento del modelo base.

### Datos y separación

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

### Configuración ejecutada

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

### Cómo se midió

La evaluación usa máscaras de instancia, con puntuación del detector ≥ 0.5, probabilidad de máscara ≥ 0.5 e IoU ≥ 0.5. Se emparejan predicciones y referencias uno a uno, empezando por la mayor IoU y sin reutilizar instancias. Una detección duplicada cuenta como falso positivo. Se recortan ambas clases de máscaras a los mismos píxeles válidos; se ignoran predicciones que quedan completamente vacías.

Se agregan verdaderos positivos (TP), falsos positivos (FP) y falsos negativos (FN) de todas las imágenes para calcular precisión, recall y F1. También se informa la IoU media de los emparejamientos correctos. **Estas métricas no son COCO AP ni mAP.**

El protocolo procesa escenas completas de 650 píxeles y mantiene el límite original de 100 detecciones por imagen. Ninguna escena de validación o prueba supera 100 instancias de referencia utilizables. La aplicación, en cambio, usa mosaicos de 512 píxeles con 128 de solapamiento y filtros de área y duplicados. Por eso los resultados siguientes describen el protocolo supervisado de SpaceNet, y no una medición de exactitud de todo el procesamiento desplegado en Chullo.

El test reservado solo se evaluó después de seleccionar la mejor época por validación. Se comparó una vez el modelo original con el candidato seleccionado, sobre exactamente las mismas escenas y con los mismos umbrales. No se ajustaron umbrales ni se eligió otra época a partir de los resultados de test.

### Resultados

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

### Checkpoint y evidencias

Los resultados se guardaron en `outputs/training/spacenet2_khartoum_v1/`:

- `run.json`: configuración, hashes, ejecución, época elegida y resultado general.
- `baseline_val.json` y `best_val.json`: validación usada para seleccionar.
- `best_val_stored_checkpoint.json`: validación tras recargar el archivo final.
- `baseline_test.json`, `candidate_test.json` y `comparison.json`: métricas agregadas, resultados por escena y comparación.
- `events.jsonl` y `epochs.json`: registro de pérdidas y evolución.
- `best_model.pth`: checkpoint separado, con tensores flotantes almacenados como float16 y enteros conservados.

El archivo `best_model.pth` mide **88,170,873 bytes** y tiene SHA256 **`5e929fa3cf1f657b1fa9afd2cbdc48ea475629d8c78910c98c9a99c8fffb90c6`**. Al cargarlo, los parámetros del modelo vuelven a float32 mediante carga estricta del estado. Se evaluó este archivo exacto después de guardarlo; su F1 de validación fue el mismo 56.48 %, y las cifras de test corresponden a él. No se publican estados del optimizador como parte del detector.

### Recursos y reproducción

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

### Alcance del candidato

Este experimento demuestra que el proyecto puede adquirir datos de referencia, adaptar un detector, guardar un modelo propio derivado y compararlo sin seleccionar por test. Para afirmar una mejora local se necesitan anotaciones revisadas de Chullo y una evaluación independiente de sus imágenes. Además, Khartoum proviene de otra ciudad, otra fecha, otro sensor y otros tipos de techos; la adaptación puede ayudar a ciertas apariencias y perjudicar otras.

El detector identifica contornos candidatos de edificaciones. No determina ocupación, peligro de inundación, estabilidad del terreno ni permiso de construcción. La clasificación de proximidad a la torrentera sigue siendo un análisis geográfico posterior e independiente del entrenamiento.
