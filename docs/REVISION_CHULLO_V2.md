# Revisión de etiquetas y ajuste local de Chullo v2

Las tres rondas terminaron: **ninguna de las siete configuraciones cumplió el criterio de validación**. Se conserva el modelo local v1 en la aplicación y no se promueve un checkpoint v2. El nuevo conjunto de prueba permanece reservado, sin evaluar, porque no se seleccionó un candidato elegible. La auditoría de etiquetas y el entrenamiento quedaron registrados aunque no produjeron un reemplazo aprobado.

## Qué se revisó y de dónde proceden las referencias

Se inspeccionaron visualmente los 16 cuartos de las cuatro áreas de entrenamiento de Chullo, comparando la captura RGB original con los polígonos v1. No se utilizaron predicciones para dibujar o corregir etiquetas. El objetivo fue identificar referencias sobre vías, posibles patios, vegetación, sombras y estructuras cuya cubierta no podía distinguirse con confianza.

La captura de Google Earth Pro / Airbus tiene fecha **2026-03-04**. El TIFF de trabajo conserva tres bandas RGB, cuadrícula de 0.5 m y CRS EPSG:32719. Para inspeccionar límites se reproyectó el JPEG original mediante su transformación afín; no se generó detalle con IA. Ni la cuadrícula ni la ampliación demuestran la resolución nativa del sensor o la exactitud topográfica.

Las referencias son **cubiertas visibles**, no viviendas, propiedades, ocupación o linderos catastrales. La procedencia permanece explícita: `annotation_method=assistant_visual_manual` y `human_reviewed=false`. La revisión fue realizada por un asistente de IA, sin validación humana o de campo. Todas las áreas comparten la misma captura; la separación geográfica no crea una adquisición independiente.

Las copias revisadas están en `data/training/chullo_local_v2/train_*.pixels.json`. Se conservaron v1 y sus referencias de validación y prueba; sus hashes se comprobaron antes y después de la auditoría. El registro por ID, las razones y las imágenes de evidencia están en `outputs/label_audit_v2/audit.json`, `AUDITORIA_ETIQUETAS.md` y `*_v2_review.png`.

## Correcciones y conteos de entrenamiento

Se retiraron dos referencias de `train_1100_500`: **r26**, situada sobre calzada y un alcorque de árbol, y **r83**, sobre calzada con vehículos estacionados. Sus IDs y geometrías se conservan en `retired_polygons`; esas superficies reconocibles quedan como fondo.

Se añadieron exclusiones completas para diez instancias ambiguas:

- `train_1600_100`: r19 y r50.
- `train_1100_500`: r05, r30, r43 y r68.
- `train_1600_500`: r06, r34, r35 y r67.

No se recortó la parte aparentemente visible de esas instancias para convertirla en un techo nuevo. Sus contornos originales permanecen en las referencias, con una región `ignore` que cubre la instancia completa. Tampoco se redibujaron límites o huecos inciertos para favorecer las métricas. Las observaciones sin cambio sobre r11/r16/r63 de `train_1100_100` y r14 de `train_1100_500` quedan registradas.

| Área padre de entrenamiento | Polígonos conservados | Regiones ignore | Máscaras realmente supervisadas |
|---|---:|---:|---:|
| `train_1100_100` | 67 | 0 | 67 |
| `train_1600_100` | 61 | 5 | 58 |
| `train_1100_500` | 76 | 12 | 64 |
| `train_1600_500` | 67 | 8 | 61 |
| **Total** | **271** | **25** | **250** |

Los 271 polígonos proceden de 273 referencias originales menos las dos retiradas. Las 25 regiones de exclusión incluyen 15 previas y diez nuevas. El conteo de **250 targets** procede de la carga real del dataset después de rasterizar y excluir instancias completas; no es el conteo de uniones o contactos vectoriales. No debe describirse como 250 casas verificadas.

V2 utiliza los cuatro padres completos de **192×192 píxeles**, incluido `train_1100_500`, que v1 no podía aprovechar mediante recortes sin exclusiones. Se añaden tres escenas negativas de entrenamiento, de 192×192, 96×96 y 160×160, identificadas visualmente como terreno, cauce y parque sin cubiertas anotables. Estas escenas tienen cero referencias y enseñan fondo; su revisión también es del asistente. En total hay siete escenas de entrenamiento.

## Cómo se excluyen las zonas inciertas

El uso de exclusiones durante entrenamiento exige dos opciones coordinadas: `BuildingDataset(..., allow_training_ignore=True)` y `enable_training_ignore(model)`. El flujo anterior sin esta opción conserva su comportamiento.

1. Se rasterizan las referencias y regiones ignore en la cuadrícula de la imagen. Si una referencia contiene un píxel no supervisable, se elimina **toda** esa instancia de los targets y se añade su máscara completa a la exclusión. La exclusión se propaga si otra referencia se superpone. Las instancias de menos de cuatro píxeles se ignoran en lugar de convertirse en fondo.
2. El RGB original permanece intacto. Rotaciones, reflejos y redimensionamiento mantienen alineadas imagen, referencias y máscara de supervisión.
3. RPN y las cabezas de propuestas no toman como ejemplos negativos las cajas que toquen píxeles inciertos. Los positivos proceden de referencias completas conservadas.
4. La pérdida de máscara solo utiliza muestras de la región de interés totalmente respaldadas por píxeles supervisables. Así, una zona incierta no aporta un objetivo negativo a esa pérdida.

La regla de intersección de cajas es conservadora y puede descartar muchos ejemplos de fondo. No garantiza por sí sola menos falsos positivos o fusiones. En evaluación se usa el comportamiento de inferencia de Torchvision; las claves y dimensiones de los pesos se conservan.

Las once pruebas sintéticas CPU de `tests/test_training_ignore.py` pasaron: alineación, exclusión de instancias completas, pérdida y gradiente, escenas vacías, equivalencia con Torchvision sin ignore e invariancia de la inferencia. Esa revisión independiente no ejecutó CUDA durante el entrenamiento activo. El código adaptado conserva la licencia original en [LICENSES/torchvision-BSD-3-Clause.txt](../LICENSES/torchvision-BSD-3-Clause.txt).

## Separación de desarrollo y prueba reservada

El manifiesto `data/training/chullo_local_v2/development_manifest.json` incluye únicamente entrenamiento y validación. Su SHA256 congelado es `d779cf316f11f48363f51211f814d8361102c76c35aa89976e542194e84ca4de`. El preparador verifica fuente, referencias, exclusiones y hashes, y rechaza sobrescribir un manifiesto congelado.

La validación tiene **tres escenas y 70 referencias utilizables**: las dos áreas v1 con 30 y 40 cubiertas, más una escena negativa. Sus referencias no se redibujan según los resultados. Las regiones ignore se aplican a referencias y predicciones al medir. Reutilizar este conjunto para varias rondas y umbrales aumenta el riesgo de ajustar las decisiones a esa validación; no equivale a evidencia independiente.

La prueba histórica de v1 ya se evaluó en el experimento anterior. Para v2 se fijó un nuevo conjunto reservado: dos áreas geográficas nuevas, con **19 y 34 cubiertas utilizables**, y una escena negativa. Las dos áreas anotadas tienen siete y seis regiones ignore, que excluyen aproximadamente 11.17 % y 8.50 % de sus superficies. Se anotaron visualmente sin consultar predicciones y se congelaron en `data/training/chullo_local_v2_holdout`; el registro está en `outputs/label_audit_v2/holdout_freeze.json`.

No hay cruces entre particiones. El inventario completo registra separaciones mínimas entre límites de 74 m entrenamiento–validación, 214 m entrenamiento–prueba y aproximadamente 154.05 m validación–prueba, incluyendo escenas negativas. Estos márgenes evitan superposición, pero no eliminan correlación espacial ni el origen común de la captura.

**El nuevo test no se utiliza durante el entrenamiento, la selección de época o la elección de umbrales.** Solo corresponde evaluarlo al terminar las rondas y fijar una decisión. Las métricas de esa evaluación, si se realiza, se documentarán sin volver a elegir el candidato a partir de ellas.

## Qué significa superar el criterio de validación

Se comparan máscaras de instancias mediante emparejamiento uno a uno, voraz por mayor IoU, con **IoU ≥ 0.5**. TP son referencias emparejadas, FP predicciones sin pareja y FN referencias sin pareja. Se agregan conteos de todas las escenas antes de calcular precisión, sensibilidad, F1 y F0.5. No son métricas COCO AP.

El diagnóstico de fusión cuenta una predicción cuando cubre al menos el 50 % de **dos o más referencias distintas**. También registra cuántas referencias quedan afectadas, sin duplicarlas dentro de cada escena. Esta medida depende de la definición provisional de cubierta; no identifica viviendas fusionadas con certeza catastral.

La comparación principal es el modelo local v1 con escala 1.5625, umbral de puntuación 0.5 y umbral de máscara 0.5. En las tres escenas de validación registra 37 TP, 68 FP, cuatro predicciones fusionadas y cinco referencias afectadas. Para ser elegible, un candidato debe cumplir simultáneamente:

- **TP ≥ 37**: no perder aciertos frente a esa referencia de comparación.
- **FP < 68**: reducir falsos positivos.
- **Fusiones ≤ 4** y **referencias afectadas ≤ 5**: no empeorar estos dos diagnósticos.

Entre candidatos elegibles se prioriza F0.5, que da más peso a la precisión; los desempates consideran referencias afectadas, FP y sensibilidad. Un valor F1 mayor por sí solo no aprueba el candidato. Los umbrales de puntuación son parámetros del detector, no probabilidades calibradas de que exista una vivienda.

## Rondas realizadas

Todas las rondas mantienen el criterio anterior, usan semilla 42, batch 1, AdamW, precisión mixta durante entrenamiento, recorte de gradiente a 1 y rotaciones/reflejos. Las estadísticas de BatchNorm permanecen congeladas. Para evaluar cada candidato se convierten sus parámetros a FP16 en memoria y se recargan esos valores en un evaluador FP32. Solo se escribe un checkpoint si cumple el criterio; en estas rondas no ocurrió. La selección solo observa validación.

| Ronda | Inicialización y cambios | Presupuesto | Estado observado |
|---|---|---|---|
| `from_usa` | Checkpoint USA; contexto completo, anchors pequeños y adaptación de `layer4`; escala 1.5625 | Tres experimentos de 80 épocas | Completada; ninguno cumplió el criterio |
| `warm_start` | Checkpoint Chullo v1; ajuste conservador y adaptación de `layer4`; escala 1.5625 | Dos experimentos de 60 épocas | Completada; ninguno cumplió el criterio |
| `detail_scale` | Checkpoint USA; anchors habituales frente a pequeños; escala 3.125 | Dos experimentos de 80 épocas | Completada; ninguno cumplió el criterio |

La primera ronda usa tasa de aprendizaje de `1e-4` para las cabezas y `1e-5` para `layer4` cuando se habilita después de la época 10. La segunda reduce esas tasas a `2e-5` y `2e-6`, respectivamente. Los demás parámetros del cuerpo ResNet permanecen congelados. Los resultados completos de ambas rondas constan en sus protocolos, con `validation_gate_passed=false` y `selected=null`.

La última ronda volvió al checkpoint USA con cabezas a `1e-4` y el cuerpo ResNet congelado. Comparó anchors `[32,64,128,256,512]` y `[16,32,64,128,256]`, y revisó candidatos cada cinco épocas. Exploró puntuaciones **0.25, 0.35, 0.5, 0.65, 0.8 y 0.9**, combinadas con umbrales de máscara **0.5 y 0.6**. Ampliar de escala 1.5625 a 3.125 presenta techos mayores a la red —un padre de 192 pasa de 300 a 600 píxeles—, pero **no añade información nueva a la imagen**.

La referencia principal conservó escala 1.5625. También se calculó un control de los checkpoints previos a escala 3.125 para distinguir el efecto de ampliar la entrada del efecto del ajuste de pesos. No se cambió el criterio por el resultado de ese control. Las rondas posteriores fueron decisiones de desarrollo informadas por validación; no se presentan como un único experimento fijado antes de observar cualquier resultado.

Los registros se encuentran en `outputs/training/chullo_refinement_v2/`, `chullo_refinement_v2_warm/` y `chullo_refinement_v2_detail/`. Los comandos se ejecutan desde la raíz con `scripts/refine_chullo_model.py`, indicando `--suite from_usa`, `--suite warm_start` o `--suite detail_scale`, el presupuesto de épocas y un directorio de salida nuevo. No se debe sobrescribir evidencia de una ronda anterior.

## Resultados finales y decisión

Se completaron **520 épocas y 3640 actualizaciones del optimizador**, sobre siete escenas por época. Los registros por época indican **cero pasos omitidos por AMP**: 1680 actualizaciones en `from_usa`, 840 en `warm_start` y 1120 en `detail_scale`. Los tres protocolos terminaron con `selected=null` y `validation_gate_passed=false`.

La tabla muestra la referencia principal, su control ampliado y el punto con mayor F1 observado en cada ronda. Los tres últimos renglones son un **resumen exploratorio de validación**, no checkpoints seleccionados o aprobados. Todos usan puntuación 0.5 y máscara 0.5, aunque las rondas evaluaron la rejilla completa descrita antes.

| Modelo o punto observado | Escala | TP | FP | FN | Precisión | Sensibilidad | F1 | F0.5 | Fusiones | Referencias afectadas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chullo v1, referencia principal | 1.5625 | 37 | 68 | 33 | 0.3524 | 0.5286 | 0.4229 | 0.3776 | 4 | 5 |
| Chullo v1, control ampliado | 3.125 | 10 | 115 | 60 | 0.0800 | 0.1429 | 0.1026 | 0.0877 | 0 | 0 |
| `from_usa/full_context`, época 15 | 1.5625 | 31 | 29 | 39 | 0.5167 | 0.4429 | 0.4769 | 0.5000 | 1 | 2 |
| `warm_start/local_conservative`, época 38 | 1.5625 | 33 | 33 | 37 | 0.5000 | 0.4714 | 0.4853 | 0.4940 | 3 | 6 |
| `detail_scale/small_anchors`, época 30 | 3.125 | 33 | 41 | 37 | 0.4459 | 0.4714 | 0.4583 | 0.4508 | 0 | 0 |

Los puntos exploratorios redujeron FP, pero también perdieron aciertos frente a los 37 TP exigidos. El punto mostrado de `warm_start` además afectó seis referencias mediante fusiones, por encima del máximo de cinco. Que F1 o precisión aumenten en algunos puntos no satisface el criterio conjunto. El control ampliado también muestra que aumentar la escala de un checkpoint existente puede empeorar la detección; no es una mejora automática.

**Decisión: conservar Chullo v1 en la aplicación y no evaluar el nuevo test en esta ronda.** No se obtuvo un checkpoint seleccionado v2 ni una nueva métrica de prueba que demuestre mejora. Las cifras del test histórico v1 mantienen su significado anterior; estos resultados de validación no las actualizan ni las sustituyen.

La evidencia detallada permanece en los tres directorios de `outputs/training` indicados arriba. El [resumen portable](../reports/training/chullo_refinement_v2/resumen.json), las copias exactas de los tres registros y la [auditoría de etiquetas](../reports/training/chullo_refinement_v2/label_audit/AUDITORIA_ETIQUETAS.md) están incluidos en el repositorio. Los valores de la tabla proceden de `epochs.json` y `baselines_validation.json`, no de nuevas inferencias o de una evaluación del test. Como ningún punto cumplió el criterio, no se guardó ni registró un checkpoint v2; los máximos exploratorios no son modelos disponibles en la aplicación.

Para repetir los experimentos desde los datos congelados incluidos, usar carpetas de salida nuevas:

```powershell
.venv-ml\Scripts\python.exe scripts/refine_chullo_model.py --suite from_usa --epochs 80 --output outputs/training/repeticion_usa
.venv-ml\Scripts\python.exe scripts/refine_chullo_model.py --suite warm_start --epochs 60 --output outputs/training/repeticion_local
.venv-ml\Scripts\python.exe scripts/refine_chullo_model.py --suite detail_scale --epochs 80 --output outputs/training/repeticion_escala
```

Se requiere el entorno de IA con CUDA y los checkpoints iniciales con sus hashes fijados; el entrenamiento no los descarga automáticamente. Las semillas y los datos están fijados, pero no se promete igualdad bit a bit entre ejecuciones CUDA. `scripts/evaluate_chullo_refinement.py` separa la congelación del protocolo y la evaluación final: rechaza una corrida sin candidato elegible antes de acceder al test. No se ejecutó esa evaluación en esta revisión.

El trabajo deja etiquetas revisadas, soporte explícito de exclusión y experimentos reproducibles, pero no demuestra una mejora suficiente para reemplazar el modelo. Más referencias humanas, otras capturas y una evaluación externa serían necesarias para sostener afirmaciones más amplias de precisión.

La atribución de las imágenes es **Google Earth · Image © 2026 Airbus · captura 2026-03-04**. No se declara una licencia abierta ni una autorización específica confirmada para entrenamiento; conservar atribución no equivale a obtener esa autorización. La licencia de Torchvision corresponde al código y no cambia la procedencia de las imágenes. Ninguna métrica de detección establece peligro hidráulico, habitabilidad o permiso para construir.
