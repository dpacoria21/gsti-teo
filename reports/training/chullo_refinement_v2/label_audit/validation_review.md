# Auditoría de las referencias locales v1 y propuesta de evaluación v2

Estado: revisión de solo lectura de imágenes, etiquetas, código y resultados ya existentes. No se modificaron las referencias v1, no se entrenó ni se ejecutó nueva inferencia. La revisión visual es del asistente, no una certificación humana.

## Dictamen

Las referencias v1 sirven para una demostración técnica provisional, pero **no son suficientemente fiables para presentar las métricas como exactitud confirmada de edificios o viviendas**. No hay solapes geométricos grandes; esa consistencia no garantiza la interpretación semántica de cada superficie. La captura tiene perspectiva, sombras y pequeños elementos que se pueden confundir con cubiertas. Encontré un posible vehículo etiquetado como cubierta y varias instancias parcialmente excluidas por áreas ambiguas. No convertí estas observaciones en nuevas etiquetas ni alteré las métricas publicadas.

El aumento de F1 local procede de una recuperación mayor de referencias, junto con más falsos positivos y más uniones respecto al original USA. La reducción de uniones frente al ajuste SpaceNet no demuestra que se hayan resuelto las uniones frente al original. Son comparaciones distintas.

## Evidencia geométrica reproducible

`geometry_audit.json` registra la auditoría por ID: área rasterizada, área evaluable, fracción excluida, caja y solapes. Se empleó rasterización por centro de píxel, con la misma convención de etiquetas e ignore del cargador.

| Escena | Polígonos originales | Instancias evaluables (>=4 píxeles) | Área excluida por ignore |
| --- | ---: | ---: | ---: |
| val_1100_900 | 30 | 30 | 11.84 % |
| val_1600_900 | 40 | 40 | 8.24 % |
| test_1100_1400 | 51 | 51 | 6.69 % |
| test_1600_1400 | 70 | 69 | 7.39 % |

En las cuatro escenas no hay intersecciones entre etiquetas mayores de 1 píxel cuadrado. Solo un par comparte un píxel rasterizado: `val_1600_900/roof_022` y `roof_023`, intersección vectorial 0.713731 píxeles cuadrados. Las intersecciones subpíxel restantes son `val_1600_900/roof_043`–`roof_044` (0.003513), `test_1600_1400/...roof_020`–`...roof_021` (0.040480) y `...roof_033`–`...roof_037` (0.227586); ninguna comparte píxeles rasterizados. Son incidencias menores, no una explicación suficiente de los errores del modelo.

El benchmark evaluó 70 referencias de validación y 120 de prueba. `test_1600_1400_roof_019` está completamente dentro de ignore y se elimina: tiene 0 píxeles evaluables. Esto explica la diferencia entre las 121 referencias originales de test y las 120 utilizadas.

## Hallazgos visuales por ID

Todos los límites siguientes son coordenadas locales del recorte de 192 × 192. Se inspeccionó primero el RGB y las referencias; no se usaron predicciones para decidir ninguna etiqueta.

| ID / área | Evidencia observada | Interpretación de la auditoría |
| --- | --- | --- |
| `test_1600_1400_roof_013`, x=183..191, y=64..70 | Objeto blanco aislado en la superficie de la vía, por debajo de la fila edificada y con apariencia compatible con un vehículo. El polígono cubre solo una parte del objeto blanco visible. Véase `test_1600_1400_roof013_review.png`. | **Posible error semántico importante.** La apariencia favorece vehículo frente a cubierta, pero requiere revisión humana/imagen más clara. No se elimina ni se modifica v1 después de conocer los resultados. |
| `test_1600_1400_roof_012`, x=182..192, y=50..58 | La anotación describe una cubierta clara estrecha; el RGB muestra una franja principalmente oscura junto al borde edificado y la vía. | Superficie no resuelta: sombra, vegetación, elemento de fachada o cubierta. La nota no basta para confirmar la clase. Revisar en una futura versión de referencias, sin corregirla por lo que predice un modelo. |
| `val_1600_900/roof_015`, x=67..81, y=126..143 | El 43.7 % de la máscara cae en ignore. | La instancia evaluada es solo un fragmento de la superficie anotada. No equivale a verificar el contorno completo de esa cubierta. |
| `val_1600_900/roof_016` y `roof_033` | El 29.2 % de cada máscara cae en ignore. En `roof_033` se observan sombra y planos contiguos en el extremo occidental. | Límite físico e interpretación de la instancia requieren revisión. La operación de recorte está implementada, pero condiciona qué parte se premia/penaliza. |
| `val_1100_900`, x=31..45, y=123..133, sin ID | Pequeña superficie gris clara dentro de vegetación, sin etiqueta ni ignore. | **Pregunta para revisión**, no omisión confirmada: puede ser cubierta pequeña, pérgola o superficie abierta. No declarar todo píxel sin etiqueta como negativo fiable sin inspección exhaustiva. |

Los overlays completos son `val_1100_900_labels.png`, `val_1600_900_labels.png`, `test_1100_1400_labels.png` y `test_1600_1400_labels.png`. Verde: referencias congeladas; rojo: ignore. Se mantienen los IDs originales. No son imágenes de predicciones.

No se cuantifica una tasa de error de las etiquetas con esta revisión: faltan un segundo anotador independiente y una adjudicación humana. Tampoco se da por correcto un polígono simplemente por ser válido o no superponerse.

## Qué mide el código y sus límites

1. `src/training/dataset.py` conserva RGB, rasteriza referencias y excluye ignore de la máscara evaluable. `runner.evaluate_model` aplica esa misma máscara a las predicciones y retira las que quedan vacías. Hay simetría en la exclusión; no observé un tratamiento distinto de nodata entre referencias y predicciones.
2. El recorte de regiones ignoradas cambia el objeto medido: una predicción que sobrepasa una cubierta hacia ignore puede perder ese exceso antes de calcular IoU; una máscara puede quedar fragmentada y seguir contando como una instancia. Es una convención explícita, no una prueba de que el contorno completo esté bien. Para v2 se debe fijar esta convención antes de evaluar y publicar también cuánto área/instancias se excluyen.
3. `match_masks` usa emparejamiento voraz por mayor IoU, uno a uno, con umbral 0.5. Precisión, recall y F1 se agregan por conteos; no son AP. Duplicados cuentan como FP. No observé una fórmula incorrecta de F1 en los resultados guardados.
4. `merged_instances` cuenta una predicción que cubre >=50 % de al menos dos referencias. Variaciones en la forma de dividir las cubiertas modifican este diagnóstico. No cuenta viviendas ni propiedades fusionadas.
5. Hay un máximo de 100 detecciones por imagen. Ninguna de las dos imágenes val ni test supera ese número de referencias y el candidato publicado produjo menos de 100 predicciones por imagen; no aparece como límite activo en estos resultados.
6. El factor 1.5625 mantiene la escala aproximada de un mosaico de producción de 512 → 800 píxeles. La diferencia relevante restante es el contexto de recortes pequeños, los mosaicos solapados y los filtros de área/duplicados de producción. No atribuir todos los errores a un simple cambio de escala ni comparar métricas de recortes como si fueran métricas del flujo completo.
7. La selección por validación se mantiene separada del test. La validación del checkpoint empaquetado fue 43.02 % frente a 44.19 % durante selección; el test corresponde al archivo empaquetado. Esa diferencia está declarada.
8. Dos escenas test de una sola captura no sostienen un intervalo de generalización fiable. El bootstrap por imagen no incorpora la dependencia de la captura ni la incertidumbre de anotación. El test v1 ya se ha inspeccionado: no debe convertirse en un objetivo de selección repetida para v2.

La figura `outputs/training_validation/chullo_local_v1_comparison.png` contiene las dos áreas de **validación**, usadas para elegir la época. La inspección confirma ese rotulado. No es una selección independiente ni una evaluación visual de test.

## Protocolo propuesto para el siguiente experimento

- Conservar intactos pesos, etiquetas, split y métricas v1. Las correcciones futuras de referencias deben crear otra versión y quedar adjudicadas sin mirar predicciones; si se reevalúa v1, informar ambas versiones y no reemplazar silenciosamente el resultado previo.
- Usar el holdout nuevo fijado por cobertura geográfica antes de evaluar, con IDs `holdout_300_1600` y `holdout_2100_1100`, más el negativo fijo que prepare el coordinador. Delinear todas las cubiertas visibles y marcar dudas como ignore, sin interpretar uso residencial ni inventar divisiones por propiedad.
- Congelar selección, anotaciones, regla de exclusión y hashes antes de cualquier evaluación nueva. Idealmente un revisor humano, que no vea predicciones, adjudica las dudas. Sin ese paso, el resultado se mantiene como piloto provisional del asistente.
- Incorporar negativos reales de entrenamiento elegidos en RGB por una cuadrícula/estratos geográficos: vías con vehículos, vegetación, suelo y otras superficies sin cubierta. Guardar GeoJSON vacío solo cuando el área completa esté revisada y se confirme ausencia de cubiertas; una región sin anotaciones no equivale automáticamente a negativo. No convertir predicciones o errores aparentes del modelo en etiquetas correctas.
- Los chips train deben estar completamente anotados. El cargador no implementa ignore en la pérdida de Mask R-CNN: no introducir regiones ambiguas incompletas en train. Permitir ejemplos negativos vacíos solo después de verificar el flujo con un caso real y una prueba pequeña de ejecución.
- Fijar primero arquitectura, presupuesto de épocas, esquema de aumentos y métrica de selección. Elegir checkpoint y, si se desea, umbral operativo exclusivamente en validación, incluyendo sus negativos. Evaluar una vez original, v1 y candidato final sobre exactamente el mismo holdout nuevo y checkpoint empaquetado.
- Publicar TP/FP/FN, precisión, recall, F1, uniones, referencias afectadas, resultados por escena, área evaluable y FP por área negativa. Mostrar también los fallos. No declarar mejora uniforme si se gana recall a costa de precisión o de más uniones.
- Mantener separados los resultados de máscaras en recortes y los del flujo completo de producción. Una evaluación externa de verdad requiere imágenes de otras áreas/fechas/capturas con derechos de uso adecuados y referencias humanas, no solo más recortes de la misma imagen.

## Disjunción del holdout propuesto

Fuente: raster de 2621 × 2045 píxeles, EPSG:32719, píxel 0.50 m. Se compararon los rectángulos completos contra las ocho áreas originales v1, incluida la escena de entrenamiento excluida. Ambas ventanas están dentro de la imagen y tienen área de intersección cero con todas las áreas v1.

| ID | Ventana en la imagen original [x,y,ancho,alto] | Área v1 más cercana | Separación mínima |
| --- | --- | --- | ---: |
| `holdout_300_1600` | [300,1600,192,192] | `test_1100_1400` | 304.03 m |
| `holdout_2100_1100` | [2100,1100,192,192] | `val_1600_900` | 154.05 m |

La disjunción es geométrica. No convierte los recortes en capturas independientes ni garantiza generalización. No se consultaron las salidas de ningún modelo sobre esas ventanas para seleccionarlas o anotarlas.

## Integridad de v1

Hashes de los JSON auditados, conservados sin cambios:

- `val_1100_900.pixels.json`: `a05a98c32a098b818601814ddeb06852bd8ba255ea398bb6db3f5daddfabc7a5`.
- `val_1600_900.pixels.json`: `6f27d60ba38956c5b000a48b11091e7b02805d74f90763f39828e951d288c0cb`.
- `test_1100_1400.pixels.json`: `cc770977d3774425cb9ca6a59bf56fa87a755c5835b16e68564b94698210a306`.
- `test_1600_1400.pixels.json`: `7d6f42aac0210dfe052774ca9915b53223d533f9f5bc6be7d256afb1f6b30db9`.

Manifiesto v1: `0631962c46b7cdb2a6b3b17cdf1cd3afd32e22f09c1d0bf36090f989b841ef4e`. Los conteos publicados siguen intactos; este documento no sustituye esos resultados por una evaluación corregida a posteriori.
