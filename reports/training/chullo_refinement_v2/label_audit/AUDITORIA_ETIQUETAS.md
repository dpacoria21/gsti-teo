# Auditoría de etiquetas Chullo v2

Auditoría visual realizada por el asistente sobre la imagen original y las anotaciones v1; **no es revisión humana**. Se inspeccionaron los 16 cuartos de los cuatro padres de entrenamiento. No se usaron predicciones ni se modificaron validación, prueba o v1.

Se conservó la geometría y los IDs de las instancias inciertas y se añadió ignore sobre la instancia completa. Dos referencias sobre calzada se retiraron de las referencias activas y se conservaron en `retired_polygons`. No se crearon techos parciales ni se cambiaron huecos o linderos.

| Padre | Referencias conservadas | Retiradas | Ignore original → v2 | Candidatas sin contacto vectorial con ignore |
|---|---:|---:|---:|---:|
| train_1100_100 | 67 | 0 | 0 → 0 | 67 |
| train_1600_100 | 61 | 0 | 3 → 5 | 56 |
| train_1100_500 | 76 | 2 | 8 → 12 | 63 |
| train_1600_500 | 67 | 0 | 4 → 8 | 58 |

Los conteos vectoriales no son los targets finales: rasterización, nodata y la exclusión íntegra al tocar ignore pueden reducirlos. El preparador de v2 debe obtener y registrar el conteo real del dataset.

## Evidencia por referencia

- **train_1100_100/r11 — retain_with_limitation**: Franja de cubierta clara junto a sombra profunda. El límite exacto es incierto; la sombra no demuestra ausencia de techo. Se conserva la instancia sin inventar un lindero. Evidencia: `train_1100_100_q1.png`.
- **train_1100_100/r16 — retain_with_limitation**: Plano de cubierta visible junto a franja oscura longitudinal; posible fachada o sombra en el borde. Se conserva la geometría aproximada sin crear una instancia parcial. Evidencia: `train_1100_100_q1.png`.
- **train_1100_100/r63 — retain_with_limitation**: Cubierta grande continua. El hueco interior podría ser patio o volumen elevado; la imagen no permite resolverlo. Se preservan el contorno y el hueco originales; no se divide por tamaño. Evidencia: `train_1100_100_q3.png`.
- **train_1600_100/r19 — ignore_entire_instance**: La punta superior incluye superficie verde junto al plano blanco inferior; no se distingue patio/vegetación de terraza o cubierta. No se transforma la porción visible en un objeto nuevo. Evidencia: `train_1600_100_q2.png`.
- **train_1600_100/r50 — ignore_entire_instance**: Pequeña zona junto a la calle con vegetación, sombra y objetos; el RGB no sostiene una cubierta separada con límites fiables. No se afirma identidad de vehículo. Evidencia: `train_1600_100_q3.png`.
- **train_1100_500/r05 — ignore_entire_instance**: Vegetación o cavidad entre muros y superficies claras; no se confirma una cubierta opaca individual. Se ignora toda la instancia original. Evidencia: `train_1100_500_q1.png`.
- **train_1100_500/r14 — retain_with_limitation**: Bandas paralelas y plano gris coherentes con una cubierta corrugada. Se conserva; no se reclasifica como patio solo por su textura. Evidencia: `train_1100_500_q1.png`.
- **train_1100_500/r26 — retire_clear_nonroof**: El polígono está sobre la calzada continua entre fachadas e incluye el alcorque de un árbol; no contiene un plano de cubierta. Se retira como referencia de techo y queda como fondo visible. Evidencia: `train_1100_500_q2.png`.
- **train_1100_500/r30 — ignore_entire_instance**: Pequeño recinto con sombras y superficies de diferentes alturas al borde de la vía; no se distingue techo de patio descubierto con fiabilidad. Evidencia: `train_1100_500_q2.png`.
- **train_1100_500/r43 — ignore_entire_instance**: Franja estrecha blanca y oscura entre cubiertas y vía; la proyección oblicua impide separar fachada, sombra y cubierta. Se conserva el contorno original bajo ignore. Evidencia: `train_1100_500_q3.png`.
- **train_1100_500/r68 — ignore_entire_instance**: Pequeña cavidad verde/oscura entre cubiertas; no hay evidencia suficiente de un techo individual. La instancia entera se ignora. Evidencia: `train_1100_500_q4.png`.
- **train_1100_500/r83 — retire_clear_nonroof**: El contorno cruza calzada junto al bordillo y vehículos estacionados visibles. No es una cubierta fija; se retira como referencia, conservando ID y geometría en el historial. Evidencia: `train_1100_500_q4.png`.
- **train_1600_500/r06 — ignore_entire_instance**: Parte inferior con vegetación u oclusión junto a plano gris superior; continuidad de cubierta incierta. Se conserva la instancia completa bajo ignore, sin recortar un nuevo techo parcial. Evidencia: `train_1600_500_q1.png`.
- **train_1600_500/r34 — ignore_entire_instance**: Estructura lineal con elementos paralelos junto a la calle; el RGB no distingue cubierta continua, estructura abierta en construcción o fachada. Se ignora la instancia completa. Evidencia: `train_1600_500_q1.png`.
- **train_1600_500/r35 — ignore_entire_instance**: Estructura lineal contigua a r34 con huecos oscuros; no se confirma cubierta opaca independiente. No se fusiona ni se convierte en objeto parcial. Evidencia: `train_1600_500_q1.png`.
- **train_1600_500/r67 — ignore_entire_instance**: Predomina sombra/posible vegetación junto al parque y muro; no se confirma un techo separado. Se conserva su geometría original bajo ignore. Evidencia: `train_1600_500_q4.png`.

## Alcance y límites

Las etiquetas describen cubiertas visibles, no viviendas, ocupación, propiedad o linderos catastrales. Las cubiertas contiguas y los patios interiores siguen teniendo ambigüedad. La auditoría es provisional, deriva de una sola captura y no demuestra mejora del modelo: eso debe medirse después con el protocolo congelado. Las revisiones no se hicieron a partir de falsos positivos del test.

Se verificaron hashes de todos los archivos de v1 antes y después, validez geométrica y ausencia de solapes entre referencias mayores de 1 px². Los contornos/huecos conservados no cambiaron. El detalle original se reproyectó de forma afín para inspección; no es superresolución ni generación de imagen.

Google Earth · Image © 2026 Airbus · captura 2026-03-04. La atribución no constituye licencia abierta ni confirmación de autorización para entrenamiento.
