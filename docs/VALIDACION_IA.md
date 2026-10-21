# Validación del flujo de imágenes e IA

Fecha: 14 de octubre de 2026. **La inferencia sobre imagen real de Chullo ya está integrada y comprobada**, además del caso de Mariano Melgar. La auditoría de adquisición está en `data/imagery/acquisition_report.json`.

## Verificación de modularización — 21 de octubre de 2026

Se separaron interfaz, servicios de aplicación, repositorios y etapas del motor IA. `app.py` conserva únicamente el punto de entrada. Los imports públicos anteriores continúan funcionando mediante fachadas. Se mantuvieron los parámetros científicos y las salidas de demostración.

- Suite del entorno del mapa: **67 pruebas aprobadas**; cuatro módulos se omiten porque requieren Rasterio, instalado en el entorno ML.
- Suite del entorno IA (motor, configuración, pipeline, georreferencia y CLI raster): **29 pruebas aprobadas**. Ocho pruebas de configuración también forman parte de la suite del mapa; los conteos por entorno no deben sumarse como pruebas distintas. Quedan avisos de deprecación de bibliotecas en operaciones afines y Shapely, sin fallos de pruebas.
- Ruff: lint y comprobación de formato correctos en `app.py`, `src/`, `tests/` y los dos CLI de ejecución/inspección.
- AppTest comprueba banda de 25 → 50 m, inventario → IA, cambios pendientes de umbral, fuentes y CSV/GeoJSON/ZIP. Los filtros de tabla no recortan las exportaciones. Se bloquea la inferencia durante estas pruebas y se verifica que los hashes de la corrida guardada permanezcan iguales.
- El núcleo espacial se ejecuta desde otro directorio, con las importaciones de Streamlit y PyTorch bloqueadas. Importar `app.py` no renderiza ni inicia procesos.
- Una nueva inferencia real sobre Chullo produjo **1006 contornos**, 1591 candidatos previos a supresión de duplicados y el mismo GeoJSON **byte por byte**, incluidos IDs, orden, atributos y geometrías. La vista previa también es idéntica. CUDA: 10,594 s medidos dentro del motor; no es latencia total del proceso ni garantía de rendimiento.
- SHA256 del GeoJSON antes y después: `67b5f825310fbcae5e7820c99d9525a5f1c1634800a7e1195b0f49492abb8073`. Parámetros, hash de imagen y hash de pesos coinciden. Evidencia local: `outputs/refactor/comparacion_ia.json` y `outputs/refactor/validacion_ia/`.
- Se corrigió un fallo previo de recuperación: una detección fallida ya no sustituye el manifiesto válido por `running` o `error`. Las tentativas se guardan aparte; la publicación respalda los archivos y restaura la versión anterior ante fallos de escritura manejados. Doce pruebas nuevas cubren primera corrida, actualización, inferencia fallida, salida parcial, respaldo fallido, publicación fallida, bloqueo y subproceso Windows. No se afirma recuperación transaccional ante cortes de energía.
- Navegador real: imagen Google Earth/Airbus, mapa y contornos visibles; **237** seleccionados, **27** dentro de 25 m y **210** fuera. Captura en `outputs/presentacion/prototipo_modularizado.png`.

Estas verificaciones comprueban funcionamiento y ausencia de regresiones. No miden precisión del detector en Arequipa ni convierten las bandas en evaluación de peligro.

## Chullo: captura Google Earth Pro e inferencia real

- Google Earth Pro 7.3.7.1327, vista cenital y norte arriba, relieve y capas adicionales desactivados. JPG original y copia con referencias: 8192 × 6390, cámara idéntica. Fecha del visor 3/4/2026 (MM/DD/YYYY), es decir 4 de marzo de 2026. Créditos: **Google Earth; Image © 2026 Airbus**. Evidencia en `data/imagery/chullo/earth_fecha_evidencia.png`.
- Cuatro esquinas del rectángulo KML ajustan la transformación afín; dos marcadores reservados la comprueban. RMSE de comprobación 0,227 m. Los controles proceden del mismo visor, no son puntos topográficos: **no determinan precisión absoluta ni exactitud de las distancias con OSM**.
- Imagen completa transformada, conservando créditos, a GeoTIFF RGB 2621 × 2045, EPSG:32719, cuadrícula 0,5 m. Resolución nativa del sensor no verificada. Originales, controles, residuos y hashes documentados en `data/imagery/chullo/`.
- Modelo real, 35 ventanas: 1006 contornos en toda la imagen; **237** intersectan el sector piloto. **27** dentro de 25 m / 210 fuera; **56** dentro de 50 m / 181 fuera. Distancias de 0 a 330,140 m.
- Auditoría independiente: distancias idénticas al motor; cero geometrías inválidas, vacías o duplicadas, ni IDs duplicados. Hashes de imagen, JPG original, puntos, pesos y GeoJSON correctos.
- AppTest ejecutó el botón real, cambió banda y alternó IA/inventario sin errores. El inventario Microsoft conserva sus 177 contornos y sus propios resultados.
- Overlay independiente `outputs/validacion/chullo_ia_auditoria_overlay.png`: varias cubiertas detectadas y eje aproximadamente sobre el cauce visible; numerosas omisiones y falsos positivos sobre una cancha, vegetación y patios. No es un censo de casas ni evaluación de peligro.
- Seis pruebas del nuevo ajuste geográfico pasan, incluyendo rechazo de errores de comprobación excesivos.
- Navegador: mapa e imagen visibles, créditos Google Earth/Airbus en la atribución y descarga real del CSV verificada con 237 filas, fecha 2026-03-04 y procedencia `geoai_building_inference`. Captura de la aplicación en `outputs/presentacion/prototipo_ia_chullo.jpg`.
- Validación automatizada total: 38 pruebas en el entorno del mapa y 12 del motor ML y georreferenciación en el entorno de IA; 50 aprobadas entre ambos entornos. Los dos módulos ML se omiten en el entorno que solo contiene las dependencias del mapa.

## Caso adicional de Mariano Melgar

El flujo completo se verificó adicionalmente con una imagen real de Mariano Melgar, Arequipa. La interfaz, las fuentes, los nombres de archivos y el mapa señalan que es otro sector. El inventario de Chullo no se presenta como una detección del nuevo modelo.

## Imagen y modelo

- Fuente: [CENEPRED / SIGRID, documento 6798](https://sigrid.cenepred.gob.pe/sigridv3/documento/6798).
- Ortofoto de dron del 5 de junio de 2018. No es imagen satelital ni representación actual.
- Georreferencia nativa UTM 19S conservada. Recorte de 400 × 400 m, 800 × 800 píxeles RGB, resolución de trabajo 0,5 m; original 0,1093 m. No se usó superresolución.
- SHA256 de imagen: `f90f882e7f80ed75c920836dd78d8f16130e0d51506881462c69aa42ebc8ea14`.
- Pesos preentrenados Mask R-CNN ResNet50-FPN de `giswqs/geoai`; carga estricta de las 307 claves del checkpoint y hash verificado.
- No hay entrenamiento local. Los contornos de Microsoft no se utilizan como salida del modelo ni como etiquetas de validación.

## Comprobaciones funcionales

1. Inferencia real en GPU sobre cuatro ventanas de la imagen, con solapamiento. El modelo generó 263 candidatos; la depuración de solapamientos conservó 180 contornos, con umbral 0,50 y área mínima de 10 m².
2. El cliente valida imagen, sector y estado de ejecución; comprueba hash y conteo del GeoJSON antes de cargarlo. No sustituye una detección vacía por inventarios existentes.
3. Los contornos se georreferencian desde máscaras, no desde rectángulos. El análisis calcula distancias mínimas a dos ejes locales de OpenStreetMap en EPSG:32719.
4. Los ejes fueron superpuestos a la ortofoto para verificar su correspondencia general con el cauce. No se midió precisión topográfica ni se confirmó un nombre oficial; se muestran como torrentera y ramal locales.
5. Resultado: 180 contornos; 19 dentro de 25 m; 43 dentro de 50 m. Distancias de 5,679 a 242,033 m. No se interpretan como peligro ni faja marginal.
6. AppTest ejecutó el botón del modelo sin simulación, cambió bandas y sector, y comprobó ausencia de excepciones. Chullo conserva sus 177 contornos de inventario.
7. El navegador ejecutó una nueva inferencia y mostró el resultado. Se prepararon CSV, GeoJSON, ZIP con procedencia y mapa estático de respaldo.
8. Pruebas automatizadas: 38 del mapa/cliente/exportación y 6 del motor ML, distribuidas en sus entornos Python.

## Revisión visual y límites

Se revisaron la imagen de entrada, la superposición de máscaras (`outputs/inference/mariano_melgar_demo/qa_overlay.png`) y el mapa con ejes y distancias. Hay muchas cubiertas alineadas, pero también omisiones en manzanas densas y falsos positivos en patios o calles. Un umbral 0,80 redujo el conteo a 135 sin eliminar todos los errores; no se presenta como una calibración validada.

El modelo marca 12 contornos que llegan al borde de la imagen y 7 que tocan un borde interno de ventana. Las primeras geometrías pueden estar incompletas. Estos indicadores se conservan en los resultados.

No hay etiquetas manuales independientes ni evaluación de campo; no se reporta exactitud, precisión, recall o IoU sobre Arequipa. Tampoco se identifica uso residencial, ocupación o seguridad de las construcciones. El ejemplo verifica funcionamiento y trazabilidad del procesamiento.
