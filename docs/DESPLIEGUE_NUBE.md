# Despliegue de la aplicación con reconocimiento de imágenes

El despliegue usa el mismo código modular, GeoTIFF y modelo verificado del prototipo. `cloud/app.py` prepara los pesos y configura el intérprete del worker antes de abrir Streamlit. La inferencia permanece en un proceso separado. No depende de que la laptop esté encendida.

## Aplicación desplegada y acceso

**[Abrir Chullo: acceso directo verificado](https://chullo-arequipa-dpacoria21.streamlit.app/~/+/).** El 21 de octubre de 2026 se guardó el acceso público solicitado por el titular y se verificó en **Settings → Sharing → This app is public and searchable**. Cualquier persona puede usar la aplicación sin iniciar sesión. El repositorio continúa privado; sus permisos son independientes de los de la aplicación.

La ruta directa permite usar el detector, las distancias, las exportaciones y el mapa interactivo con la imagen real. La [URL principal de Community Cloud](https://chullo-arequipa-dpacoria21.streamlit.app/) permanece en la pantalla de carga en Chrome; su causa no está confirmada. Se utiliza la ruta directa como acceso verificado mientras se resuelve esa limitación de la página principal.

## Semáforo desplegado y comprobado — 23 de octubre de 2026

El código incorpora un semáforo de proximidad con dos controles: **Límite de la zona roja (m)**, inicialmente 25 m, y **Ancho de la zona amarilla (m)**, inicialmente 25 m. El límite exterior es la suma de ambos valores y no supera los 100 m preparados para los sectores actuales. Modificarlos actualiza los colores, conteos, tabla y exportaciones sin volver a ejecutar la IA.

Se utiliza la distancia mínima, sin redondear, entre el contorno completo de la edificación y el cauce cartografiado más cercano, calculada en UTM 19S. Rojo corresponde a `d <= límite rojo`; amarillo a `límite rojo < d <= límite rojo + ancho amarillo`; verde a distancias mayores que el límite exterior. La franja amarilla se mide desde el límite rojo hacia fuera, a cada lado del eje.

El color expresa proximidad. **No clasifica peligro real ni autoriza construir; verde no significa seguro ni edificable.** Una evaluación de esos aspectos necesita análisis hidráulico, datos del terreno y capas oficiales pertinentes que el prototipo no incorpora.

El CSV y GeoJSON registran `proximity_class`, `proximity_label`, `red_band_m`, `yellow_width_m` y `band_m` (límite exterior combinado). El ZIP conserva además reglas, parámetros y conteos por categoría. Así se puede interpretar un archivo aunque después se cambien los controles.

La revisión **`5b726ef`** se publicó en GitHub y se comprobó en el alojamiento el **23 de octubre de 2026**. La ruta directa `/~/+/` cargó el mapa y los **237 contornos de la demo guardada de Chullo**. La prueba de los controles obtuvo:

| Límite rojo | Ancho amarillo | Límite exterior | Rojos | Amarillos | Verdes |
|---|---|---|---:|---:|---:|
| 25 m | 25 m | 50 m | 27 | 29 | 181 |
| 25 m | 26 m | 51 m | 27 | 30 | 180 |

Cambiar el ancho amarillo de 25 a 26 m reclasificó los mismos contornos sin ejecutar otra inferencia. Se descargó realmente desde el navegador el archivo `chullo_deteccion_semaforo_rojo_25m_amarillo_hasta_51m.zip`, conservado en Descargas. Se verificó que contiene `resumen_edificaciones.csv`, `relaciones_cauces.csv`, `edificaciones.geojson`, `metadatos.json` y `LEEME.txt`; sus metadatos registran `red_band_m=25`, `yellow_width_m=26`, `band_m=51` y los conteos rojo 27, amarillo 30 y verde 180.

La captura de evidencia local se conserva en `outputs/cloud_validation/semaforo.jpg`. Esta verificación acredita el semáforo alojado y su exportación sobre la demo guardada. Las pruebas de inferencia CPU y capturas del 21 de octubre documentadas abajo se mantienen como evidencia histórica de una ejecución distinta. La limitación de carga de la URL raíz continúa; el acceso comprobado usa la ruta directa.

## Streamlit Community Cloud

La aplicación se desplegó desde la cuenta conectada a GitHub con estos valores:

| Campo | Valor |
|---|---|
| Repository | `dpacoria21/gsti-teo` |
| Branch | `main` |
| Main file path | `cloud/app.py` |
| Python version, en Advanced settings | `3.12` |

Community Cloud encuentra `cloud/requirements.txt` junto al punto de entrada. Incluye PyTorch y torchvision CPU, las bibliotecas geográficas y Streamlit con versiones fijadas. `packages.txt` instala las bibliotecas de sistema `libgomp1` y `libexpat1`, necesarias para el runtime numérico y la lectura de GeoTIFF. No requiere una clave API de IA. El primer arranque descarga los pesos públicos desde una revisión concreta de Hugging Face y verifica su SHA256; los siguientes accesos en ese proceso reutilizan el archivo.

El entorno fija **Streamlit 1.66.0** y **PyArrow 24.0.0**. La versión de PyArrow coincide con la medida de compatibilidad aplicada por Community Cloud y vinculada a [apache/arrow#50471](https://github.com/apache/arrow/issues/50471). El input de nube toma las dependencias declaradas de `requirements.in` y su lock resuelve las dependencias Linux sin alterar el lock ni el entorno local.

Conservar el repositorio privado. El acceso a la aplicación se administra por separado en la configuración de Community Cloud y se abrió al público por solicitud del titular. Este cambio no concede acceso al repositorio ni modifica la conexión OAuth de GitHub. La conexión OAuth y los términos del proveedor los revisa el titular de la cuenta.

El alojamiento gratuito tiene recursos compartidos: comprobar una inferencia completa antes de considerar el despliegue funcional. Un mapa que abre no prueba que haya memoria suficiente para ejecutar Mask R-CNN. Si el proveedor termina el worker por falta de memoria, utilizar el contenedor en un servidor con capacidad suficiente; no sustituir silenciosamente la inferencia por resultados guardados.

## Visor del mapa en nube

Con `CHULLO_DEPLOYMENT=cloud`, `src/ui/map_panel.py` genera el HTML de Folium y lo presenta mediante `streamlit.components.v1.html`. El HTML de las capas se incluye directamente en el iframe, sin cargar el frontend específico de `streamlit_folium.st_folium`, cuyo timeout se observó en Community Cloud. Leaflet conserva dentro del mapa el desplazamiento, zoom, control de capas y fichas de los contornos; estas acciones no necesitan callbacks a Python. El selector de edificación de Streamlit continúa preparando el encuadre elegido.

El modo local conserva `st_folium`. Ambos modos utilizan `build_map`, las mismas geometrías y las mismas distancias; el cambio afecta a la presentación, no al detector ni al análisis espacial. Las teselas y bibliotecas del visor siguen requiriendo conexión. La revisión `d6b69cf` se comprobó visualmente en la ruta directa después de reiniciar el servidor: mostró la imagen real, los 237 contornos de la demo, el acercamiento a una edificación seleccionada y su ficha al pulsar el polígono.

## Actualizar código y reiniciar

Community Cloud recibe los cambios publicados en `main`, pero durante esta comprobación un cambio de código no renovó los módulos ya importados en el proceso. La configuración del proyecto utiliza `server.fileWatcherType = "none"`. Si los registros muestran la versión nueva pero la interfaz sigue mostrando el comportamiento anterior, realizar un reinicio explícito:

1. Descargar los resultados que se quieran conservar y esperar a que termine cualquier inferencia en curso.
2. Abrir **Manage app**, abrir el menú de opciones y elegir **Reboot app**; confirmar **Reboot**.
3. Esperar a que finalice la preparación y abrir una sesión nueva. Verificar el cambio concreto, además de que la página responda.

El reinicio interrumpe las sesiones activas y puede eliminar sus archivos temporales. No cambia los permisos de acceso de la aplicación ni del repositorio. El procedimiento del proveedor está en [Reboot your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/reboot-your-app).

## Contenedor Linux

Requiere Docker y conexión durante la construcción. El contexto excluye credenciales, entornos, cargas privadas y las capturas originales. Descarga el modelo durante la construcción, sin incorporarlo al repositorio Git.

```sh
docker build -t chullo-prototipo:cloud .
docker run --rm -p 127.0.0.1:8501:8501 --memory=4g --cpus=2 chullo-prototipo:cloud
```

Abrir `http://127.0.0.1:8501`. En un proveedor que ejecute contenedores, utilizar el mismo Dockerfile, configurar el puerto del servicio con `PORT` y publicar mediante HTTPS. El proceso escucha en `0.0.0.0:$PORT`; la imagen se ejecuta con un usuario sin privilegios. `/_stcore/health` sirve como comprobación de salud. El alojamiento debe admitir conexiones WebSocket de Streamlit.

4 GB es un margen inicial de capacidad, no una garantía para cualquier imagen o tráfico. La validación debe medir el uso real. Para un servicio de varias instancias, hacen falta una cola y almacenamiento compartido; el bloqueo actual coordina únicamente los procesos de una instancia.

## Sesiones y archivos

- Las demostraciones de Chullo y Mariano Melgar son entradas de lectura. Ejecutar el detector desde la interfaz crea una carpeta nueva por sesión y tentativa.
- El servidor permite una sola inferencia simultánea. Otra sesión recibe un aviso de que el motor está ocupado y puede reintentar al terminar.
- Las cargas se guardan en carpetas de sesión. El límite de subida es 64 MB; el contrato del raster valida dimensiones, resolución, canales y cobertura geográfica.
- El resultado se selecciona en esa sesión únicamente después de una ejecución completa. Un fallo conserva la selección anterior.
- Las cachés tienen límites de entradas y caducidad. No constituyen almacenamiento permanente.
- El disco del servicio puede ser efímero: descargar CSV, GeoJSON o ZIP antes de cerrar la sesión. Los reinicios pueden eliminar cargas y resultados nuevos; las demostraciones incluidas se restauran con el código.

## Comprobación del despliegue

La comprobación automatizada del contenedor puede ejecutarse con:

```sh
docker run --rm --memory=4g --cpus=2 chullo-prototipo:cloud python scripts/check_cloud.py --run-inference
```

Ejecuta la interfaz, solicita una detección real, valida distancias y exportaciones y comprueba que la demo incluida permanece intacta. No sustituye la revisión de la URL del proveedor.

1. Abrir la URL HTTPS, comprobar que el mapa muestra la demo y las fuentes.
2. Pulsar **Detectar en esta imagen** con Chullo y esperar a **Detección completada**. Verificar en la procedencia la nueva ejecución y el dispositivo CPU.
3. Comprobar los valores iniciales de 25 m de límite rojo y 25 m de ancho amarillo. Cambiar el ancho amarillo y comprobar que las distancias y los contornos se conservan mientras se actualizan directamente la franja, los colores y los conteos; no debe iniciarse otra inferencia. Verificar también el ajuste del límite rojo y el máximo exterior de 100 m.
4. Descargar CSV, GeoJSON y ZIP. Verificar que contienen todas las filas del resultado aplicado, los parámetros elegidos y las categorías correspondientes. Comprobar los límites exactos con las distancias sin redondear: igualdad con el límite rojo pertenece a rojo; igualdad con el límite exterior pertenece a amarillo. Contrastar los conteos del ZIP con las filas exportadas.
5. Abrir otra sesión y comprobar que sigue mostrando su propia selección. No debe reemplazarse la demo almacenada.

La evaluación sigue siendo de proximidad al eje cartografiado. El despliegue no incorpora simulación hidráulica, delimitación legal ni una validación independiente de exactitud. El semáforo no permite concluir que una edificación esté en peligro o que un terreno admita construcción.

## Validación local del entorno cloud — 21 de octubre de 2026

Esta sección conserva evidencia histórica de la versión anterior al semáforo.

La imagen Docker Linux con Python 3.12 y PyTorch CPU completó las 106 pruebas. La comprobación de interfaz con una inferencia real de Chullo, limitada a 2 CPU y 4 GB, terminó en 53,64 segundos y alcanzó un pico de memoria del contenedor de 1 457,5 MiB. Produjo 235 contornos dentro del sector, 27 a 25 m o menos; se verificaron CSV y GeoJSON y la integridad de la demo original.

La demo guardada de GPU conserva 237 contornos. La ejecución CPU no es idéntica byte por byte; los resultados nuevos identifican su propia ejecución y parámetros. Estos valores prueban el funcionamiento de esta configuración local, no la exactitud de las detecciones ni el tiempo o la capacidad del proveedor.

Las comprobaciones aisladas Linux de Streamlit 1.66.0 y PyArrow 24.0.0 pasaron la resolución de dependencias, `pip check`, importación y asignación Arrow entre hilos, las tres pruebas de interfaz y las exportaciones de la demo. No repitieron la inferencia completa ni modifican las mediciones locales anteriores. La revisión visual alojada se registra por separado a continuación.

## Validación en Community Cloud — 21 de octubre de 2026

Esta sección conserva evidencia histórica de la versión anterior al semáforo. Los conteos por una sola banda no se presentan como categorías verificadas de la nueva interfaz.

La verificación se realizó desde la ruta directa indicada arriba, usando la imagen incluida de Chullo y una nueva ejecución del detector CPU con puntuación mínima 0,50. El manifiesto registra su finalización en **2026-10-21T16:23:56.531507-05:00** (hora de Lima).

| Comprobación | Resultado alojado |
|---|---|
| Tiempo de inferencia registrado | 125,064 s |
| Contornos en toda la imagen | 1 004 |
| Contornos que intersectan el sector piloto | 235 |
| Contornos dentro de 25 m del eje | 27 |
| Banda de 50 m aplicada | 55 dentro y 180 fuera, sobre los mismos 235 contornos |
| Distancias CSV de 25 m frente a 50 m | Idénticas: cero diferencias; cambia la pertenencia a la banda |
| CSV descargado | 235 registros |
| GeoJSON descargado | 235 geometrías |
| ZIP descargado | Los cinco archivos esperados, comprobados |

El ZIP contiene `resumen_edificaciones.csv`, `relaciones_cauces.csv`, `edificaciones.geojson`, `metadatos.json` y `LEEME.txt`. Las descargas corresponden a esa nueva corrida. Una segunda sesión nueva mostró los **237 contornos de la demo original**, confirmando que la inferencia de la primera sesión no sustituyó la demostración compartida. No se midió el pico de memoria del proveedor: los **1 457,5 MiB** de la sección anterior pertenecen únicamente al contenedor local.

La comprobación visual final, después del reinicio con el visor HTML de Folium, mostró la imagen real y los **237 contornos de la demo guardada**. Seleccionar `ai_003711de861f7e55` resaltó y acercó la edificación, con distancia de **25,0 m** y puntuación **0,883**; pulsar su polígono abrió la ficha correspondiente. La captura de evidencia local se conserva en `outputs/cloud_validation/despliegue.jpg`.

Esta comprobación visual utilizó la demo restaurada tras el reinicio y no ejecutó nuevamente el detector. La corrida CPU de **235 contornos** y sus exportaciones corresponden a la prueba alojada descrita en la tabla. El mapa funciona mediante la ruta directa; la pantalla de carga de la URL principal permanece como limitación pendiente.

## Referencias del proveedor

- [Desplegar en Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
- [Dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
- [Condiciones técnicas y acceso a repositorios privados](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).
