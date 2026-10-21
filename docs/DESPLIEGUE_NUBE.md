# Despliegue de la aplicación con reconocimiento de imágenes

El despliegue usa el mismo código modular, GeoTIFF y modelo verificado del prototipo. `cloud/app.py` prepara los pesos y configura el intérprete del worker antes de abrir Streamlit. La inferencia permanece en un proceso separado. No depende de que la laptop esté encendida.

## Aplicación desplegada y acceso

URL principal: [Chullo en Streamlit Community Cloud](https://chullo-arequipa-dpacoria21.streamlit.app/). El acceso de la aplicación se verificó en **Settings → Sharing → Only specific people can view this app**. La aplicación y el repositorio son privados; entrar requiere una cuenta autorizada.

En la comprobación del 21 de octubre de 2026, la página principal permaneció en la pantalla de carga en Chrome. La [ruta directa de la aplicación](https://chullo-arequipa-dpacoria21.streamlit.app/~/+/), observada en el iframe que sirve Community Cloud, sí permitió ejecutar el detector y las descargas. Tras las interacciones, el componente `streamlit_folium.st_folium` mostró un timeout y el mapa quedó en carga incompleta. La ruta directa se conserva como acceso temporal comprobado para el cálculo y las exportaciones; la visualización del mapa y la URL principal siguen pendientes. No se atribuye una causa concreta sin evidencia.

## Streamlit Community Cloud

La aplicación se desplegó desde la cuenta conectada a GitHub con estos valores:

| Campo | Valor |
|---|---|
| Repository | `dpacoria21/gsti-teo` |
| Branch | `main` |
| Main file path | `cloud/app.py` |
| Python version, en Advanced settings | `3.12` |

Community Cloud encuentra `cloud/requirements.txt` junto al punto de entrada. Incluye PyTorch y torchvision CPU, las bibliotecas geográficas y Streamlit con versiones fijadas. `packages.txt` instala las bibliotecas de sistema `libgomp1` y `libexpat1`, necesarias para el runtime numérico y la lectura de GeoTIFF. No requiere una clave API de IA. El primer arranque descarga los pesos públicos desde una revisión concreta de Hugging Face y verifica su SHA256; los siguientes accesos en ese proceso reutilizan el archivo.

El proveedor sustituyó PyArrow 25.0.1 por 24.0.0 durante el despliegue, como medida de compatibilidad vinculada a [apache/arrow#50471](https://github.com/apache/arrow/issues/50471). El archivo cloud fija ahora **PyArrow 24.0.0** para coincidir con ese entorno. El input de nube toma las dependencias declaradas de `requirements.in`, sin alterar el lock ni el entorno local. La resolución Linux conserva las demás versiones y omite `colorama` y `tzdata`, dependencias específicas de Windows que se habían heredado del lock local.

También se fija **Streamlit 1.66.0**, publicado el 14 de octubre de 2026, para comprobar sus correcciones de compatibilidad de iframes. La [corrección oficial de iframes con origen opaco](https://github.com/streamlit/streamlit/pull/17409) no confirma la causa del problema observado aquí. El cambio requiere comprobar de nuevo la URL alojada antes de dar la visualización por resuelta. El lock omite `httptools` y `toml`, que dejaron de ser dependencias obligatorias en esa versión; conserva las demás versiones.

Conservar el repositorio privado. El acceso a la aplicación se administra en la configuración de Community Cloud; no cambiarlo a público para resolver un problema de conexión al repositorio. La conexión OAuth y los términos del proveedor los revisa el titular de la cuenta.

El alojamiento gratuito tiene recursos compartidos: comprobar una inferencia completa antes de considerar el despliegue funcional. Un mapa que abre no prueba que haya memoria suficiente para ejecutar Mask R-CNN. Si el proveedor termina el worker por falta de memoria, utilizar el contenedor en un servidor con capacidad suficiente; no sustituir silenciosamente la inferencia por resultados guardados.

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
3. Cambiar la banda de 25 a 50 m, aplicar el análisis y comprobar que las distancias se conservan mientras cambia la clasificación.
4. Descargar CSV y GeoJSON y verificar que contienen todas las filas del resultado aplicado.
5. Abrir otra sesión y comprobar que sigue mostrando su propia selección. No debe reemplazarse la demo almacenada.

La evaluación sigue siendo de proximidad al eje cartografiado. El despliegue no incorpora simulación hidráulica, delimitación legal ni una validación independiente de exactitud.

## Validación local del entorno cloud — 21 de octubre de 2026

La imagen Docker Linux con Python 3.12 y PyTorch CPU completó las 106 pruebas. La comprobación de interfaz con una inferencia real de Chullo, limitada a 2 CPU y 4 GB, terminó en 53,64 segundos y alcanzó un pico de memoria del contenedor de 1 457,5 MiB. Produjo 235 contornos dentro del sector, 27 a 25 m o menos; se verificaron CSV y GeoJSON y la integridad de la demo original.

La demo guardada de GPU conserva 237 contornos. La ejecución CPU no es idéntica byte por byte; los resultados nuevos identifican su propia ejecución y parámetros. Estos valores prueban el funcionamiento de esta configuración local, no la exactitud de las detecciones ni el tiempo o la capacidad del proveedor.

Después de fijar PyArrow 24.0.0, una comprobación aislada Linux verificó la resolución y consistencia de dependencias, importación y asignación Arrow entre hilos, cuatro ejecuciones de la interfaz y exportaciones CSV, GeoJSON y ZIP de la demo. No repitió la inferencia completa ni modifica las mediciones locales anteriores.

Con Streamlit 1.66.0 y PyArrow 24.0.0, la comprobación aislada Linux pasó `pip check`, las tres pruebas de interfaz existentes y tres ejecuciones con generación del componente Folium y sus capas. El cambio de banda de la demo a 50 m conservó el resultado esperado de 56 contornos dentro. Estas pruebas inspeccionan la interfaz y sus datos desde Python; no comprueban la ejecución de JavaScript ni la descarga de recursos en el navegador alojado.

## Validación en Community Cloud — 21 de octubre de 2026

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

La pantalla de carga de la URL principal y el timeout del componente del mapa continúan pendientes. La prueba del detector y de las exportaciones mediante la ruta directa no demuestra que esos problemas de visualización hayan quedado corregidos.

## Referencias del proveedor

- [Desplegar en Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
- [Dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
- [Condiciones técnicas y acceso a repositorios privados](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).
