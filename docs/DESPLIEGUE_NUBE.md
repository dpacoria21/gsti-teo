# Despliegue de la aplicación con reconocimiento de imágenes

El despliegue usa el mismo código modular, GeoTIFF y modelo verificado del prototipo. `cloud/app.py` prepara los pesos y configura el intérprete del worker antes de abrir Streamlit. La inferencia permanece en un proceso separado. No depende de que la laptop esté encendida.

## Streamlit Community Cloud

En la cuenta conectada a GitHub, crear una aplicación con estos valores:

| Campo | Valor |
|---|---|
| Repository | `dpacoria21/gsti-teo` |
| Branch | `main` |
| Main file path | `cloud/app.py` |
| Python version, en Advanced settings | `3.12` |

Community Cloud encuentra `cloud/requirements.txt` junto al punto de entrada. Incluye PyTorch y torchvision CPU, las bibliotecas geográficas y Streamlit con versiones fijadas. `packages.txt` instala las bibliotecas de sistema `libgomp1` y `libexpat1`, necesarias para el runtime numérico y la lectura de GeoTIFF. No requiere una clave API de IA. El primer arranque descarga los pesos públicos desde una revisión concreta de Hugging Face y verifica su SHA256; los siguientes accesos en ese proceso reutilizan el archivo.

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

La demo guardada de GPU conserva 237 contornos. La ejecución CPU no es idéntica byte por byte; los resultados nuevos identifican su propia ejecución y parámetros. Estos valores prueban el funcionamiento de esta configuración, no la exactitud de las detecciones ni el tiempo o la capacidad del proveedor. La URL de nube aún necesita su propia comprobación después de publicar.

## Referencias del proveedor

- [Desplegar en Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
- [Dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
- [Condiciones técnicas y acceso a repositorios privados](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).
