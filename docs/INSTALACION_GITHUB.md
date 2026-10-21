# Instalar el prototipo desde GitHub

El repositorio contiene el código modular, las dependencias fijadas, los datos procesados de dos sectores, los GeoTIFF de trabajo y dos ejecuciones reales guardadas. El repositorio privado es el medio de colaboración del proyecto; no constituye una licencia abierta de las imágenes de terceros.

## Requisitos

- Windows y Python 3.12, o `uv` con capacidad de obtener Python 3.12.
- Git para clonar, o descargar y extraer el ZIP del repositorio desde GitHub.
- Acceso al repositorio privado con tu propia cuenta.
- Conexión durante la primera instalación. Las teselas del mapa base y algunos recursos del visor también necesitan Internet.

Clona el [repositorio privado del proyecto](https://github.com/dpacoria21/gsti-teo) y entra en la carpeta clonada. Todos los comandos siguientes se ejecutan desde esa carpeta.

```powershell
git clone https://github.com/dpacoria21/gsti-teo.git
cd gsti-teo
```

## Abrir los resultados preparados

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\PREPARAR.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\INICIAR.ps1
```

También puedes abrir `INICIAR.cmd`. El navegador muestra `http://127.0.0.1:8501/`. La aplicación carga los contornos guardados de Chullo y calcula distancias, bandas y exportaciones sin volver a ejecutar la red. El entorno IA todavía no es necesario para consultar estos resultados.

Si el puerto está ocupado por otra aplicación:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\INICIAR.ps1 -Port 8502
```

Para detener solo el servidor de este proyecto:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\DETENER.ps1
```

## Habilitar nuevas detecciones de imágenes

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\PREPARAR_IA.ps1
```

El script crea `.venv-ml`, instala PyTorch/torchvision y las bibliotecas geográficas, y descarga los pesos `building_footprints_usa.pth` desde la revisión fijada en `models/manifest.json`. Verifica el SHA256 del checkpoint antes de utilizarlo. El archivo de pesos pesa 176.225.905 bytes y no está en Git. La instalación de PyTorch necesita espacio y descargas adicionales.

La selección automática usa CUDA cuando encuentra `nvidia-smi`; para una computadora sin GPU NVIDIA compatible o para forzar CPU:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\PREPARAR_IA.ps1 -Device cpu
```

En la aplicación, elige **Chullo**, **Detección IA desde imagen** y pulsa **Detectar en esta imagen**. Se procesa el GeoTIFF incluido y se publica el resultado al terminar correctamente. Cambiar la banda y pulsar **Analizar sector** reutiliza los contornos; cambiar la puntuación requiere una nueva detección para aplicarla.

## Qué contiene Git y qué permanece local

| Incluido en el repositorio | Propósito |
|---|---|
| `app.py`, `src/`, `scripts/`, `tests/`, configuración y requisitos | Ejecutar, verificar y continuar el desarrollo. |
| `data/processed/chullo/` y `data/processed/mariano_melgar/` | Sectores, cauces, inventarios y trazabilidad de los datos preparados. |
| `data/sources.json` | Fuentes, condiciones, fechas y alcance de las capas. |
| GeoTIFF RGB y manifiesto de cada sector | Ejecutar la detección sobre los píxeles de la demostración. |
| Puntos, residuos, informe de captura y evidencia de fecha de Chullo | Documentar cómo se georreferenció la captura. |
| `outputs/inference/{chullo_demo,mariano_melgar_demo}/` | GeoJSON, manifiesto y vista previa de las dos corridas guardadas. |
| `outputs/arquitectura/`, `outputs/presentacion/` y `outputs/google_earth/` | Diagramas, mapas, exportaciones ligeras y referencias de captura; se omiten los ZIP. |
| `models/manifest.json` | Identificación exacta y procedencia del modelo descargable. |

No se incluyen `.venv`, `.venv-ml`, los pesos, `data/raw/`, cargas de usuarios, los dos JPG originales de Google Earth de aproximadamente 21,8 MB cada uno, el ortomosaico fuente de Mariano Melgar, archivos temporales, registros, auditorías locales, respaldos ni entregas ZIP. Los originales de captura conservados en la computadora de preparación son necesarios para repetir la georreferenciación desde cero. El GeoTIFF preparado sí permite repetir la inferencia.

El ZIP de entrega preparado en la computadora original es distinto del ZIP que genera GitHub: el primero puede contener pesos y originales adicionales; el de GitHub solo contiene los archivos versionados. Los entornos Python se crean en el equipo destino en ambos casos.

Los manifiestos conservan rutas absolutas de la máquina donde se generaron como evidencia histórica. El cargador localiza los archivos incluidos por su ruta dentro de la carpeta clonada cuando la ruta histórica no existe. `.gitattributes` conserva los bytes de JSON, GeoJSON y CSV para no romper las comprobaciones de integridad.

## Verificar y desarrollar

Instala las herramientas de desarrollo con `uv`:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app.py src tests scripts/inspect_raster.py scripts/run_detection.py
.\.venv\Scripts\python.exe -m ruff format --check app.py src tests scripts/inspect_raster.py scripts/run_detection.py
```

Con el entorno IA preparado, ejecuta sus pruebas:

```powershell
.\.venv-ml\Scripts\python.exe -m pytest tests/test_building_detection.py tests/test_inference_config.py tests/test_inference_pipeline.py tests/test_georeference_capture.py tests/test_raster_cli.py -q
```

Estas pruebas comprueban el código con ejemplos controlados; no equivalen a medir la precisión del detector sobre viviendas etiquetadas de Arequipa. La inferencia real se solicita desde la aplicación o con el comando documentado en el README.

`scripts/prepare_data.py --offline` necesita `data/raw/`, que no está en Git. Para volver a obtener esos originales se puede ejecutar el script sin `--offline`, sujeto a la disponibilidad de las fuentes. No hace falta regenerarlos para abrir los datos procesados incluidos.

## Atribución y alcance

La imagen de Chullo procede de Google Earth Pro y mantiene el crédito **Google Earth; Image © 2026 Airbus**. Es material del prototipo académico sujeto a las condiciones de sus proveedores; no se declara como dataset abierto ni se utilizó para entrenar. La segunda imagen corresponde a CENEPRED. Los inventarios Microsoft y los ejes OpenStreetMap conservan su procedencia y condiciones en `data/sources.json`.

Las bandas expresan proximidad al eje cartografiado. No son delimitaciones legales ni una clasificación de peligro de inundación. No se añadió una licencia nueva al código ni se extendieron las licencias de las dependencias a los datos de terceros.
