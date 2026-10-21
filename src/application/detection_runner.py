"""Run the isolated inference process without importing Streamlit or PyTorch."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from shapely.geometry import box

from ..data_loader import load_study_area
from ..settings import PROJECT_ROOT, environment_python

ROOT = PROJECT_ROOT
ML_PYTHON = environment_python(".venv-ml")


def ml_status() -> dict:
    required = [ML_PYTHON, ROOT / "scripts/run_detection.py", ROOT / "models/building_footprints_usa.pth"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    return {
        "available": not missing,
        "python": str(ML_PYTHON),
        "reason": "" if not missing else "Faltan componentes: " + ", ".join(missing),
    }


def ml_available() -> bool:
    return ml_status()["available"]


def inspect_raster(path: str | Path, area_id: str = "chullo") -> dict:
    """Read pixels/georeference in the ML environment before starting the model."""
    if not ML_PYTHON.is_file():
        raise ValueError("El entorno de IA no está instalado. Ejecute PREPARAR_IA.ps1.")
    result = subprocess.run(
        [str(ML_PYTHON), str(ROOT / "scripts/inspect_raster.py"), str(Path(path).resolve())],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise ValueError("No se puede analizar la imagen: " + (result.stderr or result.stdout)[-1500:])
    info = json.loads(result.stdout)
    config = load_study_area(area_id)["config"]
    if not box(*info["bounds_wgs84"]).intersects(box(*config["bbox"])):
        raise ValueError(
            f"La imagen no cubre el sector seleccionado ({config.get('name', area_id)}). Otros lugares necesitan sus propias capas de cauces."
        )
    return info


def run_detection(
    raster_path: str | Path,
    output_dir: str | Path,
    confidence: float = 0.5,
    image_manifest: str | Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
    device: str = "auto",
    area_id: str = "chullo",
) -> dict:
    status = ml_status()
    if not status["available"]:
        raise ValueError(status["reason"])
    if not 0.05 <= float(confidence) <= 0.99:
        raise ValueError("La confianza debe estar entre 0.05 y 0.99.")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Dispositivo desconocido.")
    raster = Path(raster_path).resolve()
    inspect_raster(raster, area_id=area_id)
    destination = Path(output_dir).resolve()
    if not destination.is_relative_to(ROOT / "outputs/inference"):
        raise ValueError("Los resultados deben guardarse dentro de outputs/inference.")
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(ML_PYTHON),
        "-u",
        str(ROOT / "scripts/run_detection.py"),
        "--image",
        str(raster),
        "--output-dir",
        str(destination),
        "--threshold",
        str(float(confidence)),
        "--device",
        device,
    ]
    if image_manifest:
        command.extend(["--image-manifest", str(Path(image_manifest).resolve())])
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        CHULLO_RUN_LOCK_OWNER=str(os.getpid()),
    )
    lock = destination / ".running.lock"
    try:
        lock_handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError("Ya hay una detección ejecutándose en esta carpeta. Espere a que termine.") from None
    lock_handle.write(str(os.getpid()))
    lock_handle.close()
    try:
        return _execute_worker(command, destination, environment, on_progress)
    finally:
        lock.unlink(missing_ok=True)


def _execute_worker(command, destination, environment, on_progress):
    with (destination / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and on_progress:
                    on_progress(event)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
    if code:
        tail = (destination / "worker.log").read_text(encoding="utf-8")[-2000:]
        raise RuntimeError(f"El detector no completó la ejecución (código {code}). {tail}")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("La ejecución no confirmó resultados completos.")
    return manifest
