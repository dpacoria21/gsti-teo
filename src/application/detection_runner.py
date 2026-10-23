"""Run the isolated inference process without importing Streamlit or PyTorch."""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from shapely.geometry import box

from ..data_loader import load_study_area
from ..settings import PROJECT_ROOT, ml_python

ROOT = PROJECT_ROOT
ML_PYTHON = ml_python()


def _display_path(path: Path) -> str:
    """Keep local diagnostics short without assuming an interpreter is in ROOT."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ml_status() -> dict:
    required = [ML_PYTHON, ROOT / "scripts/run_detection.py", ROOT / "models/building_footprints_usa.pth"]
    missing = [_display_path(path) for path in required if not path.is_file()]
    return {
        "available": not missing,
        "python": str(ML_PYTHON),
        "reason": "" if not missing else "Faltan componentes: " + ", ".join(missing),
    }


def ml_available() -> bool:
    return ml_status()["available"]


@contextmanager
def _inference_slot():
    """Allow one worker across sessions/processes; the OS owns lock lifetime.

    Keep the file in place: removing it could let another process lock a new
    inode while a previous reader still owns the old one. Closing the descriptor
    releases the lock, including when its process terminates unexpectedly.
    """
    lock_path = ROOT / ".runtime" / "inference.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            # Windows byte-range locks need a byte to lock on every filesystem.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(
                    "El motor de IA está ocupado con otra detección. Espere a que termine y vuelva a intentarlo."
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    "El motor de IA está ocupado con otra detección. Espere a que termine y vuelva a intentarlo."
                ) from error
        yield


def inspect_raster(path: str | Path, area_id: str = "chullo") -> dict:
    """Read pixels/georeference in the ML environment before starting the model."""
    if not ML_PYTHON.is_file():
        raise ValueError("El intérprete de IA no está disponible. Configure CHULLO_ML_PYTHON o prepare .venv-ml.")
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
    model_variant: str = "usa",
) -> dict:
    status = ml_status()
    if not status["available"]:
        raise ValueError(status["reason"])
    if not 0.05 <= float(confidence) <= 0.99:
        raise ValueError("La confianza debe estar entre 0.05 y 0.99.")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Dispositivo desconocido.")
    raster = Path(raster_path).resolve()
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
    if model_variant != "usa":
        command.extend(["--model-variant", model_variant])
    environment = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        CHULLO_RUN_LOCK_OWNER=str(os.getpid()),
    )
    with _inference_slot():
        inspect_raster(raster, area_id=area_id)
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
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
    if code:
        tail = (destination / "worker.log").read_text(encoding="utf-8")[-2000:]
        raise RuntimeError(f"El detector no completó la ejecución (código {code}). {tail}")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("La ejecución no confirmó resultados completos.")
    return manifest
