"""Verify configurable worker paths and real OS locks without loading the model."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src import settings
from src.application import detection_runner


def test_ml_interpreter_defaults_to_local_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CHULLO_ML_PYTHON", raising=False)
    assert settings.ml_python(tmp_path) == settings.environment_python(".venv-ml", tmp_path)


def test_ml_interpreter_accepts_current_python_outside_project(tmp_path, monkeypatch):
    monkeypatch.setenv("CHULLO_ML_PYTHON", sys.executable)
    assert settings.ml_python(tmp_path) == Path(sys.executable).absolute()


def test_ml_interpreter_preserves_virtual_environment_symlink(tmp_path, monkeypatch):
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    try:
        interpreter.symlink_to(sys.executable)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Creating symlinks is not available on this host: {error}")
    monkeypatch.setenv("CHULLO_ML_PYTHON", str(interpreter))
    selected = settings.ml_python(tmp_path)
    assert selected == interpreter
    assert selected.is_symlink()


def test_status_reports_missing_external_interpreter_without_relative_path_failure(tmp_path, monkeypatch):
    root = tmp_path / "project"
    interpreter = tmp_path / "system" / "missing-python"
    monkeypatch.setattr(detection_runner, "ROOT", root)
    monkeypatch.setattr(detection_runner, "ML_PYTHON", interpreter)
    status = detection_runner.ml_status()
    assert not status["available"]
    assert str(interpreter) in status["reason"]


def test_status_accepts_existing_external_interpreter(tmp_path, monkeypatch):
    for name in ("scripts/run_detection.py", "models/building_footprints_usa.pth"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(detection_runner, "ROOT", tmp_path)
    monkeypatch.setattr(detection_runner, "ML_PYTHON", Path(sys.executable))
    assert detection_runner.ml_status()["available"]


def test_global_slot_releases_on_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_runner, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="fixture failure"):
        with detection_runner._inference_slot():
            raise RuntimeError("fixture failure")
    with detection_runner._inference_slot():
        assert (tmp_path / ".runtime/inference.lock").exists()


def test_global_slot_rejects_another_process(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_runner, "ROOT", tmp_path)
    source = (
        "from pathlib import Path\n"
        "from src.application import detection_runner as runner\n"
        f"runner.ROOT = Path({str(tmp_path)!r})\n"
        "try:\n"
        "    with runner._inference_slot():\n"
        "        raise SystemExit('second process unexpectedly acquired the slot')\n"
        "except ValueError as error:\n"
        "    assert 'ocupado' in str(error)\n"
    )
    with detection_runner._inference_slot():
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=settings.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode == 0, result.stderr


def test_global_slot_releases_after_holder_process_exits_abruptly(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_runner, "ROOT", tmp_path)
    ready = tmp_path / "holder-ready"
    source = (
        "from pathlib import Path\n"
        "import os, sys\n"
        "from src.application import detection_runner as runner\n"
        f"runner.ROOT = Path({str(tmp_path)!r})\n"
        "with runner._inference_slot():\n"
        f"    Path({str(ready)!r}).touch()\n"
        "    sys.stdin.read(1)\n"
        "    os._exit(7)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", source],
        cwd=settings.PROJECT_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "The fixture process did not acquire its lock."
        with pytest.raises(ValueError, match="ocupado"):
            with detection_runner._inference_slot():
                pytest.fail("The live holder must retain the slot.")
        # os._exit skips context-manager cleanup in the actual Python process,
        # including when Windows starts it through a virtualenv launcher.
        process.stdin.write(b"x")
        process.stdin.flush()
        process.wait(timeout=10)
        assert process.returncode == 7
        with detection_runner._inference_slot():
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdin.close()
        process.stderr.close()


def test_different_result_directories_still_share_one_inference_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_runner, "ROOT", tmp_path)
    monkeypatch.setattr(detection_runner, "ml_status", lambda: {"available": True})
    monkeypatch.setattr(detection_runner, "inspect_raster", lambda *a, **kw: {})
    destination = tmp_path / "outputs/inference/first"
    other_destination = tmp_path / "outputs/inference/second"
    image = tmp_path / "image.tif"

    def worker(command, output, environment, on_progress):
        assert environment["CHULLO_RUN_LOCK_OWNER"] == str(os.getpid())
        with pytest.raises(ValueError, match="ocupado"):
            detection_runner.run_detection(image, other_destination)
        return {"status": "completed"}

    monkeypatch.setattr(detection_runner, "_execute_worker", worker)
    assert detection_runner.run_detection(image, destination)["status"] == "completed"
    assert not (destination / ".running.lock").exists()
