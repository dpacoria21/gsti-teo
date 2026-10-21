"""Exercise the process boundary with tiny local workers, without model inference."""

import json
import os
import sys

import pytest

from src.application import detection_runner


def worker_command(tmp_path, source):
    script = tmp_path / "worker_fixture.py"
    script.write_text(source, encoding="utf-8")
    return [sys.executable, "-u", str(script)]


def test_worker_streams_progress_and_requires_completed_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    command = worker_command(
        tmp_path,
        (
            "from pathlib import Path\n"
            "import json\n"
            "print('library diagnostic, not a progress event')\n"
            "print(json.dumps({'stage': 'tile', 'progress': 0.5}))\n"
            f"Path({str(manifest)!r}).write_text(json.dumps({{'status':'completed','count':0}}))\n"
        ),
    )
    events = []
    result = detection_runner._execute_worker(command, tmp_path, dict(os.environ), events.append)
    assert events == [{"stage": "tile", "progress": 0.5}]
    assert result == {"status": "completed", "count": 0}
    assert "library diagnostic" in (tmp_path / "worker.log").read_text(encoding="utf-8")


def test_successful_exit_cannot_publish_unfinished_results(tmp_path):
    manifest = tmp_path / "manifest.json"
    command = worker_command(
        tmp_path, (f'from pathlib import Path\nPath({str(manifest)!r}).write_text(\'{{"status":"running"}}\')\n')
    )
    with pytest.raises(ValueError, match="completos"):
        detection_runner._execute_worker(command, tmp_path, dict(os.environ), None)


def test_failed_worker_reports_diagnostic_even_with_previous_completed_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "completed"}))
    command = worker_command(tmp_path, "print('fixture worker failed')\nraise SystemExit(3)\n")
    with pytest.raises(RuntimeError, match="fixture worker failed"):
        detection_runner._execute_worker(command, tmp_path, dict(os.environ), None)


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_runner, "ROOT", tmp_path)
    monkeypatch.setattr(detection_runner, "ml_status", lambda: {"available": True})
    monkeypatch.setattr(detection_runner, "inspect_raster", lambda *a, **kw: {})
    return tmp_path / "outputs" / "inference" / "run"


def test_worker_failure_releases_lock_without_touching_input(tmp_path, isolated_runner, monkeypatch):
    image = tmp_path / "input.tif"
    image.write_bytes(b"synthetic test input; no real raster")

    def fail(command, destination, environment, on_progress):
        assert (isolated_runner / ".running.lock").is_file()
        assert environment["CHULLO_RUN_LOCK_OWNER"] == (isolated_runner / ".running.lock").read_text()
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(detection_runner, "_execute_worker", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        detection_runner.run_detection(image, isolated_runner)
    assert not (isolated_runner / ".running.lock").exists()
    assert image.read_bytes() == b"synthetic test input; no real raster"


def test_active_run_cannot_be_overwritten(tmp_path, isolated_runner, monkeypatch):
    isolated_runner.mkdir(parents=True)
    lock = isolated_runner / ".running.lock"
    lock.write_text("other-worker")
    monkeypatch.setattr(detection_runner, "_execute_worker", lambda *a: pytest.fail("must not start"))
    with pytest.raises(ValueError, match="ejecutándose"):
        detection_runner.run_detection(tmp_path / "input.tif", isolated_runner)
    assert lock.read_text() == "other-worker"
