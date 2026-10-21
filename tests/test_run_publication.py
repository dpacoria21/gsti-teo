"""Failure-safe publication with synthetic workers, never the pretrained model."""

import hashlib
import inspect
import json
import os
import subprocess
import sys

import pytest

from scripts import run_detection
from src.repositories import run_publication
from src.settings import PROJECT_ROOT


def write_result(directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    buildings = json.dumps({"type": "FeatureCollection", "features": [], "fixture": label}).encode()
    (directory / "buildings.geojson").write_bytes(buildings)
    (directory / "preview.png").write_bytes(b"synthetic preview " + label.encode())
    manifest = {
        "status": "completed",
        "count": 0,
        "fixture": label,
        "outputs": {"buildings_sha256": hashlib.sha256(buildings).hexdigest()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def snapshot(directory):
    return {
        name: (directory / name).read_bytes() for name in run_publication.RESULT_FILES if (directory / name).exists()
    }


def execute(output, detector):
    return run_detection.main(["--image", "synthetic-input.tif", "--output-dir", str(output)], detector=detector)


def good_detector(**kwargs):
    assert kwargs["output_dir"].name == "staged"
    kwargs["progress"](stage="inference", progress=0.5)
    write_result(kwargs["output_dir"], "new")
    kwargs["progress"](stage="completed", progress=1.0, count=0)


def events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_first_completed_run_publishes_all_outputs_then_releases_owned_lock(tmp_path, capsys):
    output = tmp_path / "run"
    assert execute(output, good_detector) == 0
    assert len(snapshot(output)) == 3
    assert json.loads((output / "manifest.json").read_text())["fixture"] == "new"
    assert not (output / ".running.lock").exists()
    assert [event["stage"] for event in events(capsys)] == ["inference", "completed"]


def test_successful_update_keeps_ui_parent_lock_and_worker_log(tmp_path, monkeypatch):
    output = tmp_path / "run"
    write_result(output, "previous")
    lock = output / ".running.lock"
    lock.write_text("UI-owner", encoding="utf-8")
    monkeypatch.setenv("CHULLO_RUN_LOCK_OWNER", "UI-owner")
    (output / "worker.log").write_text("UI owns its worker log", encoding="utf-8")
    assert execute(output, good_detector) == 0
    assert json.loads((output / "manifest.json").read_text())["fixture"] == "new"
    assert lock.read_text() == "UI-owner"
    assert (output / "worker.log").read_text() == "UI owns its worker log"
    backup = next((output / ".attempts").iterdir()) / "previous"
    assert json.loads((backup / "manifest.json").read_text())["fixture"] == "previous"


def test_child_cli_respects_real_parent_lock_and_json_progress(tmp_path):
    output = tmp_path / "run"
    write_result(output, "previous")
    lock = output / ".running.lock"
    lock.write_text(str(os.getpid()), encoding="utf-8")
    worker = tmp_path / "synthetic_cli_worker.py"
    worker.write_text(
        "import sys, json, hashlib\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from scripts.run_detection import main\n"
        + inspect.getsource(write_result)
        + inspect.getsource(good_detector)
        + f"raise SystemExit(main(['--image', 'fixture.tif', '--output-dir', {str(output)!r}], detector=good_detector))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(worker)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        env=dict(os.environ, CHULLO_RUN_LOCK_OWNER=str(os.getpid())),
    )
    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["stage"] for line in result.stdout.splitlines()] == ["inference", "completed"]
    assert lock.read_text() == str(os.getpid())
    assert json.loads((output / "manifest.json").read_text())["fixture"] == "new"


@pytest.mark.parametrize("partial", [False, True])
def test_detector_failure_preserves_previous_bytes_and_records_attempt(tmp_path, capsys, partial):
    output = tmp_path / "run"
    write_result(output, "previous")
    before = snapshot(output)

    def failing_detector(**kwargs):
        if partial:
            (kwargs["output_dir"] / "buildings.geojson").write_text("unfinished new output")
        raise RuntimeError("synthetic inference failure")

    assert execute(output, failing_detector) == 1
    assert snapshot(output) == before
    attempt = next((output / ".attempts").iterdir())
    assert json.loads((attempt / "status.json").read_text())["status"] == "error"
    assert "synthetic inference failure" in (attempt / "traceback.txt").read_text()
    assert not (output / ".running.lock").exists()
    assert events(capsys)[-1]["stage"] == "error"


@pytest.mark.parametrize("failed_file", ["buildings.geojson", "preview.png", "manifest.json"])
def test_publication_write_failure_restores_previous_bytes(tmp_path, monkeypatch, capsys, failed_file):
    output = tmp_path / "run"
    write_result(output, "previous")
    before = snapshot(output)
    replace = run_publication._replace

    def fail_publication(source, destination):
        if source.parent.name == "staged" and destination.name == failed_file:
            raise PermissionError("synthetic publication failure")
        replace(source, destination)

    monkeypatch.setattr(run_publication, "_replace", fail_publication)
    assert execute(output, good_detector) == 1
    assert snapshot(output) == before
    assert not (output / ".running.lock").exists()
    assert "completed" not in [event["stage"] for event in events(capsys)]


def test_first_run_publication_failure_leaves_no_partial_public_results(tmp_path, monkeypatch):
    output = tmp_path / "run"
    replace = run_publication._replace

    def fail_manifest(source, destination):
        if destination.name == "manifest.json":
            raise OSError("synthetic disk failure")
        replace(source, destination)

    monkeypatch.setattr(run_publication, "_replace", fail_manifest)
    assert execute(output, good_detector) == 1
    assert snapshot(output) == {}


def test_backup_failure_cannot_modify_any_previous_result(tmp_path, monkeypatch):
    output = tmp_path / "run"
    write_result(output, "previous")
    before = snapshot(output)

    def fail_backup(*args):
        raise OSError("synthetic full disk")

    monkeypatch.setattr(run_publication.shutil, "copy2", fail_backup)
    assert execute(output, good_detector) == 1
    assert snapshot(output) == before


def test_incomplete_worker_cannot_replace_previous_completed_run(tmp_path):
    output = tmp_path / "run"
    write_result(output, "previous")
    before = snapshot(output)

    def incomplete(**kwargs):
        (kwargs["output_dir"] / "manifest.json").write_text('{"status":"running"}')

    assert execute(output, incomplete) == 1
    assert snapshot(output) == before


def test_cli_does_not_reuse_another_workers_lock(tmp_path):
    output = tmp_path / "run"
    write_result(output, "previous")
    before = snapshot(output)
    lock = output / ".running.lock"
    lock.write_text("other-worker", encoding="utf-8")
    assert execute(output, lambda **kwargs: pytest.fail("worker must not start")) == 1
    assert snapshot(output) == before
    assert lock.read_text() == "other-worker"
    assert not (output / ".attempts").exists()
