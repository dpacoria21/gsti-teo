"""Multiuser isolation of uploaded images and completed detection selection."""

import json

import pytest

from src.application.images import data_fingerprint, stored_upload
from src.application.session_workspace import (
    new_detection_output,
    remember_result,
    selected_result,
    session_identifier,
    session_output_root,
)


def test_identical_uploads_are_isolated_between_sessions(tmp_path):
    first_state, second_state = {}, {}
    first_id, second_id = session_identifier(first_state), session_identifier(second_state)
    assert first_id != second_id
    assert session_identifier(first_state) == first_id
    first = stored_upload(b"same image", "first.tif", "chullo", tmp_path, session_id=first_id)
    second = stored_upload(b"same image", "second.tif", "chullo", tmp_path, session_id=second_id)
    assert first[0] != second[0]
    assert first[3] != second[3]
    assert first[0].read_bytes() == second[0].read_bytes() == b"same image"
    assert json.loads(first[1].read_text(encoding="utf-8"))["original_filename"] == "first.tif"
    assert json.loads(second[1].read_text(encoding="utf-8"))["original_filename"] == "second.tif"
    assert data_fingerprint(tmp_path) == (), "Private uploads must not invalidate shared base-map caches."


def test_new_attempts_do_not_replace_demo_or_other_session_results(tmp_path):
    first_state, second_state = {}, {}
    first_id = session_identifier(first_state)
    second_id = session_identifier(second_state)
    demo = tmp_path / "outputs" / "inference" / "chullo_demo"
    first_run = new_detection_output(first_id, tmp_path)
    next_attempt = new_detection_output(first_id, tmp_path)
    other_session_run = new_detection_output(second_id, tmp_path)
    assert len({first_run, next_attempt, other_session_run, demo}) == 4
    assert first_run.is_relative_to(session_output_root(first_id, tmp_path))
    assert not first_run.is_relative_to(session_output_root(second_id, tmp_path))
    assert selected_result(first_state, "chullo:bundled", demo) == demo
    first_run.mkdir(parents=True)
    (first_run / "manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
    remember_result(first_state, "chullo:bundled", first_run)
    assert selected_result(first_state, "chullo:bundled", demo) == first_run
    assert selected_result(second_state, "chullo:bundled", demo) == demo
    assert selected_result(first_state, "mariano_melgar:bundled", demo) == demo
    assert selected_result(first_state, "chullo:upload:new-image", demo) == demo


@pytest.mark.parametrize("unsafe_id", ["../another-session", "a/b", "", "f" * 33, None])
def test_session_storage_rejects_unsafe_identifiers(tmp_path, unsafe_id):
    with pytest.raises(ValueError, match="sesión"):
        session_output_root(unsafe_id, tmp_path)


def test_cli_upload_storage_remains_compatible(tmp_path):
    raster, _, _, output = stored_upload(b"image", "../name.tif", "chullo", tmp_path)
    assert raster.parent.parent == tmp_path / "data" / "uploads"
    assert raster.name == "image.tif"
    assert output.parent == tmp_path / "outputs" / "inference"
