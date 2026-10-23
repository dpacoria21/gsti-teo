"""Model registration, checkpoint integrity and provenance; CPU test doubles only."""

import hashlib
import json

import numpy as np
import pytest

from src import model_catalog
from src.inference import model as model_loader


@pytest.fixture
def registered_candidate(tmp_path, monkeypatch):
    directory = tmp_path / "registered_model"
    directory.mkdir()
    checkpoint = directory / "candidate.pth"
    checkpoint.write_bytes(b"Test-only checkpoint bytes; never used as production weights")
    metadata = {
        "id": "test-only/spacenet-candidate",
        "name": "Test-only adjusted detector",
        "variant": model_catalog.TRAINED_VARIANT,
        "filename": checkpoint.name,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": model_loader.sha256_file(checkpoint),
        "trained_locally": True,
        "evaluation_report": "evaluation.json",
        "evaluation_summary": {"test": {"f1": 0.6}, "scope": "Synthetic test metadata"},
        "license": "test-only",
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "evaluation.json").write_text('{"test_fixture": true}', encoding="utf-8")
    monkeypatch.setattr(model_catalog, "TRAINED_MANIFEST", manifest)
    return checkpoint, metadata, manifest


def test_registered_candidate_resolves_exact_checkpoint_and_metadata_without_download(
    registered_candidate, monkeypatch
):
    checkpoint, metadata, _ = registered_candidate

    def no_original_download(*args, **kwargs):
        pytest.fail("An adjusted-model request must not resolve to the original checkpoint")

    monkeypatch.setattr(model_loader, "ensure_model", no_original_download)
    path, actual = model_loader.resolve_variant(model_catalog.TRAINED_VARIANT)
    assert path == checkpoint.resolve()
    assert actual == {**metadata, "path": str(checkpoint.resolve())}
    assert list(model_catalog.model_choices()) == [model_catalog.ORIGINAL_VARIANT, model_catalog.TRAINED_VARIANT]


@pytest.mark.parametrize("same_size", [False, True])
def test_changed_checkpoint_bytes_are_rejected_before_deserialization(registered_candidate, same_size):
    checkpoint, _, _ = registered_candidate
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(bytes([original[0] ^ 1]) + original[1:] if same_size else original + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        model_loader.resolve_variant(model_catalog.TRAINED_VARIANT)


def test_registered_checkpoint_cannot_escape_its_model_directory(registered_candidate):
    _, metadata, manifest = registered_candidate
    external = manifest.parent.parent / "outside.pth"
    external.write_bytes(b"outside")
    manifest.write_text(json.dumps({**metadata, "filename": "../outside.pth"}), encoding="utf-8")
    with pytest.raises(ValueError, match="carpeta"):
        model_loader.resolve_variant(model_catalog.TRAINED_VARIANT)
    assert model_catalog.model_choices() == {"usa": "Original · GeoAI USA"}


@pytest.mark.parametrize("invalid", ["{", "null", "[]", "17", "{}"])
def test_invalid_optional_manifest_does_not_disable_original_model(registered_candidate, invalid):
    _, _, manifest = registered_candidate
    manifest.write_text(invalid, encoding="utf-8")
    assert model_catalog.model_choices() == {"usa": "Original · GeoAI USA"}


def test_unknown_model_variant_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(model_loader, "ensure_model", lambda **kwargs: pytest.fail("Must reject the selector"))
    with pytest.raises(ValueError, match="desconocido"):
        model_loader.resolve_variant("not-registered")


def test_half_checkpoint_loads_strictly_into_float32_cpu_model(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    detection = pytest.importorskip("torchvision.models.detection")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def construct(**kwargs):
        assert kwargs["weights"] is None and kwargs["weights_backbone"] is None
        assert kwargs["num_classes"] == 2
        return TinyModel()

    monkeypatch.setattr(detection, "maskrcnn_resnet50_fpn", construct)
    checkpoint = tmp_path / "half.pth"
    torch.save({"weight": torch.tensor([0.5, 0.25], dtype=torch.float16), "counter": torch.tensor(7)}, checkpoint)
    loaded, device = model_loader.load_model(checkpoint, "cpu")
    assert device == "cpu" and loaded.training is False
    assert loaded.weight.dtype == torch.float32
    assert loaded.weight.detach().tolist() == [0.5, 0.25]
    assert loaded.counter.dtype == torch.int64 and loaded.counter.item() == 7
    torch.save({"wrong_weight": torch.ones(2)}, checkpoint)
    with pytest.raises(RuntimeError, match="state_dict"):
        model_loader.load_model(checkpoint, "cpu")


def test_selected_model_provenance_survives_full_raster_pipeline(registered_candidate, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from src.inference import pipeline

    checkpoint, metadata, _ = registered_candidate
    raster = tmp_path / "input_rgb.tif"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=3,
        dtype="uint8",
        crs="EPSG:32719",
        transform=from_origin(227000, 8185000, 0.5, 0.5),
    ) as destination:
        destination.write(np.random.default_rng(4).integers(0, 256, (3, 128, 128), dtype="uint8"))

    class TestOnlyDetector:
        def __call__(self, images):
            assert images[0].device.type == "cpu"
            height, width = images[0].shape[-2:]
            masks = torch.zeros((1, 1, height, width))
            masks[0, 0, 20:40, 20:40] = 1
            return [{"scores": torch.tensor([0.9]), "labels": torch.tensor([1]), "masks": masks}]

    calls = []

    def load_registered(path, device):
        calls.append((path, device))
        assert path == checkpoint.resolve()
        return TestOnlyDetector(), "cpu"

    monkeypatch.setattr(pipeline, "load_model", load_registered)
    output = tmp_path / "test_inference"
    result = pipeline.detect_buildings(raster, output, device="cpu", model_variant=model_catalog.TRAINED_VARIANT)
    assert calls == [(checkpoint.resolve(), "cpu")]
    assert result["model_id"] == metadata["id"]
    for key in ("id", "variant", "sha256", "size_bytes", "trained_locally", "evaluation_report"):
        assert result["model"][key] == metadata[key]
    assert result["model"]["path"] == str(checkpoint.resolve())
    assert result["execution"]["device"] == "cpu"
    assert result["image"]["sha256"] == model_loader.sha256_file(raster)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == result
    raw = (output / "buildings.geojson").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["outputs"]["buildings_sha256"]
    features = json.loads(raw)["features"]
    assert len(features) == result["count"] == 1
    assert features[0]["properties"]["model_id"] == metadata["id"]
    assert features[0]["properties"]["image_id"] == result["image_id"]
    assert (output / "preview.png").is_file()


@pytest.mark.parametrize("variant", [model_catalog.ORIGINAL_VARIANT, model_catalog.TRAINED_VARIANT])
def test_application_preserves_variant_across_worker_boundary(tmp_path, monkeypatch, variant):
    from src.application import detection_runner

    monkeypatch.setattr(detection_runner, "ROOT", tmp_path)
    monkeypatch.setattr(detection_runner, "ml_status", lambda: {"available": True})
    monkeypatch.setattr(detection_runner, "inspect_raster", lambda *args, **kwargs: {})
    commands = []

    def record_worker(command, *args):
        commands.append(command)
        return {"status": "completed"}

    monkeypatch.setattr(detection_runner, "_execute_worker", record_worker)
    output = tmp_path / "outputs" / "inference" / "test_run"
    detection_runner.run_detection(tmp_path / "input.tif", output, model_variant=variant, device="cpu")
    if variant == model_catalog.TRAINED_VARIANT:
        assert commands[0][commands[0].index("--model-variant") + 1] == variant
    else:
        assert "--model-variant" not in commands[0] or commands[0][commands[0].index("--model-variant") + 1] == variant
    assert not (output / ".running.lock").exists()
