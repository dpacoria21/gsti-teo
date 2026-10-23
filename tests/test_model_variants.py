"""Model registration, checkpoint integrity and provenance; CPU test doubles only."""

import hashlib
import json

import numpy as np
import pytest

from src import model_catalog
from src.inference import model as model_loader


@pytest.fixture(autouse=True)
def isolated_catalog(tmp_path, monkeypatch):
    """Never depend on real optional weights or manifests on the developer machine."""
    monkeypatch.setattr(model_catalog, "TRAINED_MANIFEST", tmp_path / "missing_spacenet" / "manifest.json")
    monkeypatch.setattr(model_catalog, "LOCAL_MANIFEST", tmp_path / "missing_chullo" / "manifest.json")


@pytest.fixture(params=model_catalog.TRAINED_VARIANTS)
def registered_candidate(tmp_path, monkeypatch, request):
    directory = tmp_path / "registered_model"
    directory.mkdir()
    checkpoint = directory / "candidate.pth"
    checkpoint.write_bytes(b"Test-only checkpoint bytes; never used as production weights")
    metadata = {
        "id": "test-only/spacenet-candidate",
        "name": "Test-only adjusted detector",
        "variant": request.param,
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
    constant = "TRAINED_MANIFEST" if request.param == model_catalog.TRAINED_VARIANT else "LOCAL_MANIFEST"
    monkeypatch.setattr(model_catalog, constant, manifest)
    return checkpoint, metadata, manifest


def test_registered_candidate_resolves_exact_checkpoint_and_metadata_without_download(
    registered_candidate, monkeypatch
):
    checkpoint, metadata, _ = registered_candidate

    def no_original_download(*args, **kwargs):
        pytest.fail("An adjusted-model request must not resolve to the original checkpoint")

    monkeypatch.setattr(model_loader, "ensure_model", no_original_download)
    path, actual = model_loader.resolve_variant(metadata["variant"])
    assert path == checkpoint.resolve()
    assert actual == {**metadata, "path": str(checkpoint.resolve())}
    assert list(model_catalog.model_choices()) == [model_catalog.ORIGINAL_VARIANT, metadata["variant"]]


@pytest.mark.parametrize("same_size", [False, True])
def test_changed_checkpoint_bytes_are_rejected_before_deserialization(registered_candidate, same_size):
    checkpoint, metadata, _ = registered_candidate
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(bytes([original[0] ^ 1]) + original[1:] if same_size else original + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        model_loader.resolve_variant(metadata["variant"])


def test_registered_checkpoint_cannot_escape_its_model_directory(registered_candidate):
    _, metadata, manifest = registered_candidate
    external = manifest.parent.parent / "outside.pth"
    external.write_bytes(b"outside")
    manifest.write_text(json.dumps({**metadata, "filename": "../outside.pth"}), encoding="utf-8")
    with pytest.raises(ValueError, match="carpeta"):
        model_loader.resolve_variant(metadata["variant"])
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
    tile_size = 512 if metadata["variant"] == model_catalog.LOCAL_VARIANT else 256
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
            assert tuple(images[0].shape) == (3, tile_size, tile_size)
            assert not images[0][:, 128:, :].any() and not images[0][:, :, 128:].any()
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
    result = pipeline.detect_buildings(
        raster, output, device="cpu", model_variant=metadata["variant"], tile_size=tile_size, overlap=64
    )
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


def test_local_variant_rejects_wrong_tile_size_before_loading_weights_or_images(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    from src.inference import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Tile-size validation must happen before reading imagery or acquiring/loading weights.")

    monkeypatch.setattr(pipeline, "resolve_variant", forbidden)
    monkeypatch.setattr(pipeline, "load_model", forbidden)
    monkeypatch.setattr(pipeline, "inspect_image", forbidden)
    with pytest.raises(ValueError, match="tile_size=512"):
        pipeline.detect_buildings(
            tmp_path / "not-opened.tif",
            tmp_path / "not-created",
            model_variant=model_catalog.LOCAL_VARIANT,
            tile_size=256,
            overlap=64,
        )
    assert not (tmp_path / "not-created").exists()


def test_cli_reports_incompatible_local_tile_without_publishing_or_loading_weights(tmp_path, monkeypatch, capsys):
    pytest.importorskip("rasterio")
    from scripts.run_detection import main
    from src.inference import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("An incompatible local-model tile must never read imagery or load weights.")

    monkeypatch.setattr(pipeline, "resolve_variant", forbidden)
    monkeypatch.setattr(pipeline, "load_model", forbidden)
    monkeypatch.setattr(pipeline, "inspect_image", forbidden)
    output = tmp_path / "rejected"
    result = main(
        [
            "--image",
            str(tmp_path / "not-opened.tif"),
            "--output-dir",
            str(output),
            "--model-variant",
            model_catalog.LOCAL_VARIANT,
            "--tile-size",
            "256",
            "--overlap",
            "64",
        ]
    )
    assert result == 1
    event = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert event["stage"] == "error" and "tile_size=512" in event["message"]
    assert not (output / "manifest.json").exists()
    assert not (output / ".running.lock").exists()


@pytest.mark.parametrize("variant", model_catalog.REGISTERED_VARIANTS)
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
    if variant in model_catalog.TRAINED_VARIANTS:
        assert commands[0][commands[0].index("--model-variant") + 1] == variant
    else:
        assert "--model-variant" not in commands[0] or commands[0][commands[0].index("--model-variant") + 1] == variant
    assert not (output / ".running.lock").exists()


@pytest.mark.parametrize("variant", model_catalog.TRAINED_VARIANTS)
def test_cli_forwards_adjusted_variant_and_publishes_matching_provenance(tmp_path, variant):
    from PIL import Image

    from scripts.run_detection import main

    requested = []
    model_id = "synthetic-test/cli-candidate"
    output = tmp_path / "published"

    def test_detector(**kwargs):
        requested.append(kwargs["model_variant"])
        staging = kwargs["output_dir"]
        geojson = b'{"type": "FeatureCollection", "features": []}'
        (staging / "buildings.geojson").write_bytes(geojson)
        Image.new("RGB", (2, 2), "black").save(staging / "preview.png")
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "count": 0,
                    "model_id": model_id,
                    "model": {"id": model_id, "variant": kwargs["model_variant"], "trained_locally": True},
                    "outputs": {"buildings_sha256": hashlib.sha256(geojson).hexdigest()},
                }
            ),
            encoding="utf-8",
        )

    code = main(
        [
            "--image",
            str(tmp_path / "test_input.tif"),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--model-variant",
            variant,
        ],
        detector=test_detector,
    )
    assert code == 0 and requested == [variant]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == model_id
    assert manifest["model"]["variant"] == variant
    assert manifest["model"]["trained_locally"] is True
    assert not (output / ".running.lock").exists()


def test_trained_metadata_no_argument_keeps_spacenet_default(registered_candidate):
    _, metadata, _ = registered_candidate
    if metadata["variant"] == model_catalog.TRAINED_VARIANT:
        assert model_catalog.trained_metadata() == model_catalog.trained_metadata(model_catalog.TRAINED_VARIANT)
    else:
        with pytest.raises(FileNotFoundError):
            model_catalog.trained_metadata()
        assert model_catalog.trained_metadata(model_catalog.LOCAL_VARIANT)["variant"] == model_catalog.LOCAL_VARIANT


@pytest.mark.parametrize("checksum", [None, "", "not-a-hash", "0" * 63, "g" * 64])
def test_unpinned_adjusted_model_is_not_offered_or_resolved(registered_candidate, checksum):
    _, metadata, manifest = registered_candidate
    manifest.write_text(json.dumps({**metadata, "sha256": checksum}), encoding="utf-8")
    assert model_catalog.model_choices() == {"usa": "Original · GeoAI USA"}
    with pytest.raises(ValueError, match="SHA256"):
        model_loader.resolve_variant(metadata["variant"])


@pytest.mark.parametrize("size", [None, 0, -1, True, "10"])
def test_missing_or_invalid_checkpoint_size_is_rejected(registered_candidate, size):
    _, metadata, manifest = registered_candidate
    manifest.write_text(json.dumps({**metadata, "size_bytes": size}), encoding="utf-8")
    assert model_catalog.model_choices() == {"usa": "Original · GeoAI USA"}
    with pytest.raises(ValueError, match="tamaño"):
        model_loader.resolve_variant(metadata["variant"])


def test_missing_optional_models_leave_original_as_only_default():
    assert model_catalog.model_choices() == {model_catalog.ORIGINAL_VARIANT: "Original · GeoAI USA"}
    with pytest.raises(FileNotFoundError):
        model_loader.resolve_variant(model_catalog.LOCAL_VARIANT)


def test_optional_registration_never_replaces_pinned_original(registered_candidate, tmp_path, monkeypatch):
    original = tmp_path / "original.pth"
    original.write_bytes(b"original test fixture; no real weights")
    monkeypatch.setattr(model_loader, "ensure_model", lambda **kwargs: original)
    path, metadata = model_loader.resolve_variant()
    assert path == original
    assert original.read_bytes() == b"original test fixture; no real weights"
    assert metadata["variant"] == model_catalog.ORIGINAL_VARIANT
    assert metadata["sha256"] == model_loader.MODEL_SHA256
    assert metadata["trained_locally"] is False


def test_local_presentation_reports_one_capture_and_unreviewed_annotations():
    presentation = model_catalog.model_presentation(variant=model_catalog.LOCAL_VARIANT)
    assert presentation["title"] == "Ajustado · Chullo (piloto)"
    assert "3 recortes originales" in presentation["description"]
    assert "4 anotados; 1 excluido" in presentation["description"]
    assert "2 de validación y 2 de prueba" in presentation["description"]
    assert "Google Earth Pro / Airbus" in presentation["description"]
    assert "2026-03-04" in presentation["description"]
    limitations = " ".join(presentation["limitations"])
    assert "assistant_visual_manual" in limitations and "human_reviewed=false" in limitations
    assert "sin revisión humana" in presentation["notice"]
    assert "falsos positivos" in presentation["notice"] and "unir edificios" in presentation["notice"]
    assert "Jartum" not in str(presentation) and "sin etiquetas de entrenamiento de Chullo" not in str(presentation)


def test_manifest_presentation_and_detection_title_override_legacy_fallback(registered_candidate):
    from src.application.provenance import describe_detection

    _, metadata, manifest = registered_candidate
    presented = {
        **metadata,
        "title": "Título registrado",
        "description": "Procedencia registrada",
        "limitations": ["Alcance registrado"],
        "user_notice": "Aviso registrado para la demostración",
    }
    manifest.write_text(json.dumps(presented), encoding="utf-8")
    actual = model_catalog.model_presentation(variant=metadata["variant"])
    assert actual == {
        **{key: presented[key] for key in ("title", "description", "limitations")},
        "notice": presented["user_notice"],
    }
    assert model_catalog.model_choices()[metadata["variant"]] == presented["title"]
    assert describe_detection({"model": presented}).model_name == presented["title"]
