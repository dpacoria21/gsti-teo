"""Supervised data and metric invariants; no GPU, downloads or production writes."""

# ruff: noqa: E402

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, box

from src.training.dataset import BuildingDataset, masks_to_boxes, read_manifest, read_rgb, reference_masks
from src.training.metrics import aggregate_counts, match_masks, paired_bootstrap
from src.training.runner import TrainingConfig


def test_reference_rasterization_preserves_holes_and_multipart_instances():
    shell = Polygon([(1, 1), (7, 1), (7, 7), (1, 7)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]])
    multipart = MultiPolygon([box(8, 8, 10, 10), box(12, 12, 14, 14)])
    masks = reference_masks([shell, multipart], (16, 16), from_origin(0, 16, 1, 1))
    assert masks.shape == (2, 16, 16)
    assert masks.sum(axis=(1, 2)).tolist() == [32, 8]
    assert masks[0, 13, 2] == 0
    assert masks[1, 7, 8] == 1 and masks[1, 3, 12] == 1


def test_invalid_pixels_and_small_labels_are_excluded_without_merging():
    valid = np.ones((16, 16), dtype=bool)
    valid[12:, :4] = False
    masks = reference_masks([box(0, 0, 4, 4), box(8, 8, 9, 9)], (16, 16), from_origin(0, 16, 1, 1), valid)
    assert masks.shape == (0, 16, 16)


def test_boxes_use_exclusive_pixel_bounds_for_narrow_buildings():
    masks = torch.zeros((1, 8, 8), dtype=torch.uint8)
    masks[0, 2:6, 3] = 1
    assert masks_to_boxes(masks).tolist() == [[3, 2, 4, 6]]
    assert masks_to_boxes(torch.zeros((0, 8, 8))).shape == (0, 4)


def test_one_to_one_matching_counts_duplicate_predictions_as_false_positives():
    reference = np.zeros((1, 8, 8), dtype=bool)
    reference[0, 1:5, 1:5] = True
    predicted = np.repeat(reference, 2, axis=0)
    result = match_masks(predicted, reference)
    assert result == {"tp": 1, "fp": 1, "fn": 0, "matched_iou_sum": 1.0}
    summary = aggregate_counts([result])
    assert summary["precision"] == 0.5 and summary["recall"] == 1
    assert summary["f1"] == pytest.approx(2 / 3)


def test_iou_boundary_and_empty_reference_images():
    reference = np.ones((1, 2, 2), dtype=bool)
    predicted = reference.copy()
    predicted[:, 0, :] = False
    assert match_masks(predicted, reference)["tp"] == 1
    assert match_masks(predicted, reference, 0.51)["fn"] == 1
    assert match_masks(predicted, np.zeros((0, 2, 2)))["fp"] == 1
    assert match_masks(np.zeros((0, 2, 2)), reference)["fn"] == 1


def test_evaluation_ignores_predictions_entirely_outside_valid_imagery():
    from src.training.runner import evaluate_model

    class FixedDataset:
        scenes = [{"scene_id": "fixed"}]

        def __len__(self):
            return 1

        def __getitem__(self, _):
            reference = torch.zeros((1, 4, 4), dtype=torch.uint8)
            reference[0, :2, :2] = 1
            valid = torch.ones((4, 4), dtype=torch.bool)
            valid[2:, :] = False
            return torch.zeros((3, 4, 4)), {"masks": reference, "valid_mask": valid}

    class FixedDetector(torch.nn.Module):
        def forward(self, images):
            masks = torch.zeros((2, 1, 4, 4))
            masks[0, 0, :2, :2] = 1
            masks[1, 0, 2:, :] = 1
            return [{"masks": masks, "scores": torch.tensor([0.9, 0.9]), "labels": torch.tensor([1, 1])}]

    result = evaluate_model(FixedDetector(), FixedDataset(), "cpu", TrainingConfig())
    assert result["summary"]["tp"] == 1
    assert result["summary"]["fp"] == 0
    assert result["summary"]["f1"] == 1


def test_bootstrap_is_paired_and_deterministic():
    baseline = [{"scene_id": "a", "tp": 1, "fp": 1, "fn": 0, "matched_iou_sum": 1}]
    candidate = [{"scene_id": "a", "tp": 1, "fp": 0, "fn": 0, "matched_iou_sum": 1}]
    result = paired_bootstrap(baseline, candidate, samples=50)
    assert result["percentile_95_interval"] == pytest.approx([1 / 3, 1 / 3])
    assert result == paired_bootstrap(baseline, candidate, samples=50)
    with pytest.raises(ValueError, match="mismas escenas"):
        paired_bootstrap(baseline, [{**candidate[0], "scene_id": "b"}])


def write_dataset(tmp_path):
    scenes = []
    for index, split in enumerate(("train", "val", "test")):
        image_path, labels_path = tmp_path / f"{split}.tif", tmp_path / f"{split}.geojson"
        transform = from_origin(200_000 + index * 1000, 8_000_000, 1, 1)
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=3,
            dtype="uint8",
            crs="EPSG:32719",
            transform=transform,
        ) as output:
            output.write(np.full((3, 16, 16), 128, dtype="uint8"))
            output.write_mask(np.full((16, 16), 255, dtype="uint8"))
        left, top = transform * (2, 2)
        right, bottom = transform * (8, 8)
        labels = gpd.GeoDataFrame(geometry=[box(left, bottom, right, top)], crs="EPSG:32719")
        labels.to_crs("EPSG:4326").to_file(labels_path, driver="GeoJSON")
        scenes.append(
            {
                "scene_id": split,
                "split": split,
                "image": image_path.name,
                "labels": labels_path.name,
                "labels_source": "Synthetic reference geometry for a unit test",
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"annotation_type": "reference", "scenes": scenes}), encoding="utf-8")
    return manifest_path


def test_dataset_projects_reference_labels_and_keeps_uint8_scale(tmp_path):
    manifest = read_manifest(write_dataset(tmp_path))
    dataset = BuildingDataset(manifest, "train", augment=True)
    image, target = dataset[0]
    assert image.shape == (3, 16, 16)
    assert torch.allclose(image, torch.full_like(image, 128 / 255))
    assert target["masks"].shape == (1, 16, 16)
    assert target["area"].tolist() == [36]
    assert target["labels"].tolist() == [1]
    assert (target["boxes"][:, 2:] > target["boxes"][:, :2]).all()
    with pytest.raises(ValueError, match="no deben aumentarse"):
        BuildingDataset(manifest, "test", augment=True)


def test_manifest_rejects_duplicate_scene_across_splits_and_prediction_labels(tmp_path):
    path = write_dataset(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["scenes"][1]["image"] = manifest["scenes"][0]["image"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="repetido"):
        read_manifest(path)
    manifest["annotation_type"] = "model_predictions"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference"):
        read_manifest(path)


def test_uint16_stretch_excludes_nodata(tmp_path):
    path = tmp_path / "raw.tif"
    data = np.ones((3, 4, 4), dtype="uint16") * 1000
    data[:, :2, :] = np.arange(8).reshape(2, 4) * 100 + 100
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(1, 1, 0.1, 0.1),
    ) as output:
        output.write(data)
        valid = np.zeros((4, 4), dtype="uint8")
        valid[:2] = 255
        output.write_mask(valid)
    with rasterio.open(path) as source:
        pixels, mask = read_rgb(source)
    assert pixels[:, ~mask].sum() == 0
    assert pixels[:, 0, 0].max() == 0
    assert pixels[:, 1, 3].min() == 1


def test_config_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=float("nan"))
    with pytest.raises(ValueError):
        TrainingConfig(match_iou=1.2)


def test_run_selects_on_validation_then_compares_same_test_scenes(tmp_path, monkeypatch):
    from src.inference.model import sha256_file
    from src.training import runner

    class TinyDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.body = torch.nn.Linear(1, 1)
            self.roi_weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, images, targets=None):
            if targets is not None:
                return {"loss_mask": (self.roi_weight - 1).square()}
            height, width = images[0].shape[-2:]
            masks = torch.zeros((1, 1, height, width), device=images[0].device)
            masks[:, :, 2:8, 2:8] = 1
            return [{"masks": masks, "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}]

    def load_tiny(path, device):
        model = TinyDetector()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model, "cpu"

    manifest = write_dataset(tmp_path)
    checkpoint = tmp_path / "initial.pth"
    torch.save(TinyDetector().state_dict(), checkpoint)
    initial_sha = sha256_file(checkpoint)
    monkeypatch.setattr(runner, "load_model", load_tiny)
    output_dir = tmp_path / "run"
    result = runner.run_training(manifest, output_dir, TrainingConfig(epochs=2, bootstrap_samples=10), checkpoint)
    assert result["status"] == "completed" and result["best_epoch"] == 1
    assert result["production_model_modified"] is False and sha256_file(checkpoint) == initial_sha
    stored = torch.load(output_dir / "best_model.pth", weights_only=True)
    assert all(tensor.dtype == torch.float16 for tensor in stored.values() if tensor.is_floating_point())
    events = [json.loads(line) for line in (output_dir / "events.jsonl").read_text().splitlines()]
    last_epoch = max(index for index, event in enumerate(events) if event["event"] == "epoch")
    test_events = [index for index, event in enumerate(events) if event.get("phase", "").endswith("test")]
    assert all(index > last_epoch for index in test_events)
    comparison = json.loads((output_dir / "comparison.json").read_text())
    assert comparison["selection_used_test"] is False
    assert comparison["baseline"]["images"] == comparison["candidate"]["images"] == 1
    with pytest.raises(FileExistsError):
        runner.run_training(manifest, output_dir, TrainingConfig(epochs=1), checkpoint)
