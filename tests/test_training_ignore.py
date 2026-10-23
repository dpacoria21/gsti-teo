"""Uncertain-region contracts; synthetic CPU models only, no real fitting."""

# ruff: noqa: E402
import copy
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import box
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import maskrcnn_loss
from torchvision.ops import MultiScaleRoIAlign

from src.training.dataset import BuildingDataset, read_manifest, training_supervision
from src.training.ignore import IgnoreRoIHeads, boxes_touch_ignore, enable_training_ignore, supervised_mask_loss


def tiny_model():
    torch.set_num_threads(2)
    backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, stride=2, padding=1), torch.nn.ReLU())
    backbone.out_channels = 8
    return MaskRCNN(
        backbone,
        num_classes=None,
        min_size=32,
        max_size=32,
        rpn_anchor_generator=AnchorGenerator(((8,),), ((1.0,),)),
        rpn_pre_nms_top_n_train=20,
        rpn_pre_nms_top_n_test=20,
        rpn_post_nms_top_n_train=10,
        rpn_post_nms_top_n_test=10,
        rpn_batch_size_per_image=16,
        box_roi_pool=MultiScaleRoIAlign(["0"], 3, 2),
        box_head=TwoMLPHead(8 * 3 * 3, 16),
        box_predictor=FastRCNNPredictor(16, 2),
        box_batch_size_per_image=16,
        box_detections_per_img=10,
        mask_roi_pool=MultiScaleRoIAlign(["0"], 4, 2),
        mask_head=MaskRCNNHeads(8, (8,), 1),
        mask_predictor=MaskRCNNPredictor(8, 8, 2),
    )


def target(empty=False, supervision=True):
    masks = torch.zeros((0 if empty else 1, 32, 32), dtype=torch.uint8)
    if not empty:
        masks[0, 4:12, 4:12] = 1
    return {
        "boxes": torch.empty((0, 4)) if empty else torch.tensor([[4.0, 4.0, 12.0, 12.0]]),
        "labels": torch.ones(len(masks), dtype=torch.int64),
        "masks": masks,
        "supervision_mask": torch.full((32, 32), supervision, dtype=torch.bool),
    }


def test_whole_instance_exclusion_propagates_and_tiny_labels_are_not_background():
    masks = np.zeros((4, 10, 10), dtype=np.uint8)
    masks[0, 1:5, 1:5] = 1
    masks[1, 4:7, 4:7] = 1  # overlaps excluded roof, so must also be excluded
    masks[2, 7:10, 0:3] = 1
    masks[3, 9, 9] = 1
    valid = np.ones((10, 10), dtype=bool)
    valid[1, 1] = False
    retained, supervised = training_supervision(masks, valid)
    assert np.array_equal(retained, masks[2:3])
    assert not supervised[masks[[0, 1, 3]].any(axis=0)].any()
    assert supervised[masks[2].astype(bool)].all()


@pytest.fixture
def local_scene(tmp_path):
    image = tmp_path / "image.tif"
    transform = from_origin(0, 16, 1, 1)
    with rasterio.open(
        image, "w", driver="GTiff", width=16, height=16, count=3, dtype="uint8", crs=32719, transform=transform
    ) as dst:
        pixels = np.full((3, 16, 16), 127, dtype=np.uint8)
        pixels[:, 10:14, 2:6] = 255
        dst.write(pixels)
    labels, ignored = tmp_path / "labels.geojson", tmp_path / "ignore.geojson"
    gpd.GeoDataFrame(geometry=[box(2, 2, 6, 6), box(9, 9, 14, 14)], crs=32719).to_file(labels, driver="GeoJSON")
    gpd.GeoDataFrame(geometry=[box(12, 12, 13, 13)], crs=32719).to_file(ignored, driver="GeoJSON")
    return {
        "annotation_type": "reference",
        "scenes": [
            {
                "scene_id": "local",
                "split": "train",
                "image": str(image),
                "labels": str(labels),
                "labels_source": "synthetic manual fixture",
                "ignore_regions": str(ignored),
            }
        ],
    }


def test_dataset_opt_in_preserves_rgb_and_complete_clean_roof_with_aligned_augmentation(local_scene, monkeypatch):
    with pytest.raises(ValueError, match="No se admite"):
        BuildingDataset(local_scene, "train")
    image, result = BuildingDataset(local_scene, "train", allow_training_ignore=True)[0]
    assert result["masks"].shape == (1, 16, 16)
    assert result["masks"].sum() == 16
    assert result["boxes"].tolist() == [[2, 10, 6, 14]]
    assert not result["supervision_mask"][2:7, 9:14].any()
    assert result["supervision_mask"].dtype == torch.bool
    assert torch.equal(result["valid_mask"], result["supervision_mask"])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([1]))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    augmented, augmented_target = BuildingDataset(local_scene, "train", augment=True, allow_training_ignore=True)[0]
    assert torch.equal(augmented, torch.rot90(image, 1, (1, 2)).flip((1, 2)))
    assert torch.equal(augmented_target["masks"], torch.rot90(result["masks"], 1, (1, 2)).flip((1, 2)))
    expected = torch.rot90(result["supervision_mask"], 1, (0, 1)).flip((0, 1))
    assert torch.equal(augmented_target["supervision_mask"], expected)
    assert torch.equal(augmented_target["valid_mask"], expected)


def test_development_manifest_requires_exact_splits_before_accessing_files(local_scene, tmp_path):
    manifest = copy.deepcopy(local_scene)
    val = copy.deepcopy(manifest["scenes"][0])
    val.update(scene_id="val", split="val", image="never-open-test.tif", labels="never-open-test.geojson")
    manifest["scenes"].append(val)
    path = tmp_path / "development.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="train, val, test"):
        read_manifest(path)
    with pytest.raises(ValueError, match="No existe el archivo"):
        read_manifest(path, required_splits=("train", "val"))
    for key, filename in (("image", "validation.tif"), ("labels", "validation.geojson")):
        shutil.copyfile(manifest["scenes"][0][key], tmp_path / filename)
        val[key] = filename
    path.write_text(json.dumps(manifest), encoding="utf-8")
    resolved = read_manifest(path, required_splits=("train", "val"))
    assert {scene["split"] for scene in resolved["scenes"]} == {"train", "val"}
    val["split"] = "test"
    val["image"] = "not-accessed.tif"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactamente"):
        read_manifest(path, required_splits=("train", "val"))


def test_dataset_all_ignored_area_produces_empty_instances(local_scene):
    ignored = local_scene["scenes"][0]["ignore_regions"]
    gpd.GeoDataFrame(geometry=[box(0, 0, 16, 16)], crs=32719).to_file(ignored, driver="GeoJSON")
    image, item = BuildingDataset(local_scene, "train", allow_training_ignore=True)[0]
    assert image.any(), "Ignore must not paint the original RGB black."
    assert item["masks"].shape == (0, 16, 16) and item["boxes"].shape == (0, 4)
    assert not item["supervision_mask"].any()


def test_ignore_transform_resizes_and_does_not_mutate_input_target():
    model = enable_training_ignore(tiny_model())
    image = torch.rand(3, 16, 16)
    original = {
        "boxes": torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
        "masks": torch.ones((1, 16, 16), dtype=torch.uint8),
        "supervision_mask": torch.ones((16, 16), dtype=torch.bool),
    }
    original["supervision_mask"][:, :8] = False
    batch, resized = model.transform([image], [original])
    assert batch.image_sizes == [(32, 32)]
    assert resized[0]["boxes"].tolist() == [[0, 0, 16, 16]]
    assert resized[0]["supervision_mask"].shape == (32, 32)
    assert not resized[0]["supervision_mask"][:, :16].any()
    assert resized[0]["supervision_mask"][:, 16:].all()
    assert original["supervision_mask"].shape == (16, 16)


def test_uncertain_anchors_and_proposals_never_become_background():
    model = enable_training_ignore(tiny_model())
    item = target(empty=True)
    item["supervision_mask"][10:20, 10:20] = False
    boxes = torch.tensor([[0.0, 0.0, 5.0, 5.0], [10.0, 10.0, 15.0, 15.0], [9.9, 9.9, 10.1, 10.1]])
    assert boxes_touch_ignore(boxes, item["supervision_mask"]).tolist() == [False, True, True]
    labels, _ = model.rpn.assign_targets_to_anchors([boxes], [item])
    assert labels[0].tolist() == [0, -1, -1]
    selected, _, sampled_labels, _ = model.roi_heads.select_training_samples([boxes], [item])
    assert selected[0].tolist() == boxes[:1].tolist()
    assert sampled_labels[0].tolist() == [0]
    ignored = target(empty=True, supervision=False)
    outside = torch.tensor([[35.0, 35.0, 40.0, 40.0], [-10.0, -10.0, -5.0, -5.0]])
    labels, _ = model.rpn.assign_targets_to_anchors([outside], [ignored])
    assert labels[0].tolist() == [-1, -1], "Padding cannot create supervised negatives in an ignored scene."


def test_mask_loss_has_zero_gradient_at_uncertain_roi_pixels_and_matches_stock_without_ignore():
    proposals = [torch.tensor([[0.0, 0.0, 8.0, 8.0]])]
    masks = [torch.ones((1, 8, 8), dtype=torch.uint8)]
    labels, indices = [torch.tensor([1])], [torch.tensor([0])]
    logits = torch.randn((1, 2, 8, 8), requires_grad=True)
    full = [torch.ones((8, 8), dtype=torch.bool)]
    stock = maskrcnn_loss(logits, proposals, masks, labels, indices)
    actual = supervised_mask_loss(logits, proposals, masks, labels, indices, full)
    assert torch.allclose(actual, stock)
    uncertain = full[0].clone()
    uncertain[:, 4:] = False
    loss = supervised_mask_loss(logits, proposals, masks, labels, indices, [uncertain])
    loss.backward()
    assert not logits.grad[0, 0].any()
    assert not logits.grad[0, 1, :, 4:].any()
    assert logits.grad[0, 1, :, :3].abs().sum() > 0
    changed = logits.detach().clone()
    changed[:, :, :, 4:] += 100
    assert torch.allclose(loss, supervised_mask_loss(changed, proposals, masks, labels, indices, [uncertain]))


@pytest.mark.parametrize("all_ignored", [False, True])
def test_negative_and_all_ignored_images_have_finite_losses_and_backward(all_ignored):
    model = enable_training_ignore(tiny_model()).train()
    losses = model([torch.rand(3, 32, 32)], [target(empty=True, supervision=not all_ignored)])
    assert set(losses) == {"loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg", "loss_mask"}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if all_ignored:
        assert all(value == 0 for value in losses.values())
        assert all(parameter.grad is None or not parameter.grad.any() for parameter in model.parameters())
    else:
        assert losses["loss_classifier"] > 0 and losses["loss_objectness"] > 0
        assert losses["loss_mask"] == 0


def test_installer_preserves_state_parameters_eval_and_other_models():
    model, unaffected = tiny_model().eval(), tiny_model()
    image = torch.rand(3, 32, 32)
    state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    identities = {key: id(parameter) for key, parameter in model.named_parameters()}
    with torch.inference_mode():
        before = model([image])[0]
        assert enable_training_ignore(model) is model
        assert enable_training_ignore(model) is model
        after = model([image])[0]
    assert not isinstance(unaffected.roi_heads, IgnoreRoIHeads)
    assert identities == {key: id(parameter) for key, parameter in model.named_parameters()}
    assert state.keys() == model.state_dict().keys()
    assert all(torch.equal(tensor, model.state_dict()[key]) for key, tensor in state.items())
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_training_losses_match_stock_with_fully_supervised_target():
    original = tiny_model().train()
    candidate = enable_training_ignore(copy.deepcopy(original)).train()
    image, item = torch.rand(3, 32, 32), target()
    stock_target = {key: value for key, value in item.items() if key != "supervision_mask"}
    torch.manual_seed(123)
    baseline = original([image], [stock_target])
    torch.manual_seed(123)
    actual = candidate([image], [item])
    assert actual.keys() == baseline.keys()
    for name in actual:
        assert torch.allclose(actual[name], baseline[name], atol=1e-7), name
