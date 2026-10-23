"""Local adaptation contracts: preserve object scale and count merged roofs."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rasterio")

from src.training.metrics import merged_instances  # noqa: E402
from src.training.runner import TrainingConfig, configure_image_scale  # noqa: E402


def test_fusion_counts_two_distinct_roofs_in_one_prediction():
    reference = np.zeros((2, 20, 20), dtype=bool)
    reference[0, 2:7, 2:7] = True
    reference[1, 10:15, 10:15] = True
    merged = np.any(reference, axis=0)[None]
    assert merged_instances(merged, reference) == {
        "merged_predictions": 1, "references_affected_by_merges": 2,
    }
    assert merged_instances(reference, reference)["merged_predictions"] == 0
    assert merged_instances(np.zeros((0, 20, 20)), reference)["merged_predictions"] == 0
    assert merged_instances(merged, np.zeros((0, 20, 20)))["merged_predictions"] == 0


def test_scale_equivalence_between_chips_and_deployed_tiles():
    model = SimpleNamespace(transform=SimpleNamespace(min_size=(800,), max_size=1333))
    config = TrainingConfig(image_scale=1.5625)
    for size, expected in ((96, 150), (128, 200), (192, 300), (512, 800)):
        configure_image_scale(model, torch.zeros((3, size, size)), config)
        assert model.transform.min_size == (expected,)
        assert model.transform.max_size == expected
    configure_image_scale(model, torch.zeros((3, 192, 192)), TrainingConfig())
    assert model.transform.min_size == (800,)


@pytest.mark.parametrize("arguments", [{"image_scale": float("nan")}, {"image_scale": 0},
                                      {"image_scale": 1.5, "batch_size": 2}])
def test_invalid_scale_is_rejected(arguments):
    with pytest.raises(ValueError, match="image_scale"):
        TrainingConfig(**arguments)
