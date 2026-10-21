"""Reject incompatible spatial/runtime settings before opening data or loading ML."""

from dataclasses import FrozenInstanceError

import pytest

from src.inference.config import InferenceParameters


@pytest.mark.parametrize(
    "settings",
    [
        {"analysis_crs": "EPSG:4326"},
        {"analysis_crs": "EPSG:2227"},  # Projected feet cannot be treated as metres.
        {"tile_size": 512, "overlap": 256},
        {"tile_size": 512.5},
        {"min_area_m2": float("nan")},
        {"confidence_threshold": float("nan")},
        {"device": "invalid"},
    ],
)
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        InferenceParameters(**settings)


def test_settings_cannot_change_during_a_run():
    parameters = InferenceParameters()
    with pytest.raises(FrozenInstanceError):
        parameters.mask_threshold = 0.75
