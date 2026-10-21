import numpy as np
import pytest

pytest.importorskip("rasterio")
from affine import Affine

from scripts.georeference_capture import fit_affine


def points_for(affine):
    pixels = [(0, 0), (1000, 0), (1000, 700), (0, 700), (200, 300), (800, 500)]
    result = []
    for index, (column, row) in enumerate(pixels):
        x, y = affine * (column + 0.5, row + 0.5)
        result.append(
            {"id": str(index), "column": column, "row": row, "x": x, "y": y, "role": "fit" if index < 4 else "check"}
        )
    return result


def test_affine_recovers_rotation_translation_and_pixel_centre_convention():
    expected = Affine(0.12, 0.004, 227000, -0.003, -0.12, 8185000)
    fitted, residuals, stats = fit_affine(points_for(expected))
    np.testing.assert_allclose(tuple(fitted), tuple(expected), atol=1e-8)
    assert stats["fit"]["count"] == 4
    assert stats["check"]["count"] == 2
    assert max(record["residual_m"] for record in residuals) < 1e-7
    # Input (0,0) denotes a pixel centre, not the corner used by GDAL.
    assert fitted * (0.5, 0.5) == pytest.approx(expected * (0.5, 0.5), abs=1e-8)


def test_independent_check_error_rejects_good_fit():
    points = points_for(Affine(0.5, 0, 227000, 0, -0.5, 8185000))
    points[-1]["x"] += 4
    with pytest.raises(ValueError, match="check 4.000"):
        fit_affine(points)


def test_fit_outlier_rejected_even_with_other_correct_points():
    points = points_for(Affine(0.5, 0, 227000, 0, -0.5, 8185000))
    points[0]["x"] += 20
    with pytest.raises(ValueError, match="error máximo fit"):
        fit_affine(points)


def test_collinear_control_points_are_rejected():
    points = points_for(Affine(0.5, 0, 227000, 0, -0.5, 8185000))
    for index, point in enumerate(points[:4]):
        point["column"], point["row"] = index * 100, index * 100
    with pytest.raises(ValueError, match="colineales"):
        fit_affine(points)


def test_missing_independent_checks_are_rejected():
    points = points_for(Affine(0.5, 0, 227000, 0, -0.5, 8185000))
    points[-1]["role"] = "fit"
    with pytest.raises(ValueError, match="4 puntos fit y 2 puntos check"):
        fit_affine(points)


def test_error_limit_cannot_be_relaxed_to_hide_bad_georeferencing():
    with pytest.raises(ValueError, match="no superar 3"):
        fit_affine(points_for(Affine(0.5, 0, 227000, 0, -0.5, 8185000)), max_error_m=10)
