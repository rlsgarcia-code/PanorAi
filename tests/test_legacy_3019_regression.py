import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from panorai.data import EquirectangularImage
from panorai.projections.gnomonic.config import GnomonicConfig
from panorai.projections.gnomonic_projection import GnomonicProjection
from panorai.projections.gnomonic.strategy import GnomonicProjectionStrategy

REFERENCE = json.loads(
    (
        Path(__file__).parent
        / "fixtures/legacy/v3.0.19/reference.json"
    ).read_text(encoding="utf-8")
)


def _expected_with_defined_center() -> np.ndarray:
    expected = np.asarray(
        REFERENCE["direct_gnomonic_projection"]["expected_data"],
        dtype=np.float32,
    ).copy()
    expected[1, 2] = 55.0
    return expected


def test_public_3019_exports_are_preserved() -> None:
    for module_name, legacy_exports in REFERENCE["public_exports"].items():
        code = (
            f"import {module_name} as module; "
            f"required = set({legacy_exports!r}); "
            "assert required.issubset(set(module.__all__))"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
        )


def test_legacy_inverse_projection_center_uses_analytic_limit() -> None:
    config = GnomonicConfig(
        phi1_deg=10,
        lam0_deg=-20,
        fov_deg=70,
        x_points=5,
        y_points=3,
    )
    latitude, longitude = GnomonicProjectionStrategy(
        config
    ).from_projection_to_spherical(np.array([[0.0]]), np.array([[0.0]]))
    np.testing.assert_array_equal(latitude, np.array([[10.0]]))
    np.testing.assert_array_equal(longitude, np.array([[-20.0]]))


def test_direct_3019_gnomonic_projection_preserves_defined_samples() -> None:
    case = REFERENCE["direct_gnomonic_projection"]
    image = np.arange(128, dtype=np.float32).reshape(8, 16)
    with np.errstate(invalid="ignore"):
        result = GnomonicProjection(**case["parameters"]).project(image)
    assert result.shape == tuple(case["expected_shape_hw"])
    assert str(result.dtype) == case["expected_dtype"]
    np.testing.assert_array_equal(result, _expected_with_defined_center())


def test_container_resolution_and_undefined_center_defects_are_fixed() -> None:
    image = np.arange(128, dtype=np.float32).reshape(8, 16)
    with np.errstate(invalid="ignore"):
        face = EquirectangularImage(image).to_gnomonic(
            lat=10, lon=-20, fov=70, x_points=5, y_points=3
        )
    assert face.shape == (3, 5)
    np.testing.assert_array_equal(face.data, _expected_with_defined_center())
