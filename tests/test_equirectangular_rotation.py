from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panorai.core.numpy import spherical_rotate as numpy_spherical_rotate
from panorai.data import EquirectangularImage


ROOT = Path(__file__).resolve().parents[1]


def _rotation(latitude_deg: float, longitude_deg: float) -> np.ndarray:
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(latitude), math.sin(latitude)],
            [0.0, -math.sin(latitude), math.cos(latitude)],
        ]
    )
    yaw = np.array(
        [
            [math.cos(longitude), 0.0, math.sin(longitude)],
            [0.0, 1.0, 0.0],
            [-math.sin(longitude), 0.0, math.cos(longitude)],
        ]
    )
    return yaw @ pitch


def _smooth_ray_field(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    y, x = np.indices(shape_hw, dtype=np.float64)
    longitude = ((x + 0.5) / width) * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - ((y + 0.5) / height) * np.pi
    return np.stack(
        (
            np.sin(longitude) * np.cos(latitude),
            np.sin(latitude),
            np.cos(longitude) * np.cos(latitude),
        ),
        axis=-1,
    )


def test_rotate_longitude_moves_all_modalities_and_masks_exactly() -> None:
    height, width = 10, 20
    image = np.broadcast_to(np.arange(width), (height, width)).astype(np.float64)
    labels = np.broadcast_to(np.arange(width, dtype=np.int16), (height, width))
    support = np.ones((height, width), dtype=bool)
    support[:, 3] = False
    valid = np.ones((height, width), dtype=bool)
    valid[:, 7] = False
    panorama = EquirectangularImage(
        image, support_mask=support, valid=valid
    ).with_labels(labels)

    rotated = panorama.rotate(longitude_deg=360.0 / width)

    np.testing.assert_allclose(
        rotated.image, np.roll(image, 1, axis=1), atol=2e-15, rtol=0.0
    )
    np.testing.assert_array_equal(rotated.labels, np.roll(labels, 1, axis=1))
    np.testing.assert_array_equal(
        rotated.validity("image"), np.roll(valid & support, 1, axis=1)
    )
    np.testing.assert_array_equal(rotated.support("image"), np.roll(support, 1, axis=1))
    assert rotated.labels.dtype == labels.dtype


def test_positive_latitude_is_active_and_uses_documented_rotation_order() -> None:
    rays = _smooth_ray_field((181, 362))
    source_y = rays[..., 1]
    latitude_deg, longitude_deg = 23.0, -37.0

    actual = (
        EquirectangularImage(source_y)
        .rotate(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
        )
        .image
    )
    expected = (rays @ _rotation(latitude_deg, longitude_deg))[..., 1]

    # Linear raster interpolation approximates the analytic sinusoid. Exclude
    # only the outer pixel row, whose ERP ray footprint is border-replicated.
    np.testing.assert_allclose(actual[1:-1], expected[1:-1], atol=2.0e-4, rtol=0.0)


def test_rotated_forward_peak_moves_to_requested_latitude_and_longitude() -> None:
    shape = (181, 362)
    rays = _smooth_ray_field(shape)
    source = np.exp((rays[..., 2] - 1.0) / 0.002)

    rotated = EquirectangularImage(source).rotate(23.0, -37.0).image
    row, column = np.unravel_index(np.argmax(rotated), shape)
    longitude = ((column + 0.5) / shape[1]) * 360.0 - 180.0
    latitude = 90.0 - ((row + 0.5) / shape[0]) * 180.0

    assert latitude == pytest.approx(23.0, abs=180.0 / shape[0])
    assert longitude == pytest.approx(-37.0, abs=360.0 / shape[1])


def test_container_rotation_matches_fundamental_numpy_operation() -> None:
    image = np.random.default_rng(42).normal(size=(31, 62, 3)).astype(np.float32)
    matrix = _rotation(-11.0, 29.0)

    expected = numpy_spherical_rotate(image, matrix)
    actual = EquirectangularImage(image).rotate(-11.0, 29.0).image

    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_fundamental_numpy_nearest_preserves_boolean_dtype() -> None:
    mask = np.zeros((8, 16), dtype=bool)
    mask[:, 4] = True

    rotated = numpy_spherical_rotate(
        mask, _rotation(0.0, 360.0 / 16), interpolation="nearest"
    )

    np.testing.assert_array_equal(rotated, np.roll(mask, 1, axis=1))
    assert rotated.dtype == np.bool_


def test_bilinear_validity_is_strict_but_support_is_raywise() -> None:
    image = np.ones((12, 24), dtype=np.float32)
    valid = np.ones((12, 24), dtype=bool)
    valid[5, 8] = False
    support = np.ones((12, 24), dtype=bool)
    support[6, 10] = False
    panorama = EquirectangularImage(image, valid=valid, support_mask=support)

    rotated = panorama.rotate(longitude_deg=3.0)

    # A sub-pixel bilinear rotation invalidates every sample whose footprint
    # includes the invalid source, while geometric support follows its ray.
    assert np.count_nonzero(~rotated.validity("image")) > 2
    assert np.count_nonzero(~rotated.support("image")) == 1
    assert np.all(rotated.validity("image") <= rotated.support("image"))


def test_rotation_clears_south_cap_metadata_after_support_moves() -> None:
    image = np.ones((18, 36), dtype=np.float32)
    panorama = EquirectangularImage(
        image,
        shadow_angle=30.0,
        shadow_padded=True,
    )

    rotated = panorama.rotate(latitude_deg=50.0)

    assert rotated.shadow_angle == 0.0
    assert rotated.shadow_padded is False
    assert rotated.support_mask is not None
    latitude = np.pi / 2.0 - (np.arange(18) + 0.5) / 18 * np.pi
    solid_angle_weight = np.broadcast_to(np.cos(latitude)[:, None], image.shape)
    original_area = solid_angle_weight[~panorama.support_mask].sum()
    rotated_area = solid_angle_weight[~rotated.support_mask].sum()
    assert rotated.support_mask[-1].any()
    assert not np.array_equal(rotated.support_mask, panorama.support_mask)
    assert rotated_area == pytest.approx(original_area, rel=0.08)


@pytest.mark.parametrize(
    ("latitude_deg", "longitude_deg", "error"),
    [
        (True, 0.0, TypeError),
        (0.0, False, TypeError),
        (float("nan"), 0.0, ValueError),
        (0.0, float("inf"), ValueError),
    ],
)
def test_rotation_rejects_ambiguous_or_non_finite_angles(
    latitude_deg, longitude_deg, error
) -> None:
    panorama = EquirectangularImage(np.ones((8, 16), dtype=np.float32))

    with pytest.raises(error, match="finite real"):
        panorama.rotate(latitude_deg, longitude_deg)


def test_torch_rotation_matches_numpy_and_preserves_autograd() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import spherical_rotate as torch_spherical_rotate

    values = np.random.default_rng(7).normal(size=(2, 3, 19, 38)).astype(np.float32)
    tensor = torch.tensor(values, requires_grad=True)
    matrix = _rotation(9.0, -17.0)

    actual = EquirectangularImage(tensor).rotate(9.0, -17.0).image
    direct = torch_spherical_rotate(tensor, matrix)
    expected = np.stack(
        [
            np.stack([numpy_spherical_rotate(channel, matrix) for channel in batch])
            for batch in values
        ]
    )

    torch.testing.assert_close(actual, direct, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, atol=3e-5, rtol=1e-5
    )
    actual.square().mean().backward()
    assert tensor.grad is not None
    assert bool(torch.isfinite(tensor.grad).all())


def test_torch_nearest_preserves_batched_integer_labels_exactly() -> None:
    torch = pytest.importorskip("torch")
    from panorai.data import RasterFieldSpec

    labels = torch.arange(2 * 8 * 16, dtype=torch.int64).reshape(2, 8, 16)
    panorama = EquirectangularImage.from_field(
        "classes", labels, spec=RasterFieldSpec.categorical()
    )

    rotated = panorama.rotate(longitude_deg=360.0 / 16).field("classes")

    torch.testing.assert_close(rotated, torch.roll(labels, 1, dims=-1))
    assert rotated.dtype == torch.int64


def test_torch_accepts_one_rotation_per_batch_element() -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import spherical_rotate as torch_spherical_rotate

    values = torch.arange(2 * 1 * 8 * 16, dtype=torch.int64).reshape(2, 1, 8, 16)
    rotations = torch.tensor(
        np.stack(
            (
                _rotation(0.0, 360.0 / 16),
                _rotation(0.0, -2.0 * 360.0 / 16),
            )
        ),
        dtype=torch.float64,
    )

    actual = torch_spherical_rotate(values, rotations, interpolation="nearest")
    expected = torch.stack(
        (
            torch.roll(values[0], shifts=1, dims=-1),
            torch.roll(values[1], shifts=-2, dims=-1),
        )
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_torch_rotation_validation_does_not_require_linalg_det(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from panorai.core.torch import SphericalRotationPlan

    def unsupported(*args, **kwargs):
        raise NotImplementedError("linalg.det is unavailable on MPS")

    monkeypatch.setattr(torch.linalg, "det", unsupported)
    like = torch.zeros(1, 1, 8, 16)
    plan = SphericalRotationPlan.from_rotation(
        (8, 16), torch.eye(3), like=like
    )
    assert plan.grid.shape == (1, 8, 16, 2)

    reflection = torch.diag(torch.tensor((1.0, 1.0, -1.0)))
    with pytest.raises(ValueError, match="determinant \\+1"):
        SphericalRotationPlan.from_rotation(
            (8, 16), reflection, like=like
        )


def test_object_rotation_builds_one_numpy_plan_for_all_aligned_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from panorai.core.numpy import SphericalRotationPlan

    calls = 0
    original = SphericalRotationPlan.from_rotation.__func__

    def counted(cls, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(
        SphericalRotationPlan,
        "from_rotation",
        classmethod(counted),
    )
    image = np.ones((12, 24, 3), dtype=np.float32)
    labels = np.arange(12 * 24, dtype=np.int16).reshape(12, 24)
    support = np.ones((12, 24), dtype=bool)
    panorama = EquirectangularImage(image, support_mask=support).with_labels(labels)

    rotated = panorama.rotate(latitude_deg=11.0, longitude_deg=17.0)

    assert rotated.image.shape == image.shape
    assert rotated.labels.shape == labels.shape
    assert calls == 1


def test_numpy_container_rotation_does_not_import_optional_torch() -> None:
    script = r"""
import builtins
import sys
import numpy as np

original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('NumPy ERP rotation attempted to import Torch')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded
from panorai.data import EquirectangularImage
image = np.ones((8, 16, 3), dtype=np.float32)
rotated = EquirectangularImage(image).rotate(7.0, 13.0)
assert rotated.image.shape == image.shape
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
