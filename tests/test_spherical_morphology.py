from __future__ import annotations

import math

import numpy as np
import pytest

import panorai as pa
from panorai.data import RasterFieldSpec
from panorai.geometry import erp_pixels_to_rays
from panorai.ops import spherical as S


def _rays(shape_hw: tuple[int, int]) -> np.ndarray:
    rows, columns = np.indices(shape_hw)
    return erp_pixels_to_rays(np.stack((columns, rows), axis=-1), shape_hw)


def test_geodesic_disk_is_intrinsic_and_resolution_aware() -> None:
    disk = S.GeodesicDisk(8.0)
    kernel_size, active = disk.kernel((72, 144))

    assert kernel_size == (9, 9)
    assert active.shape == kernel_size
    assert active[4, 4]
    np.testing.assert_array_equal(active, np.flip(active, axis=(0, 1)))
    assert disk.interface == "panorai-spherical-morphology/v1"
    assert disk.stability == "stable"


def test_binary_dilation_wraps_the_longitude_seam() -> None:
    mask = np.zeros((24, 48), dtype=bool)
    mask[12, 0] = True

    result = S.dilate(mask, S.GeodesicDisk(8.0))

    assert result[12, 0]
    assert result[12, -1]
    assert result.dtype == np.bool_


def test_spherical_cap_has_analytic_dilation_and_erosion_radius() -> None:
    shape = (72, 144)
    rays = _rays(shape)
    centre = np.asarray((0.0, 0.0, 1.0))
    angle_deg = np.rad2deg(np.arccos(np.clip(rays @ centre, -1.0, 1.0)))
    cap = angle_deg <= 20.0
    disk = S.GeodesicDisk(10.0)

    dilated = S.dilate(cap, disk)
    eroded = S.erode(cap, disk)

    # Continuum radii are 30 and 10 degrees.  The finite oracle tolerance is
    # one 2.5-degree pixel-centre step, fixed by this declared lattice.
    assert angle_deg[dilated].max() == pytest.approx(30.0, abs=2.5)
    assert angle_deg[eroded].max() == pytest.approx(10.0, abs=2.5)
    assert angle_deg[~dilated].min() >= 30.0 - 2.5
    assert angle_deg[~eroded].min() >= 10.0 - 2.5


def test_polar_cap_has_analytic_radius_without_a_longitude_singularity() -> None:
    shape = (72, 144)
    rays = _rays(shape)
    north_pole = np.asarray((0.0, 1.0, 0.0))
    angle_deg = np.rad2deg(np.arccos(np.clip(rays @ north_pole, -1.0, 1.0)))
    cap = angle_deg <= 12.0

    dilated = S.dilate(cap, S.GeodesicDisk(8.0))

    # Longitude is singular at the pole, but great-circle distance is not.
    # The continuum radius is 20 degrees; the finite tolerance is one row.
    assert angle_deg[dilated].max() == pytest.approx(20.0, abs=2.5)
    assert angle_deg[~dilated].min() >= 20.0 - 2.5
    np.testing.assert_array_equal(dilated, np.roll(dilated, 37, axis=1))


def test_morphology_obeys_complete_lattice_laws() -> None:
    rng = np.random.default_rng(8301)
    values = rng.normal(size=(18, 36)).astype(np.float64)
    disk = S.GeodesicDisk(12.0)

    dilated = S.dilate(values, disk)
    eroded = S.erode(values, disk)
    opened = S.opening(values, disk)
    closed = S.closing(values, disk)

    assert np.all(dilated >= values)
    assert np.all(eroded <= values)
    np.testing.assert_array_equal(eroded, -S.dilate(-values, disk))
    np.testing.assert_array_equal(S.opening(opened, disk), opened)
    np.testing.assert_array_equal(S.closing(closed, disk), closed)
    assert np.all(opened <= values)
    assert np.all(closed >= values)

    raised = values + 0.25
    assert np.all(S.dilate(raised, disk) >= dilated)
    assert np.all(S.erode(raised, disk) >= eroded)

    chunked = S.dilate(values, disk, max_sampled_elements=300)
    np.testing.assert_array_equal(chunked, dilated)


def test_local_planar_limit_matches_a_euclidean_disk_at_the_equator() -> None:
    shape = (72, 144)
    centre = (shape[0] // 2, shape[1] // 2)
    impulse = np.zeros(shape, dtype=bool)
    impulse[centre] = True
    disk = S.GeodesicDisk(8.0)
    radius_cells = 4
    yy, xx = np.indices((2 * radius_cells + 1, 2 * radius_cells + 1))
    north_deg = (yy - radius_cells) * (180.0 / shape[0])
    east_deg = (xx - radius_cells) * (360.0 / shape[1])
    euclidean_disk = np.hypot(north_deg, east_deg) <= 8.0

    result = S.dilate(impulse, disk)
    radius_y, radius_x = np.asarray(euclidean_disk.shape) // 2
    patch = result[
        centre[0] - radius_y : centre[0] + radius_y + 1,
        centre[1] - radius_x : centre[1] + radius_x + 1,
    ]

    np.testing.assert_array_equal(patch, euclidean_disk)
    assert result.sum() == euclidean_disk.sum()


def test_non_axis_rotation_covariance_is_bounded_by_the_sampling_lattice() -> None:
    shape = (72, 144)
    rays = _rays(shape)
    centre = np.asarray((0.3, 0.4, 0.866), dtype=np.float64)
    centre /= np.linalg.norm(centre)
    cap = np.rad2deg(np.arccos(np.clip(rays @ centre, -1.0, 1.0))) < 12.0
    angle = math.radians(31.0)
    rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(angle), -math.sin(angle)),
            (0.0, math.sin(angle), math.cos(angle)),
        )
    )
    disk = S.GeodesicDisk(8.0)

    dilate_then_rotate = S.rotate(
        S.dilate(cap, disk).astype(np.uint8),
        rotation,
        interpolation="nearest",
    ).astype(bool)
    rotate_then_dilate = S.dilate(
        S.rotate(cap.astype(np.uint8), rotation, interpolation="nearest").astype(bool),
        disk,
    )
    intersection = np.count_nonzero(dilate_then_rotate & rotate_then_dilate)
    union = np.count_nonzero(dilate_then_rotate | rotate_then_dilate)

    assert intersection / union >= 0.92


def test_container_pipeline_grid_support_and_validity_remain_explicit(tmp_path) -> None:
    mask = np.zeros((18, 36), dtype=bool)
    mask[9, 0] = True
    support = np.ones_like(mask)
    support[5, 7] = False
    validity = support.copy()
    validity[11, 19] = False
    panorama = pa.EquirectangularImage.from_field(
        "mask",
        mask,
        spec=RasterFieldSpec.categorical(),
        support=support,
        valid=validity,
    )
    operation = S.Closing(S.GeodesicDisk(10.0), max_sampled_elements=5_000)

    filtered = panorama.filter(operation, field="mask")
    field = pa.Pipeline(pa.OnSphere(operation, input="mask", output="mask"))(panorama)

    np.testing.assert_array_equal(filtered.field("mask"), field.values)
    np.testing.assert_array_equal(field.support, operation.transform_mask(support))
    np.testing.assert_array_equal(field.validity, operation.transform_mask(validity))
    assert field.grid.is_canonical_erp
    assert field.grid.transforms[-1].operation == "closing"
    assert field.kind == "mask"

    restored = pa.Pipeline.load(
        pa.Pipeline(pa.OnSphere(operation, input="mask", output="mask")).save(
            tmp_path / "morphology"
        )
    )
    actual = restored(panorama)
    np.testing.assert_array_equal(actual.values, field.values)
    assert isinstance(restored.named_steps["on_sphere"].operation, S.Closing)


def test_numeric_zero_is_valid_and_nan_remains_explicitly_invalid() -> None:
    values = np.zeros((18, 36), dtype=np.float32)
    values[8, 14] = np.nan
    support = np.ones_like(values, dtype=bool)
    validity = np.ones_like(values, dtype=bool)
    validity[8, 14] = False
    panorama = pa.EquirectangularImage.from_field(
        "response",
        values,
        spec=RasterFieldSpec.continuous(),
        support=support,
        valid=validity,
    )

    filtered = panorama.filter(S.Dilate(12.0), field="response")
    result = pa.Pipeline(
        pa.OnSphere(S.Dilate(12.0), input="response", output="features")
    )(panorama)

    np.testing.assert_array_equal(filtered.field("response"), result.values)
    assert result.values[0, 0] == 0.0
    assert result.validity[0, 0]
    assert not result.validity[8, 14]
    assert not result.validity[8, 13]
    np.testing.assert_array_equal(np.isnan(result.values), ~result.validity)


def test_morphology_rejects_undefined_label_and_vector_orderings() -> None:
    with pytest.raises(TypeError, match="categorical labels"):
        S.dilate(np.zeros((8, 16), dtype=np.int32), 10.0)
    with pytest.raises(ValueError, match="scalar HW"):
        S.dilate(np.zeros((8, 16, 3), dtype=np.float32), 10.0)


def test_torch_morphology_matches_numpy_and_preserves_autograd(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from panorai.ops.spherical import nn as SNN

    rng = np.random.default_rng(8302)
    values = rng.normal(size=(2, 3, 12, 24)).astype(np.float64)
    tensor = torch.tensor(values, requires_grad=True)
    disk = S.GeodesicDisk(20.0)
    model = torch.nn.Sequential(
        SNN.Opening(disk, max_sampled_elements=500),
        SNN.Closing(disk, max_sampled_elements=500),
    ).eval()

    actual = model(tensor)
    expected = np.stack(
        [
            np.stack([S.closing(S.opening(channel, disk), disk) for channel in batch])
            for batch in values
        ]
    )
    np.testing.assert_array_equal(actual.detach().numpy(), expected)
    actual.sum().backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()

    pipeline = pa.Pipeline(pa.OnSphere(model, output="features"))
    restored = pa.Pipeline.load(pipeline.save(tmp_path / "torch-morphology"))
    restored_model = restored.named_steps["on_sphere"].operation
    assert isinstance(restored_model[0], SNN.Opening)
    assert restored_model[0].footprint == disk
    assert restored_model[0].max_sampled_elements == 500


def test_torch_float32_values_and_masks_share_the_numpy_relation() -> None:
    torch = pytest.importorskip("torch")
    from panorai.ops.spherical import nn as SNN

    rng = np.random.default_rng(8303)
    values = rng.normal(size=(1, 1, 9, 17)).astype(np.float32)
    validity = np.ones((9, 17), dtype=bool)
    validity[0, 0] = False
    operation = S.Dilate(S.GeodesicDisk(20.0), max_sampled_elements=200)
    layer = SNN.Dilate(S.GeodesicDisk(20.0), max_sampled_elements=200)

    actual_values, _, actual_validity = layer.forward_with_masks(
        torch.from_numpy(values),
        torch.ones((9, 17), dtype=torch.bool),
        torch.from_numpy(validity),
    )
    expected_values = operation(values[0, 0])
    expected_validity = operation.transform_mask(validity)

    np.testing.assert_array_equal(actual_values.numpy()[0, 0], expected_values)
    np.testing.assert_array_equal(actual_validity.numpy(), expected_validity)
    assert not actual_validity[0, 9]
