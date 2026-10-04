from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import pytest

import panorai as pa
from panorai.blenders.average import AverageBlender
from panorai.data._workflow import (
    back_project_modality,
    blend_reprojected,
    union_masks,
)


def _rgb(height=19, width=37):
    rng = np.random.default_rng(27)
    return rng.random((height, width, 3), dtype=np.float32)


def _full_workflow_oracle(views, name, method):
    workflow = views._workflow
    metadata = views[0]._workflow_metadata[name]
    values = []
    valid_masks = []
    supports = []
    specs = []
    with warnings.catch_warnings():
        # The frozen full-grid oracle computes irrelevant maps outside support;
        # exact polar views can therefore warn while casting those NaNs.
        warnings.simplefilter("ignore", RuntimeWarning)
        for face in views:
            value, support, valid = back_project_modality(
                face._workflow_data()[name],
                face._workflow_metadata[name]["validity"],
                face.spec,
                workflow["erp_shape"],
                kind=metadata["kind"],
                depth_policy=workflow["depth_policy"],
                min_valid_weight=workflow["min_valid_weight"],
                projector_template=workflow["projector"],
            )
            values.append(value)
            valid_masks.append(valid)
            supports.append(support)
            specs.append(face.spec)
    data, valid = blend_reprojected(
        values,
        valid_masks,
        specs,
        workflow["erp_shape"],
        method,
    )
    return data, valid, union_masks(supports)


@pytest.mark.parametrize(
    ("layout", "kwargs", "expected_count"),
    [
        ("cube", {"rotations": ((0, 7),)}, 12),
        ("icosahedron", {}, 12),
        ("fibonacci", {"count": 17}, 17),
        ("spiral", {"count": 23}, 23),
    ],
)
@pytest.mark.parametrize(
    ("name", "method", "tolerance"),
    [
        ("image", "average", 0.0),
        ("image", "closest", 0.0),
        ("image", "gaussian", 2e-7),
        ("depth", "average", 0.0),
        ("labels", "closest", 0.0),
    ],
)
def test_workflow_sparse_batch_matches_full_frame_oracle(
    layout, kwargs, expected_count, name, method, tolerance
):
    rng = np.random.default_rng(31)
    depth = rng.random((19, 37), dtype=np.float32) + 0.5
    valid = rng.random((19, 37)) > 0.15
    labels = rng.integers(0, 11, size=(19, 37), dtype=np.int16)
    panorama = (
        pa.EquirectangularImage(_rgb())
        .with_depth(depth, valid=valid)
        .with_labels(labels)
    )
    views = panorama.views(
        layout,
        size=(9, 13),
        fov=(101, 73),
        **kwargs,
    )
    assert len(views) == expected_count

    expected, expected_valid, expected_support = _full_workflow_oracle(
        views, name, method
    )
    actual = views.reconstruct(modalities=name, blend=method)
    actual_value = getattr(actual, name)

    if tolerance == 0.0:
        assert np.array_equal(actual_value, expected, equal_nan=True)
    else:
        assert np.allclose(
            actual_value, expected, atol=tolerance, rtol=0.0, equal_nan=True
        )
    assert np.array_equal(actual.validity(name), expected_valid)
    assert np.array_equal(actual.support_mask, expected_support)


def test_validity_normalized_depth_batch_matches_full_frame_oracle():
    depth = np.linspace(1.0, 3.0, 17 * 35, dtype=np.float32).reshape(17, 35)
    valid = np.ones((17, 35), dtype=bool)
    valid[3:8, 7:13] = False
    views = (
        pa.EquirectangularImage(_rgb(17, 35))
        .with_depth(depth, valid=valid)
        .views(
            "fibonacci",
            count=11,
            size=(8, 12),
            fov=(96, 68),
            depth_policy="renormalize",
            min_valid_weight=0.65,
        )
    )
    expected, expected_valid, expected_support = _full_workflow_oracle(
        views, "depth", "average"
    )
    actual = views.reconstruct(modalities="depth", blend="average")
    assert np.array_equal(actual.depth, expected, equal_nan=True)
    assert np.array_equal(actual.validity("depth"), expected_valid)
    assert np.array_equal(actual.support_mask, expected_support)


def _full_legacy_oracle(face_set, shape, preserve_dtype):
    projected = [face.to_equirectangular(shape, return_mask=True) for face in face_set]
    return face_set.blend_channels(
        [item[0] for item in projected],
        preserve_dtype,
        AverageBlender(),
        masks=[item[1] for item in projected],
    )


@pytest.mark.parametrize(
    ("sampler", "kwargs", "expected_count"),
    [
        ("cube", {}, 6),
        ("icosahedron", {}, 12),
        ("fibonacci", {"n_points": 19}, 19),
        ("spiral", {"n_points": 27}, 27),
    ],
)
def test_legacy_sampler_batch_is_exact_and_selective(sampler, kwargs, expected_count):
    image = pa.EquirectangularImage(_rgb(23, 46))
    image.attach_sampler(sampler, **kwargs)
    views = image.to_gnomonic_face_set(
        fov=82,
        rotations=[] if sampler != "cube" else [(0, 5)],
    )
    if sampler == "cube":
        expected_count *= 2
    assert len(views) == expected_count

    expected = _full_legacy_oracle(views, (23, 46), False)
    actual = views.to_equirectangular((23, 46), preserve_dtype=False)
    assert np.array_equal(actual.data, expected.data)
    assert np.array_equal(actual.support_mask, expected.support_mask)

    plan = views._legacy_batch_back_plan((23, 46))
    full_samples = len(views) * 23 * 46
    assert sum(len(item[0]) for item in plan) < full_samples


def test_legacy_multichannel_singleton_channel_matches_full_oracle():
    data = {
        "rgb": (_rgb(11, 22) * 255).astype(np.uint8),
        "depth": np.ones((11, 22), dtype=np.float32),
    }
    panorama = pa.EquirectangularImage(data)
    panorama.attach_sampler("fibonacci", n_points=9)
    views = panorama.to_gnomonic_face_set(fov=78)
    expected = _full_legacy_oracle(views, (11, 22), True)
    actual = views.to_equirectangular((11, 22), preserve_dtype=True)
    assert set(actual.data) == {"rgb", "depth"}
    for name in actual.data:
        assert np.array_equal(actual.data[name], expected.data[name])
    assert np.array_equal(actual.support_mask, expected.support_mask)


def test_legacy_selective_plan_preserves_custom_spherical_bounds():
    first = pa.GnomonicFace(np.full((9, 11), 2.0, dtype=np.float32), 5, 20, 70)
    second = pa.GnomonicFace(np.full((9, 11), 6.0, dtype=np.float32), -8, -35, 70)
    for face in (first, second):
        face.projection.config.update(
            lon_min=-120.0,
            lon_max=140.0,
            lat_min=-55.0,
            lat_max=65.0,
        )
    views = pa.GnomonicFaceSet([first, second])
    expected = _full_legacy_oracle(views, (17, 29), False)
    actual = views.to_equirectangular((17, 29), preserve_dtype=False)
    assert np.array_equal(actual.data, expected.data)
    assert np.array_equal(actual.support_mask, expected.support_mask)


def test_legacy_selective_average_rejects_supported_nonfinite_values():
    invalid = pa.GnomonicFace(np.full((7, 9), np.nan, dtype=np.float32), 0, 0, 80)
    valid = pa.GnomonicFace(np.ones((7, 9), dtype=np.float32), 0, 0, 80)
    views = pa.GnomonicFaceSet([invalid, valid])
    with pytest.raises(ValueError, match="non-finite values marked as valid"):
        views.to_equirectangular((13, 26), preserve_dtype=False)


def test_custom_blender_keeps_full_frame_compatibility_fallback():
    class RecordingBlender:
        def __init__(self):
            self.shapes = None

        def blend(self, images, masks, return_mask=False):
            self.shapes = [image.shape for image in images]
            return AverageBlender().blend(images, masks, return_mask=return_mask)

    views = pa.EquirectangularImage(_rgb(13, 26)).views(
        "fibonacci", count=7, size=(7, 9)
    )
    blender = RecordingBlender()
    result = views.reconstruct(modalities="image", blend=blender)
    assert blender.shapes == [(13, 26, 3)] * 7
    assert result.image.shape == (13, 26, 3)


def test_batch_plan_cache_is_bounded_reused_and_thread_safe():
    views = pa.EquirectangularImage(_rgb(15, 30)).views(
        "fibonacci", count=8, size=(7, 11), fov=(89, 67)
    )
    first = views.reconstruct().image
    assert len(views._batch_back_plans) == 1
    first_plan = next(iter(views._batch_back_plans.values()))
    second = views.reconstruct().image
    assert next(iter(views._batch_back_plans.values())) is first_plan
    assert np.array_equal(first, second, equal_nan=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: views.reconstruct().image, range(8)))
    assert all(np.array_equal(item, first, equal_nan=True) for item in results)

    values = [face.image for face in views]
    specs = [face.spec for face in views]
    for height in range(16, 21):
        views._get_batch_back_plan((height, height * 2), values, specs)
    assert len(views._batch_back_plans) == 4


@pytest.mark.parametrize("method", ["average", "closest", "gaussian"])
@pytest.mark.parametrize("shape", [(13, 27), (3, 13, 27), (2, 3, 13, 27)])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_batch_preserves_layout_parity_and_autograd(method, shape, dtype_name):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    source = torch.rand(shape, dtype=dtype, requires_grad=True)
    views = pa.EquirectangularImage(source).views(
        "fibonacci", count=7, size=(7, 9), fov=(91, 69)
    )
    expected, expected_valid, expected_support = _full_workflow_oracle(
        views, "image", method
    )
    actual = views.reconstruct(blend=method)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(
        actual.image, expected, atol=tolerance, rtol=0, equal_nan=True
    )
    assert torch.equal(actual.validity("image"), expected_valid)
    assert torch.equal(actual.support_mask, expected_support)
    actual.image.nan_to_num().sum().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
