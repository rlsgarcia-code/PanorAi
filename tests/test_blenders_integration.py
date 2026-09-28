import numpy as np
import pytest

from panorai.blenders import (
    AverageBlender,
    BundleAdjustmentBlender,
    ClosestBlender,
    FeatheringBlender,
    GaussianBlender,
    HuberBlender,
    HuberNoConfidenceBlender,
    HuberSpatialBlender,
    OverlapCounterBlender,
    OverlapStdBlender,
    OverlapStdFeatheredBlender,
)
from panorai.data import GnomonicFace, GnomonicFaceSet
from panorai.geometry import GnomonicSpec
from panorai.projections.gnomonic_projection import GnomonicProjection


def test_closest_blender_treats_valid_black_as_data():
    blender = ClosestBlender()
    black = np.zeros((3, 3, 3), dtype=np.float32)
    bright = np.full((3, 3, 3), 10.0, dtype=np.float32)
    black_support = np.ones((3, 3), dtype=bool)
    bright_support = np.zeros((3, 3), dtype=bool)
    bright_support[0, :] = True

    output, support = blender.blend(
        [black, bright], [black_support, bright_support], return_mask=True
    )

    assert np.array_equal(support, black_support)
    assert np.array_equal(output[1:, :], black[1:, :])
    assert np.isfinite(output).all()


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 2)])
def test_huber_blender_preserves_input_shape_and_explicit_support(shape):
    blender = HuberBlender()
    first = np.zeros(shape, dtype=np.float32)
    second = np.full(shape, 2.0, dtype=np.float32)
    first_mask = np.ones(shape[:2], dtype=bool)
    second_mask = np.ones(shape[:2], dtype=bool)
    second_mask[0, 0] = False

    output, support = blender.blend(
        [first, second], [first_mask, second_mask], delta=10.0, return_mask=True
    )

    assert output.shape == shape
    assert support.shape == shape[:2]
    assert np.allclose(output[0, 0], 0.0)
    assert np.allclose(output[1, 1], 1.0, atol=1e-4)


@pytest.mark.parametrize(
    "blender",
    [
        AverageBlender(),
        ClosestBlender(),
        FeatheringBlender(),
        HuberBlender(),
        HuberNoConfidenceBlender(),
        HuberSpatialBlender(),
    ],
    ids=lambda blender: type(blender).__name__,
)
def test_fusion_blenders_preserve_shape_and_union_support(blender):
    images = [
        np.zeros((3, 4, 2), dtype=np.float32),
        np.full((3, 4, 2), 4.0, dtype=np.float32),
    ]
    masks = [
        np.array(
            [[True, True, False, False], [True, True, False, False], [False] * 4]
        ),
        np.array(
            [[False, True, True, False], [False, True, True, False], [False] * 4]
        ),
    ]

    output, support = blender.blend(images, masks, return_mask=True)

    assert output.shape == images[0].shape
    assert np.array_equal(support, masks[0] | masks[1])
    assert np.isfinite(output).all()
    assert np.all(output[~support] == 0)


def test_diagnostic_blenders_use_masks_not_values():
    images = [
        np.zeros((2, 2, 3), dtype=np.float32),
        np.full((2, 2, 3), 2.0, dtype=np.float32),
    ]
    masks = [
        np.array([[True, True], [False, False]]),
        np.array([[True, False], [True, False]]),
    ]

    count, count_support = OverlapCounterBlender().blend(
        images, masks, return_mask=True
    )
    std, std_support = OverlapStdBlender().blend(
        images, masks, return_mask=True
    )
    feathered_std, feathered_support = OverlapStdFeatheredBlender().blend(
        images, masks, return_mask=True
    )

    expected_support = masks[0] | masks[1]
    assert np.array_equal(count[..., 0], [[2, 1], [1, 0]])
    assert np.isclose(std[0, 0, 0], 1.0)
    assert std[0, 1, 0] == 0
    assert feathered_std.shape == (2, 2, 1)
    assert np.array_equal(count_support, expected_support)
    assert np.array_equal(std_support, expected_support)
    assert np.array_equal(feathered_support, expected_support)


def test_valid_nonfinite_samples_fail_instead_of_becoming_zero():
    image = np.array([[np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite values marked as valid"):
        AverageBlender().blend([image], [np.array([[True]])])


def test_experimental_bundle_adjustment_has_explicit_scalar_contract():
    images = [
        np.ones((2, 2), dtype=np.float32),
        np.full((2, 2), 1.1, dtype=np.float32),
    ]
    masks = [
        np.ones((2, 2), dtype=bool),
        np.array([[True, True], [True, False]]),
    ]

    output, support = BundleAdjustmentBlender().blend(
        images, masks, return_mask=True
    )

    assert output.shape == images[0].shape
    assert np.array_equal(support, masks[0] | masks[1])
    assert np.isfinite(output).all()
    with pytest.raises(ValueError, match="scalar"):
        BundleAdjustmentBlender().blend(
            [images[0][..., None]], [masks[0]]
        )


def test_gaussian_blender_uses_real_gnomonic_projector_and_masks():
    face_shape = (5, 7)
    eq_shape = (10, 20)
    template_spec = GnomonicSpec(
        center_lat_deg=0.0,
        center_lon_deg=0.0,
        hfov_deg=80.0,
        vfov_deg=80.0,
        output_shape_hw=face_shape,
    )
    projector = GnomonicProjection(spec=template_spec)
    tangent_points = [(0.0, -35.0), (0.0, 35.0)]
    images = []
    masks = []
    for value, (lat, lon) in zip((10.0, 20.0), tangent_points):
        spec = GnomonicSpec(lat, lon, 80.0, 80.0, 0.0, face_shape)
        local_projector = GnomonicProjection(spec=spec)
        face = np.full(face_shape + (3,), value, dtype=np.float32)
        projected, support = local_projector.back_project(
            face, eq_shape, return_mask=True
        )
        images.append(projected)
        masks.append(support)

    blender = GaussianBlender(
        fov_deg=80.0,
        projector=projector,
        tangent_points=tangent_points,
        sig=0.5,
    )
    output, support = blender.blend(images, masks, return_mask=True)

    assert output.shape == images[0].shape
    assert support.shape == eq_shape
    assert support.any()
    assert (~support).any()
    assert np.isfinite(output).all()
    assert np.all(output[~support] == 0)
    only_first = masks[0] & ~masks[1]
    only_second = masks[1] & ~masks[0]
    assert np.allclose(output[only_first], 10.0)
    assert np.allclose(output[only_second], 20.0)


def test_legacy_back_projection_mask_excludes_border_fill():
    face_shape = (7, 9)
    eq_shape = (18, 36)
    projector = GnomonicProjection(
        phi1_deg=0.0,
        lam0_deg=0.0,
        fov_deg=80.0,
        x_points=face_shape[1],
        y_points=face_shape[0],
        lat_points=eq_shape[0],
        lon_points=eq_shape[1],
    )
    face = np.full(face_shape + (3,), 7.0, dtype=np.float32)

    projected, support = projector.back_project(
        face, eq_shape, return_mask=True
    )

    assert support.any()
    assert (~support).any()
    assert np.allclose(projected[support], 7.0)
    assert np.all(projected[~support] == 0)


def test_faceset_blending_uses_real_back_projection_support_for_valid_zero():
    black = GnomonicFace(np.zeros((5, 5, 3), dtype=np.float32), 0.0, 0.0, 80.0)
    white = GnomonicFace(np.full((5, 5, 3), 2.0, dtype=np.float32), 0.0, 0.0, 80.0)
    faces = GnomonicFaceSet([black, white])
    faces.blender = AverageBlender()

    result = faces.to_equirectangular((10, 20), preserve_dtype=False)

    assert result.support_mask is not None
    assert result.support_mask.any()
    assert (~result.support_mask).any()
    assert np.allclose(result.data[result.support_mask], 1.0, atol=1e-5)
    assert np.all(result.data[~result.support_mask] == 0)
