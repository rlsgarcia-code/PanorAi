import numpy as np

from panorai.geometry import (
    CUBE_FACE_BASES,
    CUBE_FACE_ORDER,
    CubemapProjector,
    CubemapSpec,
    GnomonicProjector,
    GnomonicSpec,
    cubemap_to_equirectangular,
    equirectangular_to_cubemap,
    equirectangular_to_gnomonic,
    erp_pixels_to_rays,
    gnomonic_to_equirectangular,
    rays_to_erp_pixels,
)
from panorai.projections.gnomonic_projection import GnomonicProjection
from panorai.projections.gnomonic.config import GnomonicConfig
from panorai.data import EquirectangularImage


def test_cardinal_erp_pixel_coordinates_follow_canonical_frame() -> None:
    shape = (100, 200)
    pixels = np.asarray(
        [
            (99.5, 49.5),   # +Z / front
            (149.5, 49.5),  # +X / right
            (199.5, 49.5),  # -Z / seam
            (49.5, 49.5),   # -X / left
            (99.5, -0.5),   # +Y / north pole
            (99.5, 99.5),   # -Y / south pole
        ]
    )
    expected = np.asarray(
        [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ]
    )

    assert np.allclose(erp_pixels_to_rays(pixels, shape), expected, atol=1e-12)


def test_erp_pixel_ray_round_trip_including_seam_and_poles() -> None:
    shape = (64, 128)
    pixels = np.asarray(
        [
            (-0.49, 31.5),
            (127.49, 31.5),
            (63.5, -0.49),
            (63.5, 63.49),
            (23.25, 17.75),
        ],
        dtype=np.float64,
    )
    projected = rays_to_erp_pixels(erp_pixels_to_rays(pixels, shape), shape)
    circular_error = np.minimum(
        np.abs(projected.pixels_xy[:, 0] - np.mod(pixels[:, 0], shape[1])),
        shape[1] - np.abs(projected.pixels_xy[:, 0] - np.mod(pixels[:, 0], shape[1])),
    )

    assert projected.valid.all()
    assert np.allclose(projected.ranges, 1.0)
    assert np.all(circular_error < 1e-10)
    assert np.allclose(projected.pixels_xy[:, 1], pixels[:, 1], atol=1e-10)


def test_gnomonic_rectangular_fov_roll_and_non_square_shape() -> None:
    height, width = 48, 96
    y, x = np.indices((height, width), dtype=np.float32)
    panorama = np.stack((x / width, y / height, np.ones_like(x)), axis=-1)
    spec = GnomonicSpec(
        center_lat_deg=18.0,
        center_lon_deg=-42.0,
        hfov_deg=104.0,
        vfov_deg=57.0,
        roll_deg=23.0,
        output_shape_hw=(17, 29),
    )

    result = equirectangular_to_gnomonic(panorama, spec)

    assert result.data.shape == (17, 29, 3)
    assert result.support_mask.shape == (17, 29)
    assert result.support_mask.dtype == np.bool_
    assert result.support_mask.all()


def test_backprojection_exposes_support_and_keeps_invalid_depth_nan() -> None:
    face = np.full((15, 25), np.nan, dtype=np.float32)
    spec = GnomonicSpec(hfov_deg=90.0, vfov_deg=60.0, output_shape_hw=face.shape)

    result = gnomonic_to_equirectangular(face, spec, (36, 72), interpolation="nearest")

    assert result.support_mask.dtype == np.bool_
    assert result.support_mask.any()
    assert (~result.support_mask).any()
    assert np.isnan(result.data[~result.support_mask]).all()
    assert np.isnan(result.data[result.support_mask]).any()


def test_cubemap_constants_and_directional_reconstruction() -> None:
    assert CUBE_FACE_ORDER == ("front", "right", "back", "left", "up", "down")
    assert tuple(CUBE_FACE_BASES) == CUBE_FACE_ORDER
    faces = {
        face: np.full((12, 12), index, dtype=np.uint8)
        for index, face in enumerate(CUBE_FACE_ORDER)
    }

    erp = cubemap_to_equirectangular(faces, (24, 48), interpolation="nearest")

    samples = [(12, 24), (12, 36), (12, 0), (12, 12), (0, 24), (23, 24)]
    assert [int(erp.data[y, x]) for y, x in samples] == list(range(6))
    assert erp.support_mask.all()


def test_integer_labels_use_exact_nearest_values_through_cubemap() -> None:
    labels = (np.arange(32 * 64).reshape(32, 64) % 17).astype(np.int16)
    cube = equirectangular_to_cubemap(labels, 24, interpolation="nearest")
    reconstructed = cubemap_to_equirectangular(
        {name: result.data for name, result in cube.items()},
        labels.shape,
        interpolation="nearest",
    )

    assert reconstructed.data.dtype == labels.dtype
    assert set(np.unique(reconstructed.data)).issubset(set(np.unique(labels)))


def test_cubemap_round_trip_preserves_rgb_mask_and_depth_semantics() -> None:
    y, x = np.indices((24, 48))
    rgb = np.stack((x, y, (x + y) % 255), axis=-1).astype(np.uint8)
    mask = ((x + 2 * y) % 5 == 0)
    depth = (1.0 + x / 48.0 + y / 24.0).astype(np.float32)
    depth[10:14, 22:26] = np.nan

    for image in (rgb, mask):
        cube = equirectangular_to_cubemap(image, 24, interpolation="nearest")
        result = cubemap_to_equirectangular(
            {name: face.data for name, face in cube.items()},
            image.shape[:2],
            interpolation="nearest",
        )
        assert result.data.dtype == image.dtype
        assert result.support_mask.all()

    cube = equirectangular_to_cubemap(depth, 24, interpolation="nearest")
    result = cubemap_to_equirectangular(
        {name: face.data for name, face in cube.items()},
        depth.shape,
        interpolation="nearest",
    )
    assert result.data.dtype == depth.dtype
    assert np.isnan(result.data).any()
    assert not np.any(result.data[np.isfinite(result.data)] == 0.0)


def test_legacy_to_gnomonic_honours_requested_resolution() -> None:
    projection = GnomonicProjection(x_points=9, y_points=7)

    face = projection.project(np.zeros((16, 32), dtype=np.float32))

    assert face.shape == (7, 9)


def test_immutable_projectors_delegate_to_functional_api() -> None:
    panorama = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    gnomonic = GnomonicProjector(
        GnomonicSpec(output_shape_hw=(7, 11)), interpolation="nearest"
    )
    face = gnomonic.project(panorama)
    restored = gnomonic.back_project(face.data, panorama.shape)

    assert face.data.shape == (7, 11)
    assert restored.data.shape == panorama.shape
    assert restored.support_mask.dtype == np.bool_

    cubemap = CubemapProjector(CubemapSpec((8, 9)), interpolation="nearest")
    faces = cubemap.project(panorama)
    cube_restored = cubemap.back_project(
        {name: result.data for name, result in faces.items()}, panorama.shape
    )
    assert tuple(faces) == CUBE_FACE_ORDER
    assert all(result.data.shape == (8, 9) for result in faces.values())
    assert cube_restored.data.shape == panorama.shape


def test_integer_data_rejects_bilinear_interpolation() -> None:
    labels = np.zeros((8, 16), dtype=np.uint8)
    with np.testing.assert_raises(TypeError):
        equirectangular_to_gnomonic(labels, GnomonicSpec(output_shape_hw=(4, 4)))


def test_legacy_config_conversion_uses_documented_fallbacks() -> None:
    config = GnomonicConfig(
        phi1_deg=12,
        lam0_deg=-21,
        fov_deg=73,
        x_points=19,
        y_points=11,
    )
    spec = GnomonicSpec.from_config(config)
    assert spec == GnomonicSpec(12, -21, 73, 73, 0, (11, 19))


def test_canonical_spec_is_available_through_legacy_facades() -> None:
    panorama = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    spec = GnomonicSpec(
        center_lat_deg=7,
        center_lon_deg=-13,
        hfov_deg=95,
        vfov_deg=55,
        roll_deg=9,
        output_shape_hw=(9, 15),
    )

    projection = GnomonicProjection(spec=spec)
    expected = equirectangular_to_gnomonic(panorama, spec).data
    np.testing.assert_allclose(projection.project(panorama), expected)

    face = EquirectangularImage(panorama).to_gnomonic(spec=spec)
    np.testing.assert_allclose(face.data, expected)
    assert face.shape == spec.output_shape_hw
    assert face.spec == spec
    assert face.support_mask.all()
