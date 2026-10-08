from __future__ import annotations

import numpy as np
import pytest

import panorai.features._spherical_detector as detector_module
from panorai.features import (
    SphericalCoarseDoGDetector,
    SphericalCoarseDoGDetectorConfig,
    SphericalDoGDetector,
    SphericalDoGDetectorConfig,
    SphericalRefinedKeypoint,
)
from panorai.features._spherical_detector import (
    _fine_refined_levels,
    _quadratic_derivatives,
    _solve_quadratic_offset,
)
from panorai.geometry import erp_pixels_to_rays
from panorai.image_processing._sampling import sample_rays, sample_rays_multi


def _config(**changes: object) -> SphericalDoGDetectorConfig:
    values: dict[str, object] = {
        "octaves": 2,
        "max_keypoints": 80,
        "contrast_threshold": 0.003,
        "minimum_valid_support_fraction": 0.95,
        "selection_grid_shape": (6, 12),
        "convolution_backend": "numpy",
    }
    values.update(changes)
    return SphericalDoGDetectorConfig(**values)  # type: ignore[arg-type]


def _textured_panorama(shape: tuple[int, int] = (64, 128)) -> np.ndarray:
    yy, xx = np.indices(shape)
    values = np.sin(xx * 0.41) + np.sin(yy * 0.57) + np.sin((xx + yy) * 0.19)
    return np.clip(values * 35.0 + 128.0, 0.0, 255.0).astype(np.uint8)


def _quadratic_cube(
    offset: tuple[float, float, float],
    hessian: np.ndarray,
    peak: float,
) -> np.ndarray:
    extremum = np.asarray(offset, dtype=np.float64)
    cube = np.empty((3, 3, 3), dtype=np.float64)
    for scale_index, scale in enumerate((-1.0, 0.0, 1.0)):
        for north_index, north in enumerate((-1.0, 0.0, 1.0)):
            for east_index, east in enumerate((-1.0, 0.0, 1.0)):
                point = np.asarray((east, north, scale), dtype=np.float64)
                displacement = point - extremum
                cube[scale_index, north_index, east_index] = (
                    peak + 0.5 * displacement @ hessian @ displacement
                )
    return cube


def test_detector_returns_descriptor_free_public_keypoints() -> None:
    result = SphericalDoGDetector(_config()).detect(
        _textured_panorama(), panorama_id="synthetic"
    )

    assert 20 <= len(result) <= 80
    assert result.interface == "panorai-spherical-dog-detector/v2"
    assert not hasattr(result, "descriptors")
    assert all(isinstance(item, SphericalRefinedKeypoint) for item in result.keypoints)
    assert result.bearings.shape == (len(result), 3)
    assert result.source_erp_xy.shape == (len(result), 2)
    assert np.allclose(np.linalg.norm(result.bearings, axis=1), 1.0)
    assert np.all(result.scales_deg > 0.0)
    assert np.all(result.valid_support_fractions >= 0.95)
    assert np.all(np.abs(result.refined_levels - result.levels) <= 0.5)
    edge_limit = (_config().edge_threshold + 1.0) ** 2 / _config().edge_threshold
    for item in result.keypoints:
        assert item.response == pytest.approx(abs(item.interpolated_dog_response))
        assert item.edge_score < edge_limit
        assert (
            1.0
            <= item.hessian_condition
            <= _config().refinement_maximum_hessian_condition
        )
        assert 1 <= item.localization_iterations <= _config().refinement_max_iterations
        assert np.isfinite(item.tangent_offset_rad).all()
    diagnostics = result.diagnostics
    assert diagnostics.raw_extrema_count >= diagnostics.refinement_converged_count
    assert (
        diagnostics.refinement_converged_count
        >= diagnostics.interpolated_contrast_count
        >= diagnostics.refined_edge_count
        >= diagnostics.valid_support_count
    )
    assert diagnostics.valid_support_count >= diagnostics.unique_after_deduplication
    assert diagnostics.unique_after_deduplication >= diagnostics.output_after_budget
    assert diagnostics.output_after_budget == len(result)
    assert (
        sum(item.valid_support_count for item in diagnostics.octaves)
        == diagnostics.valid_support_count
    )


def test_batched_detector_is_exactly_equal_to_independent_detection() -> None:
    images = (_textured_panorama(), np.roll(_textured_panorama(), 7, axis=1))
    independent = tuple(
        SphericalDoGDetector(_config()).detect(image, panorama_id=f"view-{index}")
        for index, image in enumerate(images)
    )
    batched = SphericalDoGDetector(_config()).detect_batch(
        images,
        panorama_ids=("view-0", "view-1"),
    )

    for left, right in zip(independent, batched, strict=True):
        for field in (
            "source_erp_xy",
            "bearings",
            "responses",
            "scales_deg",
            "octaves",
            "levels",
            "valid_support_fractions",
            "refined_levels",
        ):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        assert left.diagnostics.to_dict() == right.diagnostics.to_dict()


def test_multi_image_ray_sampling_is_exactly_equal_to_independent_calls() -> None:
    rng = np.random.default_rng(31)
    images = tuple(rng.normal(size=(32, 64)).astype(np.float32) for _ in range(3))
    rays = rng.normal(size=(7, 9, 3))
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    expected = tuple(sample_rays(image, rays) for image in images)
    actual = sample_rays_multi(images, rays)

    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


def test_second_order_taylor_oracle_recovers_continuous_extremum() -> None:
    expected_offset = np.asarray((0.25, -0.30, 0.20), dtype=np.float64)
    expected_hessian = np.asarray(
        (
            (-4.0, 0.20, 0.10),
            (0.20, -2.5, -0.15),
            (0.10, -0.15, -3.0),
        ),
        dtype=np.float64,
    )
    peak = 2.75
    cube = _quadratic_cube(tuple(expected_offset), expected_hessian, peak)

    gradient, hessian = _quadratic_derivatives(cube)
    offset, condition = _solve_quadratic_offset(gradient, hessian, 1.0e6)

    assert np.allclose(hessian, expected_hessian, atol=1e-12)
    assert offset is not None
    assert np.allclose(offset, expected_offset, atol=1e-12)
    assert condition >= 1.0
    interpolated = cube[1, 1, 1] + 0.5 * gradient @ offset
    assert interpolated == pytest.approx(peak, abs=1e-12)


def test_second_order_taylor_rejects_singular_hessian() -> None:
    gradient = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    hessian = np.diag((2.0, 1.0, 0.0))

    offset, condition = _solve_quadratic_offset(gradient, hessian, 1.0e6)

    assert offset is None
    assert np.isinf(condition)


def test_refinement_relocates_then_accepts_interpolated_extremum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hessian = np.diag((-4.0, -3.0, -2.0))
    cubes = iter(
        (
            _quadratic_cube((0.80, 0.0, 0.0), hessian, 0.08),
            _quadratic_cube((0.20, -0.10, 0.15), hessian, 0.08),
        )
    )
    monkeypatch.setattr(
        detector_module, "_sample_dog_cube", lambda *args, **kwargs: next(cubes)
    )
    shape = (32, 64)
    result = detector_module._refine_dog_extremum(
        dogs=[np.zeros(shape)] * 5,
        supports=[np.ones(shape)] * 6,
        validity=np.ones(shape, dtype=bool),
        initial_xy=(31, 15),
        initial_level=2,
        octave=0,
        original_shape=shape,
        step_rad=np.pi / shape[0],
        config=_config(contrast_threshold=0.01),
    )

    assert result.rejection_reason is None
    assert result.candidate is not None
    assert result.candidate.localization_iterations == 2
    assert result.candidate.refined_level == pytest.approx(2.15)
    assert result.candidate.interpolated_dog_response == pytest.approx(0.08)


@pytest.mark.parametrize(
    ("cube", "expected_reason"),
    (
        (
            _quadratic_cube((0.1, 0.1, 0.1), np.diag((-4.0, -3.0, -2.0)), 0.005),
            "low-interpolated-contrast",
        ),
        (
            _quadratic_cube((0.1, 0.1, 0.1), np.diag((-20.0, -1.0, -2.0)), 0.08),
            "edge-response",
        ),
    ),
)
def test_refinement_rejects_contrast_and_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
    cube: np.ndarray,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(
        detector_module, "_sample_dog_cube", lambda *args, **kwargs: cube
    )
    shape = (32, 64)
    result = detector_module._refine_dog_extremum(
        dogs=[np.zeros(shape)] * 5,
        supports=[np.ones(shape)] * 6,
        validity=np.ones(shape, dtype=bool),
        initial_xy=(31, 15),
        initial_level=2,
        octave=0,
        original_shape=shape,
        step_rad=np.pi / shape[0],
        config=_config(contrast_threshold=0.01),
    )

    assert result.candidate is None
    assert result.rejection_reason == expected_reason


def test_tangent_offsets_remain_finite_near_pole_and_seam() -> None:
    bearings = (
        np.asarray((1.0e-12, 1.0, -1.0e-12)),
        np.asarray((1.0e-12, 0.0, -1.0)),
    )
    for bearing in bearings:
        rays = detector_module._tangent_offset_rays(
            bearing,
            np.asarray((-0.01, 0.0, 0.01)),
            np.asarray((0.005, 0.0, -0.005)),
        )
        assert np.isfinite(rays).all()
        assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("refinement_max_iterations", 0),
        ("refinement_maximum_offset", 0.49),
        ("refinement_maximum_hessian_condition", 0.99),
    ),
)
def test_refinement_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        SphericalDoGDetectorConfig(**{field: value})


def test_detector_is_deterministic_and_longitude_roll_equivariant() -> None:
    image = _textured_panorama()
    detector = SphericalDoGDetector(_config(max_keypoints=100))
    first = detector.detect(image)
    repeated = detector.detect(image)
    shifted = detector.detect(np.roll(image, 9, axis=1))

    assert np.array_equal(first.source_erp_xy, repeated.source_erp_xy)
    shifted_x = (first.source_erp_xy[:, 0] + 9.0) % image.shape[1]
    expected_pixels = np.stack((shifted_x, first.source_erp_xy[:, 1]), axis=1)
    expected = erp_pixels_to_rays(expected_pixels, image.shape)
    similarity = expected @ shifted.bearings.T
    nearest_deg = np.degrees(np.arccos(np.clip(similarity.max(axis=1), -1.0, 1.0)))
    assert np.mean(nearest_deg < 1.5) >= 0.8


def test_scale_aware_validity_rejects_shadow_boundary() -> None:
    image = _textured_panorama()
    validity = np.ones(image.shape, dtype=bool)
    validity[48:, :] = False
    detector = SphericalDoGDetector(
        _config(
            max_keypoints=100,
            minimum_valid_support_fraction=0.995,
        )
    )
    result = detector.detect(image, validity_mask=validity)

    assert len(result) > 0
    pixels = np.rint(result.source_erp_xy).astype(np.int64)
    assert validity[pixels[:, 1], pixels[:, 0]].all()
    assert np.all(result.valid_support_fractions >= 0.995)


def test_equal_area_selection_uses_more_cells_than_response_only() -> None:
    image = _textured_panorama((96, 192))
    balanced = SphericalDoGDetector(
        _config(max_keypoints=40, selection_grid_shape=(6, 12))
    ).detect(image)
    response = SphericalDoGDetector(
        _config(max_keypoints=40, selection_policy="response")
    ).detect(image)

    def occupied(result) -> int:
        bearings = result.bearings
        rows = np.clip(((bearings[:, 1] + 1.0) * 3.0).astype(int), 0, 5)
        lon = np.arctan2(bearings[:, 0], bearings[:, 2])
        columns = np.clip(((lon + np.pi) / (2.0 * np.pi) * 12).astype(int), 0, 11)
        return len(set(zip(rows.tolist(), columns.tolist())))

    assert occupied(balanced) >= occupied(response)


def test_coarse_detector_promotes_geometry_and_provenance_to_source_erp() -> None:
    image = _textured_panorama((128, 256))
    detector = SphericalCoarseDoGDetector(
        SphericalCoarseDoGDetectorConfig(
            proposal_height=64,
            octaves=2,
            max_keypoints=80,
            contrast_threshold=0.003,
            convolution_backend="numpy",
        )
    )

    result = detector.detect(image, panorama_id="coarse-synthetic")

    assert 20 <= len(result) <= 80
    assert result.interface == "panorai-spherical-coarse-dog-detector/v1"
    assert result.source_shape_hw == image.shape
    assert result.config.to_dict()["proposal_height"] == 64
    assert result.config.to_dict()["fine_verification"] is None
    assert np.all((result.source_erp_xy[:, 0] >= -0.5))
    assert np.all((result.source_erp_xy[:, 0] < image.shape[1] - 0.5))
    assert np.all((result.source_erp_xy[:, 1] >= -0.5))
    assert np.all((result.source_erp_xy[:, 1] < image.shape[0] - 0.5))
    expected = erp_pixels_to_rays(result.source_erp_xy, image.shape)
    assert np.allclose(expected, result.bearings, atol=1e-12)


def test_coarse_detector_can_vectorize_native_resolution_fine_verification() -> None:
    image = _textured_panorama((256, 512))
    config = SphericalCoarseDoGDetectorConfig(
        proposal_height=64,
        octaves=2,
        max_keypoints=80,
        contrast_threshold=0.003,
        convolution_backend="numpy",
        fine_verification="tangent-dog",
        fine_candidate_multiplier=2.0,
        fine_patch_size=15,
    )

    result = SphericalCoarseDoGDetector(config).detect(image)

    assert len(result) == 80
    assert result.config.to_dict()["fine_verification"] == "tangent-dog"
    assert result.diagnostics.unique_after_deduplication >= len(result)
    assert np.all(result.responses > 0.0)
    assert np.all(result.valid_support_fractions >= 0.97)
    assert np.all(result.refined_levels >= 0.0)
    expected = erp_pixels_to_rays(result.source_erp_xy, image.shape)
    assert np.allclose(expected, result.bearings, atol=1e-12)


def test_fine_verification_rejects_scale_below_public_level_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _textured_panorama((256, 512))
    config = SphericalCoarseDoGDetectorConfig(
        proposal_height=64,
        octaves=2,
        max_keypoints=80,
        contrast_threshold=0.003,
        convolution_backend="numpy",
        fine_verification="tangent-dog",
        fine_candidate_multiplier=2.0,
        fine_patch_size=15,
    )
    original_select = detector_module._select_candidates

    def inject_boundary_candidate(candidates, candidate_config):
        selected, count = original_select(candidates, candidate_config)
        if not selected:
            return selected, count
        first = selected[0]
        selected = [detector_module.replace(first, refined_level=0.25), *selected[1:]]
        return selected, count

    # The proposal detector returns a valid boundary-level keypoint. Fine
    # verification is then free to select its lower scale, but must not try to
    # construct a public keypoint with a negative refined level.
    calls = 0

    def select_with_boundary(candidates, candidate_config):
        nonlocal calls
        calls += 1
        if calls == 1:
            return inject_boundary_candidate(candidates, candidate_config)
        return original_select(candidates, candidate_config)

    monkeypatch.setattr(detector_module, "_select_candidates", select_with_boundary)

    result = SphericalCoarseDoGDetector(config).detect(image)

    assert calls >= 2
    assert np.all(result.refined_levels >= 0.0)


def test_fine_level_boundary_oracle_marks_only_negative_levels_unrepresentable() -> (
    None
):
    levels, representable = _fine_refined_levels(
        np.asarray((0.25, 0.25, 0.25)),
        np.asarray((0, 1, 2)),
    )

    np.testing.assert_allclose(levels, (-0.75, 0.25, 1.25))
    np.testing.assert_array_equal(representable, (False, True, True))


def test_coarse_area_resampling_suppresses_nyquist_checkerboard() -> None:
    yy, xx = np.indices((128, 256))
    checkerboard = (((xx + yy) % 2) * 255).astype(np.uint8)
    from panorai.features._spherical_detector import _solid_angle_area_downsample

    reduced, valid = _solid_angle_area_downsample(
        checkerboard, np.ones(checkerboard.shape, dtype=bool), (64, 128)
    )

    assert valid.all()
    assert float(np.std(reduced.astype(np.float64))) <= 0.5
    assert 127.0 <= float(np.mean(reduced)) <= 128.0


def test_coarse_gaussian_resampling_suppresses_checkerboard_and_normalizes_support() -> (
    None
):
    yy, xx = np.indices((128, 256))
    checkerboard = (((xx + yy) % 2) * 255).astype(np.uint8)
    validity = np.ones(checkerboard.shape, dtype=bool)
    validity[48:80, 96:160] = False
    from panorai.features._spherical_detector import (
        _gaussian_antialiased_downsample,
    )

    reduced, valid = _gaussian_antialiased_downsample(
        checkerboard,
        validity,
        (32, 64),
        sigma_px=2.0,
    )

    assert not valid.all()
    assert np.isfinite(reduced).all()
    assert float(np.std(reduced[valid].astype(np.float64))) <= 1.0
    assert 126.0 <= float(np.mean(reduced[valid])) <= 129.0


def test_coarse_spherical_gaussian_resampling_is_roll_equivariant() -> None:
    image = _textured_panorama((128, 256)).astype(np.float32)
    validity = np.ones(image.shape, dtype=bool)
    from panorai.features._spherical_detector import (
        _spherical_gaussian_antialiased_downsample,
    )

    reduced, valid = _spherical_gaussian_antialiased_downsample(
        image,
        validity,
        (32, 64),
        sigma_px=2.0,
        intermediate_height=64,
        backend="numpy",
    )
    shifted, shifted_valid = _spherical_gaussian_antialiased_downsample(
        np.roll(image, 8, axis=1),
        validity,
        (32, 64),
        sigma_px=2.0,
        intermediate_height=64,
        backend="numpy",
    )

    assert valid.all() and shifted_valid.all()
    assert np.allclose(np.roll(reduced, 2, axis=1), shifted, atol=1e-5)
