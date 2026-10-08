from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.features import MatchProvenance, SphericalFeatureMatches
from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.stereo import (
    DenseMatchFilterOptions,
    SphericalMatchRefinementOptions,
    SphericalStereoOptions,
    SphericalStereoResult,
    filter_matches_by_dense_range,
    refine_matches_on_sphere,
)


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )


def _ray_lattice(shape_hw: tuple[int, int]) -> np.ndarray:
    y, x = np.indices(shape_hw, dtype=np.float64)
    return erp_pixels_to_rays(np.stack((x, y), axis=-1), shape_hw)


def _render_sphere(
    camera_center: np.ndarray,
    rotation_camera_from_a: np.ndarray,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    sphere_center = np.asarray((0.15, -0.10, 0.05))
    radius = 4.4
    rays_camera = _ray_lattice(shape_hw)
    rays_world = rays_camera @ rotation_camera_from_a
    offset = camera_center - sphere_center
    projected = np.sum(rays_world * offset, axis=-1)
    distance = -projected + np.sqrt(
        projected * projected + radius * radius - float(offset @ offset)
    )
    points = camera_center + distance[..., None] * rays_world
    image = np.stack(
        (
            0.5
            + 0.22 * np.sin(5.0 * points[..., 0] + 2.0 * points[..., 2])
            + 0.18 * np.cos(7.0 * points[..., 1]),
            0.5
            + 0.25 * np.sin(4.0 * points[..., 1] - 3.0 * points[..., 2])
            + 0.13 * np.cos(8.0 * points[..., 0]),
            0.5
            + 0.23 * np.cos(6.0 * points[..., 2] + points[..., 0])
            + 0.16 * np.sin(9.0 * points[..., 1]),
        ),
        axis=-1,
    )
    return np.clip(image, 0.0, 1.0).astype(np.float32), distance.astype(np.float32)


def _offset_bearing(
    bearing: np.ndarray, east_deg: float, north_deg: float
) -> np.ndarray:
    east = np.cross(np.asarray((0.0, 1.0, 0.0)), bearing)
    east /= np.linalg.norm(east)
    north = np.cross(bearing, east)
    value = bearing + math.radians(east_deg) * east + math.radians(north_deg) * north
    return value / np.linalg.norm(value)


def _matches(bearings_a: np.ndarray, bearings_b: np.ndarray) -> SphericalFeatureMatches:
    count = len(bearings_a)
    return SphericalFeatureMatches(
        panorama_id_a="a",
        panorama_id_b="b",
        feature_indices_a=np.arange(count),
        feature_indices_b=np.arange(count),
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=np.full(count, 0.2, dtype=np.float32),
        ratio_scores=np.full(count, 0.6, dtype=np.float32),
        mutual=np.ones(count, dtype=bool),
        valid=np.ones(count, dtype=bool),
        matcher_name="analytic",
        matcher_config={},
        backend_name="analytic",
        backend_version="1",
        provenance=MatchProvenance(
            interface="panorai-spherical-features/v1",
            source_checksums=("checksum-a", "checksum-b"),
            face_pairs=tuple(("erp", "erp") for _ in range(count)),
            face_pair_groups=tuple((("erp", "erp"),) for _ in range(count)),
            deduplicated=False,
        ),
        keypoint_responses=np.ones((count, 2), dtype=np.float32),
        face_ids_a=np.full(count, "erp", dtype=object),
        face_ids_b=np.full(count, "erp", dtype=object),
    )


def _dense(range_map: np.ndarray) -> SphericalStereoResult:
    shape = range_map.shape
    return SphericalStereoResult(
        range=range_map,
        validity_mask=np.ones(shape, dtype=bool),
        confidence=np.ones(shape, dtype=np.float32),
        matching_cost=np.zeros(shape, dtype=np.float32),
        hypothesis_index=np.ones(shape, dtype=np.int32),
        inverse_range_hypotheses=np.asarray((0.1, 0.2, 0.3), dtype=np.float32),
        options=SphericalStereoOptions(
            min_range=1.0,
            max_range=10.0,
            num_hypotheses=3,
            bidirectional_consistency=False,
        ),
    )


def _angular_errors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(left * right, axis=1), -1.0, 1.0)))


def _scene_inputs():
    shape = (64, 128)
    rotation = _rotation_y(4.0)
    camera_b_in_a = np.asarray((0.38, 0.02, 0.12))
    translation = -(rotation @ camera_b_in_a)
    reference, ranges = _render_sphere(np.zeros(3), np.eye(3), shape)
    target, _ = _render_sphere(camera_b_in_a, rotation, shape)
    source_pixels = np.asarray(((64.0, 30.0), (125.0, 31.0), (95.0, 21.0)))
    bearings_a = erp_pixels_to_rays(source_pixels, shape)
    source_ranges = ranges[
        source_pixels[:, 1].astype(int), source_pixels[:, 0].astype(int)
    ]
    points = bearings_a * source_ranges[:, None]
    exact_b = points @ rotation.T + translation
    exact_b /= np.linalg.norm(exact_b, axis=1, keepdims=True)
    perturbed_b = np.stack(
        (
            _offset_bearing(exact_b[0], 0.85, -0.30),
            _offset_bearing(exact_b[1], -0.70, 0.25),
            _offset_bearing(exact_b[2], 0.60, 0.40),
        )
    )
    matches = _matches(bearings_a, perturbed_b)
    dense_filter = filter_matches_by_dense_range(
        matches,
        _dense(ranges),
        rotation,
        translation,
        options=DenseMatchFilterOptions(max_angular_error_deg=3.0),
    )
    return reference, target, matches, dense_filter, rotation, translation, exact_b


def test_refinement_reduces_target_bearing_error_and_preserves_inputs() -> None:
    reference, target, matches, dense_filter, rotation, translation, exact_b = (
        _scene_inputs()
    )
    original_a = matches.bearings_a.copy()
    original_b = matches.bearings_b.copy()
    result = refine_matches_on_sphere(
        reference,
        target,
        matches,
        dense_filter,
        rotation,
        translation,
        options=SphericalMatchRefinementOptions(
            patch_radius_px=2,
            search_radius_px=0.6,
            search_step_px=0.05,
            min_cost_improvement=1e-4,
        ),
    )

    before = _angular_errors(matches.bearings_b, exact_b)
    after = _angular_errors(result.refined_bearings_b, exact_b)
    assert result.evaluated_mask.all()
    assert result.applied_mask.sum() >= 2
    assert float(np.median(after)) < float(np.median(before))
    assert np.all(result.cost_improvement[result.applied_mask] > 0.0)
    np.testing.assert_array_equal(matches.bearings_a, original_a)
    np.testing.assert_array_equal(matches.bearings_b, original_b)
    assert not result.refined_bearings_b.flags.writeable
    assert result.provenance.source_checksums == ("checksum-a", "checksum-b")

    correspondences = result.to_bearing_correspondences()
    np.testing.assert_array_equal(correspondences.valid, dense_filter.accepted_mask)
    np.testing.assert_array_equal(correspondences.bearings_b, result.refined_bearings_b)


def test_refinement_is_seam_safe_for_a_target_near_erp_boundary() -> None:
    reference, target, matches, dense_filter, rotation, translation, exact_b = (
        _scene_inputs()
    )
    target_pixels = rays_to_erp_pixels(exact_b, reference.shape[:2]).pixels_xy
    assert target_pixels[1, 0] < 3.0 or target_pixels[1, 0] > 124.0

    result = refine_matches_on_sphere(
        reference,
        target,
        matches,
        dense_filter,
        rotation,
        translation,
        options=SphericalMatchRefinementOptions(
            search_step_px=0.05,
            min_cost_improvement=1e-4,
        ),
    )

    before = _angular_errors(matches.bearings_b[1:2], exact_b[1:2])[0]
    after = _angular_errors(result.refined_bearings_b[1:2], exact_b[1:2])[0]
    assert result.evaluated_mask[1]
    assert after <= before


def test_flat_texture_and_tight_motion_gate_preserve_original_bearings() -> None:
    reference, target, matches, dense_filter, rotation, translation, _ = _scene_inputs()
    flat = np.full_like(reference, 0.5)
    unsupported = refine_matches_on_sphere(
        flat,
        flat,
        matches,
        dense_filter,
        rotation,
        translation,
    )
    assert not unsupported.evaluated_mask.any()
    assert not unsupported.applied_mask.any()
    np.testing.assert_array_equal(unsupported.refined_bearings_b, matches.bearings_b)

    gated = refine_matches_on_sphere(
        reference,
        target,
        matches,
        dense_filter,
        rotation,
        translation,
        options=SphericalMatchRefinementOptions(
            search_step_px=0.05,
            min_cost_improvement=1e-4,
            max_angular_shift_deg=0.05,
        ),
    )
    assert gated.evaluated_mask.all()
    assert not gated.applied_mask.any()
    np.testing.assert_array_equal(gated.refined_bearings_b, matches.bearings_b)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"patch_radius_px": 0}, "patch_radius_px"),
        ({"search_radius_px": 0.0}, "search_radius_px"),
        (
            {"search_radius_px": 0.5, "search_step_px": 0.6},
            "search_step_px",
        ),
        ({"min_patch_support": 1.1}, "min_patch_support"),
        ({"min_cost_improvement": -0.1}, "min_cost_improvement"),
        ({"max_angular_shift_deg": 181.0}, "max_angular_shift_deg"),
    ),
)
def test_refinement_options_reject_invalid_values(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SphericalMatchRefinementOptions(**kwargs)


def test_refinement_rejects_filter_from_different_bearings() -> None:
    reference, target, matches, dense_filter, rotation, translation, _ = _scene_inputs()
    matches.bearings_b[0] = matches.bearings_b[1]

    with pytest.raises(ValueError, match="not produced from these match bearings"):
        refine_matches_on_sphere(
            reference,
            target,
            matches,
            dense_filter,
            rotation,
            translation,
        )
