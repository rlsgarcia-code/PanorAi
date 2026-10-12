from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.geometry import GnomonicSpec
from panorai.object_localization import (
    HypothesisGnomonicRegion,
    RegisteredRadialDepth,
    RegisteredSemanticEvidence,
    SemanticConcept,
    evaluate_spherical_region,
    spherical_region_mask,
    spherical_solid_angle_weights,
    symmetric_spherical_region_transfer,
    transfer_spherical_region,
)
from panorai.reconstruction import SphericalCameraPose


SHAPE_HW = (64, 128)


def _bearing(latitude_deg: float, longitude_deg: float) -> np.ndarray:
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    return np.asarray(
        (
            math.sin(longitude) * math.cos(latitude),
            math.sin(latitude),
            math.cos(longitude) * math.cos(latitude),
        )
    )


def _region(
    view_id: str,
    *,
    latitude_deg: float = 0.0,
    longitude_deg: float = 0.0,
    hfov_deg: float = 70.0,
    vfov_deg: float = 70.0,
) -> HypothesisGnomonicRegion:
    center = _bearing(latitude_deg, longitude_deg)
    return HypothesisGnomonicRegion(
        view_id=view_id,
        spec=GnomonicSpec(
            center_lat_deg=latitude_deg,
            center_lon_deg=longitude_deg,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            output_shape_hw=(64, 64),
        ),
        center_bearing_xyz=center,
        angular_radius_deg=0.0,
        support_bearings_xyz=center[None, :],
        track_feature_indices=(),
        sparse_support_coverage=1.0,
        semantic_evidence_available=True,
        semantic_supporting=True,
    )


def _depth(view_id: str, support: np.ndarray | None = None) -> RegisteredRadialDepth:
    valid = (
        np.ones(SHAPE_HW, dtype=bool)
        if support is None
        else np.asarray(support, dtype=bool)
    )
    values = np.full(SHAPE_HW, np.nan, dtype=np.float64)
    values[valid] = 5.0
    return RegisteredRadialDepth(
        view_id=view_id,
        values=values,
        support=valid,
        units="m",
        source_id=f"analytic-depth:{view_id}",
    )


def test_gnomonic_region_mask_wraps_across_erp_seam() -> None:
    mask = spherical_region_mask(
        _region("view", longitude_deg=179.0, hfov_deg=40.0).spec,
        SHAPE_HW,
    )

    equator = SHAPE_HW[0] // 2
    assert mask[equator, 0]
    assert mask[equator, -1]
    assert not mask[equator, SHAPE_HW[1] // 2]
    assert not mask.flags.writeable


def test_polar_region_membership_stays_bounded_and_area_weights_shrink() -> None:
    mask = spherical_region_mask(
        _region(
            "view",
            latitude_deg=84.0,
            longitude_deg=35.0,
            hfov_deg=50.0,
            vfov_deg=24.0,
        ).spec,
        SHAPE_HW,
    )
    weights = spherical_solid_angle_weights(SHAPE_HW)

    assert mask.any()
    assert np.any(mask[: SHAPE_HW[0] // 4])
    assert not np.any(mask[-SHAPE_HW[0] // 4 :])
    assert weights[0, 0] < weights[SHAPE_HW[0] // 2, 0]
    assert weights[0, 0] == pytest.approx(weights[-1, 0])


def test_regional_metrics_use_supported_solid_angle_and_preserve_zero_evidence() -> (
    None
):
    region = _region("view", hfov_deg=60.0, vfov_deg=60.0)
    region_mask = spherical_region_mask(region.spec, SHAPE_HW)
    scores = np.zeros((1, *SHAPE_HW), dtype=np.float32)
    scores[0, region_mask] = 1.0
    support = np.ones(SHAPE_HW, dtype=bool)
    evidence = RegisteredSemanticEvidence(
        "view",
        ("target",),
        scores,
        support,
        "fixture/v1",
        "fixture:cam",
    )
    concept = SemanticConcept("target", "Target", (("target", 1.0),), "fixture/v1")

    metrics = evaluate_spherical_region(evidence, concept, region)

    weights = spherical_solid_angle_weights(SHAPE_HW)
    expected_fraction = float(weights[region_mask].sum() / weights.sum())
    assert metrics.semantic_mass_recall == pytest.approx(1.0)
    assert metrics.observed_sphere_fraction == pytest.approx(expected_fraction)
    assert metrics.mean_score_enrichment == pytest.approx(1.0 / expected_fraction)

    zero_evidence = RegisteredSemanticEvidence(
        "view",
        ("target",),
        np.zeros_like(scores),
        support,
        "fixture/v1",
        "fixture:zero",
    )
    zero_metrics = evaluate_spherical_region(zero_evidence, concept, region)
    assert zero_metrics.semantic_mass_recall is None
    assert zero_metrics.mean_score_enrichment is None
    assert zero_metrics.observed_sphere_fraction == pytest.approx(expected_fraction)


def test_regional_metrics_do_not_count_unsupported_pixels() -> None:
    region = _region("view", longitude_deg=179.0, hfov_deg=50.0)
    region_mask = spherical_region_mask(region.spec, SHAPE_HW)
    support = np.ones(SHAPE_HW, dtype=bool)
    support[:, : SHAPE_HW[1] // 2] = False
    scores = np.zeros((1, *SHAPE_HW), dtype=np.float32)
    scores[0, region_mask & support] = 0.75
    evidence = RegisteredSemanticEvidence(
        "view",
        ("target",),
        scores,
        support,
        "fixture/v1",
        "fixture:partial",
    )
    concept = SemanticConcept("target", "Target", (("target", 1.0),), "fixture/v1")

    metrics = evaluate_spherical_region(evidence, concept, region)

    assert metrics.semantic_mass_recall == pytest.approx(1.0)
    assert metrics.supported_region_pixel_count == int((region_mask & support).sum())
    assert metrics.supported_region_pixel_count < metrics.region_pixel_count


def test_identity_pose_transfer_is_exact_under_nearest_rasterization() -> None:
    region_a = _region("a", longitude_deg=-175.0, hfov_deg=50.0)
    region_b = _region("b", longitude_deg=-175.0, hfov_deg=50.0)
    pose_a = SphericalCameraPose("a", np.eye(3), np.zeros(3))
    pose_b = SphericalCameraPose("b", np.eye(3), np.zeros(3))

    transfer = symmetric_spherical_region_transfer(
        region_a,
        region_b,
        _depth("a"),
        _depth("b"),
        pose_a,
        pose_b,
    )

    assert transfer.forward.solid_angle_iou == pytest.approx(1.0)
    assert transfer.reverse.solid_angle_iou == pytest.approx(1.0)
    assert transfer.symmetric_solid_angle_iou == pytest.approx(1.0)
    np.testing.assert_array_equal(
        transfer.forward.transferred_mask,
        transfer.forward.target_region_mask,
    )


def test_directional_transfer_respects_explicit_target_support() -> None:
    region_a = _region("a")
    region_b = _region("b")
    pose_a = SphericalCameraPose("a", np.eye(3), np.zeros(3))
    pose_b = SphericalCameraPose("b", np.eye(3), np.zeros(3))
    target_support = np.ones(SHAPE_HW, dtype=bool)
    target_support[:, : SHAPE_HW[1] // 2] = False

    transfer = transfer_spherical_region(
        region_a,
        region_b,
        _depth("a"),
        pose_a,
        pose_b,
        target_support=target_support,
    )

    assert transfer.solid_angle_iou == pytest.approx(1.0)
    assert not np.any(transfer.transferred_mask[~target_support])
    assert not np.any(transfer.target_region_mask[~target_support])


def test_directional_transfer_uses_registered_translation_and_rotation() -> None:
    source_row, source_column = 31, 63
    source_longitude = (source_column + 0.5) / SHAPE_HW[1] * 2.0 * math.pi - math.pi
    source_latitude = math.pi / 2.0 - (source_row + 0.5) / SHAPE_HW[0] * math.pi
    source_ray = _bearing(math.degrees(source_latitude), math.degrees(source_longitude))
    point = 5.0 * source_ray
    yaw = math.radians(23.0)
    target_rotation = np.asarray(
        (
            (math.cos(yaw), 0.0, math.sin(yaw)),
            (0.0, 1.0, 0.0),
            (-math.sin(yaw), 0.0, math.cos(yaw)),
        )
    )
    target_center = np.asarray((1.0, 0.0, 0.0))
    target_vector = target_rotation @ (point - target_center)
    target_ray = target_vector / np.linalg.norm(target_vector)
    target_longitude = math.atan2(float(target_ray[0]), float(target_ray[2]))
    target_latitude = math.asin(float(target_ray[1]))
    expected_column = (
        int(
            math.floor(
                ((target_longitude + math.pi) / (2.0 * math.pi) * SHAPE_HW[1] - 0.5)
                + 0.5
            )
        )
        % SHAPE_HW[1]
    )
    expected_row = int(
        math.floor(
            ((math.pi / 2.0 - target_latitude) / math.pi * SHAPE_HW[0] - 0.5) + 0.5
        )
    )
    source_support = np.zeros(SHAPE_HW, dtype=bool)
    source_support[source_row, source_column] = True
    source_values = np.full(SHAPE_HW, np.nan)
    source_values[source_row, source_column] = 5.0
    source_depth = RegisteredRadialDepth(
        "a", source_values, source_support, "m", "single-ray-depth"
    )

    transfer = transfer_spherical_region(
        _region(
            "a",
            latitude_deg=math.degrees(source_latitude),
            longitude_deg=math.degrees(source_longitude),
            hfov_deg=20.0,
            vfov_deg=20.0,
        ),
        _region(
            "b",
            latitude_deg=math.degrees(target_latitude),
            longitude_deg=math.degrees(target_longitude),
            hfov_deg=20.0,
            vfov_deg=20.0,
        ),
        source_depth,
        SphericalCameraPose("a", np.eye(3), np.zeros(3)),
        SphericalCameraPose("b", target_rotation, target_center),
        target_support=np.ones(SHAPE_HW, dtype=bool),
    )

    assert transfer.source_supported_depth_count == 1
    assert transfer.transferred_target_pixel_count == 1
    assert transfer.transferred_mask[expected_row, expected_column]


def test_radial_depth_rejects_supported_invalid_values() -> None:
    values = np.ones(SHAPE_HW, dtype=np.float64)
    support = np.ones(SHAPE_HW, dtype=bool)
    values[3, 4] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        RegisteredRadialDepth(
            "view",
            values,
            support,
            "m",
            "invalid-fixture",
        )


def test_symmetric_transfer_requires_matching_depth_units() -> None:
    first = _depth("a")
    second = RegisteredRadialDepth(
        "b",
        np.ones(SHAPE_HW),
        np.ones(SHAPE_HW, dtype=bool),
        "arbitrary",
        "depth:b",
    )
    pose_a = SphericalCameraPose("a", np.eye(3), np.zeros(3))
    pose_b = SphericalCameraPose("b", np.eye(3), np.zeros(3))

    with pytest.raises(ValueError, match="units"):
        symmetric_spherical_region_transfer(
            _region("a"), _region("b"), first, second, pose_a, pose_b
        )
