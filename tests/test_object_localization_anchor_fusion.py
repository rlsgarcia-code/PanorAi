from __future__ import annotations

import math

import numpy as np
import pytest

from panorai.geometry import GnomonicSpec
from panorai.object_localization import (
    HypothesisGnomonicRegion,
    RegisteredRadialDepth,
    SphericalAnchorFusionConfig,
    SphericalAnchorPairCues,
    SphericalFOVAnchor,
    fuse_registered_spherical_anchors,
    generate_spherical_fov_anchors,
    spherical_anchor_pair_affinity,
)
from panorai.reconstruction import SphericalCameraPose


SHAPE_HW = (96, 192)


def _bearing(longitude_deg: float, latitude_deg: float) -> np.ndarray:
    longitude = math.radians(longitude_deg)
    latitude = math.radians(latitude_deg)
    return np.asarray(
        (
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
            math.cos(latitude) * math.cos(longitude),
        ),
        dtype=np.float64,
    )


def _region(
    view_id: str,
    *,
    longitude_deg: float = 0.0,
    latitude_deg: float = 0.0,
    hfov_deg: float = 24.0,
    vfov_deg: float = 18.0,
    track_id: str | None = "track-object",
) -> HypothesisGnomonicRegion:
    center = _bearing(longitude_deg, latitude_deg)
    track_features = () if track_id is None else ((track_id, 0),)
    return HypothesisGnomonicRegion(
        view_id=view_id,
        spec=GnomonicSpec(
            center_lat_deg=latitude_deg,
            center_lon_deg=longitude_deg,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            output_shape_hw=(128, 128),
        ),
        center_bearing_xyz=center,
        angular_radius_deg=0.0,
        support_bearings_xyz=center[None, :],
        track_feature_indices=track_features,
        sparse_support_coverage=1.0,
        semantic_evidence_available=True,
        semantic_supporting=track_id is not None,
        extent_source="analytic-gnomonic-fixture",
    )


def _anchor(
    anchor_id: str,
    view_id: str,
    *,
    longitude_deg: float = 0.0,
    latitude_deg: float = 0.0,
    hfov_deg: float = 24.0,
    vfov_deg: float = 18.0,
    score: float = 0.9,
    track_id: str | None = "track-object",
    proposal_family_ids: tuple[str, ...] = (),
) -> SphericalFOVAnchor:
    return SphericalFOVAnchor(
        anchor_id=anchor_id,
        view_id=view_id,
        concept_id="bookcase",
        vocabulary="imagenet-1k",
        region=_region(
            view_id,
            longitude_deg=longitude_deg,
            latitude_deg=latitude_deg,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            track_id=track_id,
        ),
        score=score,
        source_hypothesis_ids=(f"hypothesis-{anchor_id}",),
        proposal_family_ids=proposal_family_ids,
    )


def _poses(*view_ids: str, same_center: bool = True) -> tuple[SphericalCameraPose, ...]:
    result = []
    for index, view_id in enumerate(view_ids):
        center = np.zeros(3) if same_center else np.asarray((float(index), 0.0, 0.0))
        result.append(SphericalCameraPose(view_id, np.eye(3), center))
    return tuple(result)


def _constant_depth(*view_ids: str) -> tuple[RegisteredRadialDepth, ...]:
    return tuple(
        RegisteredRadialDepth(
            view_id,
            np.ones(SHAPE_HW, dtype=np.float64),
            np.ones(SHAPE_HW, dtype=bool),
            "m",
            f"analytic-unit-sphere:{view_id}",
            provenance={
                "oracle": "unit radial sphere around coincident camera centres",
                "convention": "panorai-geometry-v1 radial range",
            },
        )
        for view_id in view_ids
    )


def _pair(first: str, second: str, **kwargs) -> SphericalAnchorPairCues:
    one, two = sorted((first, second))
    return SphericalAnchorPairCues(one, two, **kwargs)


def _angular_error(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def test_pair_affinity_distinguishes_missing_observed_zero_and_alternative_cues():
    first = _anchor("a", "view-a")
    second = _anchor("b", "view-b")

    missing = spherical_anchor_pair_affinity(first, second)
    observed_zero = spherical_anchor_pair_affinity(
        first, second, cues=_pair("a", "b", transfer_iou=0.0)
    )
    alternative = spherical_anchor_pair_affinity(
        first,
        second,
        cues=_pair("a", "b", transfer_iou=0.8, track_consistency=0.0),
    )

    assert missing.affinity == 0.0
    assert missing.available_association_cues == ()
    assert observed_zero.affinity == 0.0
    assert observed_zero.available_association_cues == ("transfer",)
    assert observed_zero.factors["transfer"] == 0.0
    assert alternative.affinity == pytest.approx(0.8)
    assert alternative.factors["track"] == 0.0


def test_pair_affinity_hard_gates_same_view_and_semantic_mismatch():
    first = _anchor("a", "view-a")
    second = _anchor("b", "view-a")
    second = SphericalFOVAnchor(
        second.anchor_id,
        second.view_id,
        "desk",
        second.vocabulary,
        second.region,
        second.score,
    )

    affinity = spherical_anchor_pair_affinity(
        first, second, cues=_pair("a", "b", transfer_iou=1.0)
    )

    assert not affinity.hard_compatible
    assert affinity.affinity == 0.0
    assert affinity.hard_reasons == (
        "incompatible-concept",
        "same-view-alternatives",
    )


def test_complete_link_rejects_transitive_bridge_and_is_order_invariant():
    anchors = (
        _anchor("a", "view-a", score=0.9, track_id=None),
        _anchor("b", "view-b", score=0.8, track_id=None),
        _anchor("c", "view-c", score=0.7, track_id=None),
    )
    cues = (
        _pair("a", "b", transfer_iou=0.9),
        _pair("b", "c", transfer_iou=0.9),
        _pair("a", "c", transfer_iou=0.1),
    )
    config = SphericalAnchorFusionConfig(
        pair_affinity_threshold=0.5,
        complete_link_threshold=0.5,
    )

    first = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a", "view-b", "view-c"),
        pair_cues=cues,
        config=config,
    )
    second = fuse_registered_spherical_anchors(
        tuple(reversed(anchors)),
        poses=tuple(reversed(_poses("view-a", "view-b", "view-c"))),
        pair_cues=tuple(reversed(cues)),
        config=config,
    )

    def signature(result):
        return tuple((item.anchor_ids, item.state) for item in result.clusters)

    assert signature(first) == signature(second)
    assert {item.anchor_ids for item in first.clusters} == {("a", "b"), ("c",)}
    assert not any(item.anchor_ids == ("a", "b", "c") for item in first.clusters)


def test_intra_view_nms_keeps_best_anchor_and_never_places_two_views_twice():
    anchors = (
        _anchor("a-best", "view-a", score=0.9),
        _anchor("a-duplicate", "view-a", score=0.7),
        _anchor("b", "view-b", score=0.8),
    )
    result = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a", "view-b"),
        pair_cues=(
            _pair("a-best", "b", transfer_iou=0.9),
            _pair("a-duplicate", "b", transfer_iou=0.9),
        ),
        config=SphericalAnchorFusionConfig(complete_link_threshold=0.5),
    )

    assert result.suppressed_anchor_ids == ("a-duplicate",)
    assert tuple(item.anchor_id for item in result.retained_anchors) == ("a-best", "b")
    assert result.clusters[0].view_ids == ("view-a", "view-b")
    decisions = {item.anchor.anchor_id: item for item in result.anchor_decisions}
    assert decisions["a-best"].action == "retained"
    assert decisions["a-duplicate"].action == "suppressed-overlap"
    assert decisions["a-duplicate"].representative_anchor_id == "a-best"
    assert decisions["a-duplicate"].solid_angle_iou == pytest.approx(1.0)


def test_anchor_decisions_explain_capacity_suppression_without_fake_overlap():
    anchors = tuple(
        _anchor(
            f"anchor-{index}",
            "view-a",
            longitude_deg=longitude,
            score=0.9 - 0.1 * index,
        )
        for index, longitude in enumerate((-120.0, 0.0, 120.0))
    )

    result = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a"),
        config=SphericalAnchorFusionConfig(maximum_anchors_per_view_concept=2),
    )

    decisions = {item.anchor.anchor_id: item for item in result.anchor_decisions}
    assert decisions["anchor-2"].action == "suppressed-capacity"
    assert decisions["anchor-2"].representative_anchor_id is None
    assert decisions["anchor-2"].solid_angle_iou is None
    assert len(result.anchor_decisions) == len(anchors)


def test_anchor_generator_exposes_scale_and_aspect_alternatives_with_base_preferred():
    region = _region("view-a", hfov_deg=20.0, vfov_deg=10.0)
    anchors = generate_spherical_fov_anchors(
        region,
        concept_id="bookcase",
        vocabulary="imagenet-1k",
        score=0.9,
        anchor_prefix="fixture",
    )

    assert len(anchors) == 9
    base = next(
        item
        for item in anchors
        if item.provenance["scale_factor"] == 1.0
        and item.provenance["aspect_ratio"] == 1.0
    )
    assert base.region.spec.hfov_deg == pytest.approx(20.0)
    assert base.region.spec.vfov_deg == pytest.approx(10.0)
    assert base.score == max(item.score for item in anchors)


def test_anchor_generator_deduplicates_variants_collapsed_by_fov_limits():
    anchors = generate_spherical_fov_anchors(
        _region("view-a", hfov_deg=150.0, vfov_deg=150.0),
        concept_id="bookcase",
        vocabulary="imagenet-1k",
        score=0.9,
        anchor_prefix="clipped-fixture",
    )

    assert len({item.anchor_id for item in anchors}) == len(anchors)
    assert any(
        len(item.provenance["collapsed_generator_variants"]) > 1 for item in anchors
    )


@pytest.mark.parametrize(
    ("longitude_deg", "latitude_deg"),
    ((179.0, 0.0), (20.0, 78.0)),
)
def test_registered_consensus_is_seam_and_pole_safe_and_retains_compact_fov(
    longitude_deg: float,
    latitude_deg: float,
):
    anchors = (
        _anchor(
            "a",
            "view-a",
            longitude_deg=longitude_deg,
            latitude_deg=latitude_deg,
            hfov_deg=24.0,
            vfov_deg=18.0,
        ),
        _anchor(
            "b",
            "view-b",
            longitude_deg=longitude_deg,
            latitude_deg=latitude_deg,
            hfov_deg=24.0,
            vfov_deg=18.0,
        ),
    )
    result = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a", "view-b"),
        radial_depth=_constant_depth("view-a", "view-b"),
        config=SphericalAnchorFusionConfig(
            evaluation_shape_hw=SHAPE_HW,
            complete_link_threshold=0.5,
            pair_affinity_threshold=0.5,
        ),
    )

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.state == "confirmed"
    expected = _bearing(longitude_deg, latitude_deg)
    for region in cluster.consensus_regions:
        assert _angular_error(region.center_bearing_xyz, expected) < 2.5
        assert 8.0 <= region.spec.hfov_deg < 40.0
        assert 8.0 <= region.spec.vfov_deg < 40.0
        assert region.sparse_support_coverage >= 0.90
        assert region.extent_source == "multiview-spherical-anchor-consensus"
    assert result.diagnostics["transient_masks_retained"] == 0


def test_consistent_compact_anchor_cluster_outscores_large_alternative():
    anchors = (
        _anchor("a-small", "view-a", hfov_deg=20.0, vfov_deg=16.0, score=0.9),
        _anchor("a-large", "view-a", hfov_deg=70.0, vfov_deg=70.0, score=0.7),
        _anchor("b-small", "view-b", hfov_deg=20.0, vfov_deg=16.0, score=0.9),
        _anchor("b-large", "view-b", hfov_deg=70.0, vfov_deg=70.0, score=0.7),
    )
    cues = (
        _pair("a-small", "b-small", transfer_iou=0.9),
        _pair("a-large", "b-large", transfer_iou=0.9),
        _pair("a-small", "b-large", transfer_iou=0.05),
        _pair("a-large", "b-small", transfer_iou=0.05),
    )
    result = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a", "view-b"),
        pair_cues=cues,
        config=SphericalAnchorFusionConfig(
            intra_view_nms_iou=0.95,
            pair_affinity_threshold=0.5,
            complete_link_threshold=0.5,
        ),
    )

    confirmed = [item for item in result.clusters if item.state == "confirmed"]
    assert len(confirmed) == 2
    assert confirmed[0].anchor_ids == ("a-small", "b-small")
    assert confirmed[0].score > confirmed[1].score


def test_cluster_nms_keeps_best_fov_family_and_records_suppressed_alternatives():
    anchors = tuple(
        SphericalFOVAnchor(
            item.anchor_id,
            item.view_id,
            item.concept_id,
            item.vocabulary,
            item.region,
            item.score,
            source_hypothesis_ids=("shared-object-hypothesis",),
            proposal_family_ids=(f"family-{item.view_id}",),
        )
        for item in (
            _anchor("a-small", "view-a", hfov_deg=20.0, vfov_deg=16.0, score=0.9),
            _anchor("a-large", "view-a", hfov_deg=50.0, vfov_deg=45.0, score=0.7),
            _anchor("b-small", "view-b", hfov_deg=20.0, vfov_deg=16.0, score=0.9),
            _anchor("b-large", "view-b", hfov_deg=50.0, vfov_deg=45.0, score=0.7),
        )
    )
    result = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a", "view-b"),
        pair_cues=(
            _pair("a-small", "b-small", transfer_iou=0.9),
            _pair("a-large", "b-large", transfer_iou=0.9),
            _pair("a-small", "b-large", transfer_iou=0.05),
            _pair("a-large", "b-small", transfer_iou=0.05),
        ),
        config=SphericalAnchorFusionConfig(
            intra_view_nms_iou=0.95,
            complete_link_threshold=0.5,
            cluster_nms_iou=0.1,
        ),
    )

    confirmed = [item for item in result.clusters if item.state == "confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0].anchor_ids == ("a-small", "b-small")
    assert len(result.suppressed_cluster_ids) == 1
    assert result.diagnostics["cross_view_cluster_suppressed_count"] == 1
    decisions = {item.cluster.cluster_id: item for item in result.cluster_decisions}
    suppressed = [item for item in decisions.values() if item.action != "retained"]
    assert len(suppressed) == 1
    assert suppressed[0].action == "suppressed-family-overlap"
    assert suppressed[0].representative_cluster_id == confirmed[0].cluster_id
    assert len(result.cluster_decisions) == result.diagnostics["raw_cluster_count"]


def test_cluster_nms_does_not_merge_distinct_families_from_one_hypothesis():
    anchors = tuple(
        SphericalFOVAnchor(
            item.anchor_id,
            item.view_id,
            item.concept_id,
            item.vocabulary,
            item.region,
            item.score,
            source_hypothesis_ids=("shared-upstream-hypothesis",),
            proposal_family_ids=(family,),
        )
        for item, family in (
            (
                _anchor("a-first", "view-a", longitude_deg=-1.0, score=0.9),
                "family-a-first",
            ),
            (
                _anchor("b-first", "view-b", longitude_deg=-1.0, score=0.9),
                "family-b-first",
            ),
            (
                _anchor("a-second", "view-a", longitude_deg=1.0, score=0.8),
                "family-a-second",
            ),
            (
                _anchor("b-second", "view-b", longitude_deg=1.0, score=0.8),
                "family-b-second",
            ),
        )
    )
    result = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a", "view-b"),
        pair_cues=(
            _pair("a-first", "b-first", transfer_iou=0.9),
            _pair("a-second", "b-second", transfer_iou=0.9),
            _pair("a-first", "b-second", transfer_iou=0.05),
            _pair("a-second", "b-first", transfer_iou=0.05),
        ),
        config=SphericalAnchorFusionConfig(
            intra_view_nms_iou=1.0,
            complete_link_threshold=0.5,
        ),
    )

    confirmed = [item for item in result.clusters if item.state == "confirmed"]
    assert len(confirmed) == 2
    assert result.suppressed_cluster_ids == ()


def test_cluster_nms_requires_geometric_overlap_even_for_same_family():
    anchors = (
        _anchor(
            "a-west",
            "view-a",
            longitude_deg=-55.0,
            proposal_family_ids=("family-a",),
        ),
        _anchor(
            "b-west",
            "view-b",
            longitude_deg=-55.0,
            proposal_family_ids=("family-b",),
        ),
        _anchor(
            "a-east",
            "view-a",
            longitude_deg=55.0,
            score=0.8,
            proposal_family_ids=("family-a",),
        ),
        _anchor(
            "b-east",
            "view-b",
            longitude_deg=55.0,
            score=0.8,
            proposal_family_ids=("family-b",),
        ),
    )
    result = fuse_registered_spherical_anchors(
        anchors,
        poses=_poses("view-a", "view-b"),
        pair_cues=(
            _pair("a-west", "b-west", transfer_iou=0.9),
            _pair("a-east", "b-east", transfer_iou=0.9),
            _pair("a-west", "b-east", transfer_iou=0.0),
            _pair("a-east", "b-west", transfer_iou=0.0),
        ),
        config=SphericalAnchorFusionConfig(complete_link_threshold=0.5),
    )

    confirmed = [item for item in result.clusters if item.state == "confirmed"]
    assert len(confirmed) == 2
    assert result.suppressed_cluster_ids == ()
