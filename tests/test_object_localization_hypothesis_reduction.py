from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from panorai.object_localization import (
    HypothesisAffinityConfig,
    HypothesisPairCues,
    HypothesisReductionCandidate,
    MultiviewObjectHypothesis,
    MultiviewSpatialRegion,
    hypothesis_pair_affinity,
    reduce_object_hypotheses,
)


def _spatial(
    center_xyz: tuple[float, float, float],
    *,
    frame: str = "world",
    units: str = "m",
) -> MultiviewSpatialRegion:
    center = np.asarray(center_xyz, dtype=np.float64)
    points = np.stack(
        (
            center + np.asarray((-0.1, 0.0, 0.0)),
            center + np.asarray((0.1, 0.0, 0.0)),
        )
    )
    weights = np.asarray((0.5, 0.5))
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered
    return MultiviewSpatialRegion(
        mode="metric_3d",
        frame=frame,
        units=units,
        support_kind="sparse_points",
        center_xyz=center,
        extent_covariance_xyz=covariance,
        bounding_box_xyz=np.stack((points.min(axis=0), points.max(axis=0))),
        extent_radius=0.1,
        support_points_xyz=points,
        support_weights=weights,
    )


def _hypothesis(
    hypothesis_id: str,
    center_xyz: tuple[float, float, float],
    *,
    score: float = 0.8,
    concept_id: str = "piping",
    vocabulary: str = "industrial/v1",
    track_ids: tuple[str, ...] = ("track-1",),
) -> MultiviewObjectHypothesis:
    return MultiviewObjectHypothesis(
        hypothesis_id=hypothesis_id,
        concept_id=concept_id,
        concept_name=concept_id.title(),
        vocabulary=vocabulary,
        state="proposed",
        view_ids=("a", "b"),
        track_ids=track_ids,
        proxy_ids=("coil",),
        semantic_score=score,
        spatial=_spatial(center_xyz),
    )


def _cues(
    first: str,
    second: str,
    **kwargs: object,
) -> HypothesisPairCues:
    one, two = sorted((first, second))
    return HypothesisPairCues(one, two, **kwargs)


def test_affinity_is_product_of_available_factors_and_omits_missing_cues() -> None:
    first = _hypothesis("a", (0.0, 0.0, 0.0), track_ids=("t1", "t2"))
    second = _hypothesis("b", (0.0, 0.0, 0.0), track_ids=("t2", "t3"))
    cues = _cues(
        "a",
        "b",
        geometric_compatibility=0.8,
        transfer_iou=0.5,
        appearance_cosine_similarity=0.0,
    )

    complete = hypothesis_pair_affinity(first, second, cues=cues)
    partial = hypothesis_pair_affinity(
        first,
        second,
        cues=_cues("a", "b", geometric_compatibility=0.8),
    )

    assert complete.factors == {
        "semantic": 1.0,
        "geometric": 0.8,
        "track": 1.0 / 3.0,
        "transfer": 0.5,
        "appearance": 0.5,
    }
    assert complete.affinity == pytest.approx(0.8 / 12.0)
    assert partial.affinity == pytest.approx(0.8 / 3.0)
    assert "transfer" not in partial.factors
    assert "appearance" not in partial.factors


def test_hard_gates_reject_semantic_and_depth_contradictions() -> None:
    first = _hypothesis("a", (0.0, 0.0, 0.0))
    second = _hypothesis(
        "b",
        (0.0, 0.0, 0.0),
        concept_id="vessel",
        vocabulary="other/v1",
    )

    affinity = hypothesis_pair_affinity(
        first,
        second,
        cues=_cues("a", "b", depth_separated=True),
    )

    assert not affinity.hard_compatible
    assert affinity.affinity == 0.0
    assert affinity.soft_nms_overlap == 0.0
    assert affinity.hard_reasons == (
        "depth-separated-support",
        "incompatible-concept",
        "incompatible-vocabulary",
    )


def test_fusion_uses_semantic_geometric_view_weights_and_keeps_lineage() -> None:
    first = _hypothesis("a", (0.0, 0.0, 0.0), score=0.8)
    second = _hypothesis("b", (2.0, 0.0, 0.0), score=0.4)
    candidates = (
        HypothesisReductionCandidate(first, 1.0, 1.0),
        HypothesisReductionCandidate(second, 0.5, 1.0),
    )

    result = reduce_object_hypotheses(
        candidates,
        pair_cues=(
            _cues(
                "a",
                "b",
                geometric_compatibility=1.0,
                transfer_iou=1.0,
                soft_nms_overlap=1.0,
            ),
        ),
        config=HypothesisAffinityConfig(affinity_threshold=0.5),
    )

    assert len(result.records) == 1
    fused = result.records[0]
    assert fused.action == "fused"
    assert fused.source_hypothesis_ids == ("a", "b")
    assert fused.fused_spatial is not None
    np.testing.assert_allclose(fused.fused_spatial.center_xyz, (0.4, 0.0, 0.0))
    assert fused.reduced_score == pytest.approx(0.72)
    assert result.diagnostics["retained_source_hypothesis_count"] == 2


def test_depth_separated_instances_remain_concurrent() -> None:
    first = _hypothesis("a", (0.0, 0.0, 4.0), score=0.9)
    second = _hypothesis("b", (0.0, 0.0, 8.0), score=0.8)

    result = reduce_object_hypotheses(
        (
            HypothesisReductionCandidate(first),
            HypothesisReductionCandidate(second),
        ),
        pair_cues=(
            _cues(
                "a",
                "b",
                transfer_iou=0.9,
                soft_nms_overlap=0.9,
                depth_separated=True,
            ),
        ),
    )

    assert len(result.records) == 2
    assert all(item.action == "selected" for item in result.records)
    assert {item.reduced_score for item in result.records} == {0.9, 0.8}


def test_soft_nms_decays_but_retains_unfused_alternative() -> None:
    first = _hypothesis("a", (0.0, 0.0, 0.0), score=0.9)
    second = _hypothesis("b", (0.0, 0.0, 0.0), score=0.8)
    sigma = 0.5

    result = reduce_object_hypotheses(
        (
            HypothesisReductionCandidate(first),
            HypothesisReductionCandidate(second),
        ),
        pair_cues=(
            _cues(
                "a",
                "b",
                geometric_compatibility=1.0,
                transfer_iou=1.0,
                soft_nms_overlap=0.6,
            ),
        ),
        config=HypothesisAffinityConfig(
            affinity_threshold=0.5,
            soft_nms_sigma=sigma,
            fuse_finite_components=False,
        ),
    )

    assert len(result.records) == 2
    selected = next(
        item for item in result.records if item.representative.hypothesis_id == "a"
    )
    alternative = next(
        item for item in result.records if item.representative.hypothesis_id == "b"
    )
    assert selected.action == "selected"
    assert selected.reduced_score == pytest.approx(0.9)
    assert alternative.action == "soft_nms"
    assert alternative.reduced_score == pytest.approx(0.8 * math.exp(-(0.6**2) / sigma))
    assert alternative.suppressed_by == ("a",)


def test_affinity_edges_form_transitive_duplicate_component() -> None:
    hypotheses = (
        _hypothesis("a", (0.0, 0.0, 0.0), score=0.9),
        _hypothesis("b", (0.1, 0.0, 0.0), score=0.8),
        _hypothesis("c", (0.2, 0.0, 0.0), score=0.7),
    )
    cues = (
        _cues("a", "b", geometric_compatibility=1.0, transfer_iou=1.0),
        _cues("b", "c", geometric_compatibility=1.0, transfer_iou=1.0),
        _cues("a", "c", geometric_compatibility=0.1, transfer_iou=1.0),
    )

    result = reduce_object_hypotheses(
        tuple(HypothesisReductionCandidate(item) for item in hypotheses),
        pair_cues=cues,
        config=HypothesisAffinityConfig(affinity_threshold=0.5),
    )

    assert len(result.records) == 1
    assert result.records[0].action == "fused"
    assert result.records[0].source_hypothesis_ids == ("a", "b", "c")


def test_reduction_is_deterministic_under_input_reordering() -> None:
    hypotheses = (
        _hypothesis("a", (0.0, 0.0, 0.0), score=0.9),
        _hypothesis("b", (0.0, 0.0, 0.0), score=0.8),
        _hypothesis(
            "c",
            (5.0, 0.0, 0.0),
            score=0.7,
            track_ids=("track-2",),
        ),
    )
    cues = (
        _cues("a", "b", geometric_compatibility=1.0, transfer_iou=1.0),
        _cues("a", "c", geometric_compatibility=0.0, transfer_iou=0.0),
        _cues("b", "c", geometric_compatibility=0.0, transfer_iou=0.0),
    )

    first = reduce_object_hypotheses(
        tuple(HypothesisReductionCandidate(item) for item in hypotheses),
        pair_cues=cues,
    )
    second = reduce_object_hypotheses(
        tuple(HypothesisReductionCandidate(item) for item in reversed(hypotheses)),
        pair_cues=tuple(reversed(cues)),
    )

    def signature(result):
        return tuple(
            (
                item.source_hypothesis_ids,
                item.action,
                item.reduced_score,
                (
                    None
                    if item.fused_spatial is None
                    else tuple(item.fused_spatial.center_xyz)
                ),
            )
            for item in result.records
        )

    assert signature(first) == signature(second)


def test_candidate_metadata_validation_is_strict() -> None:
    hypothesis = _hypothesis("a", (0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="geometric_quality"):
        HypothesisReductionCandidate(hypothesis, geometric_quality=1.1)
    with pytest.raises(ValueError, match="independent_view_factor"):
        HypothesisReductionCandidate(hypothesis, independent_view_factor=0.0)

    changed = replace(hypothesis, hypothesis_id="b")
    with pytest.raises(ValueError, match="strictly ordered"):
        HypothesisPairCues("b", "a")
    assert changed.hypothesis_id == "b"
