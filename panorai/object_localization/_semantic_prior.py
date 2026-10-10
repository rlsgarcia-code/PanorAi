"""Query-conditioned semantic proposal weights for relative-pose RANSAC.

The weights produced here affect only minimal-set proposal sampling.  They do
not remove matches and do not change the estimator's scoring or refinement
sets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from panorai.features import SphericalBearingCorrespondences, SphericalFeatureMatches

from ._models import SemanticQuery, SemanticRegionObservation


SEMANTIC_MATCH_PRIOR_INTERFACE = "panorai-semantic-match-prior/v1"


@dataclass(frozen=True, slots=True)
class SemanticMatchPriorConfig:
    """Conservative mixture used to turn shared CAM support into weights.

    ``uniform_mix`` is a positive floor on every valid match.  A value of one
    disables semantic preference exactly.  ``max_weight_ratio`` independently
    caps concentration after the semantic mixture is formed.
    """

    uniform_mix: float = 0.25
    semantic_power: float = 1.0
    max_weight_ratio: float = 4.0

    def __post_init__(self) -> None:
        for name in ("uniform_mix", "semantic_power", "max_weight_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.uniform_mix <= 1.0:
            raise ValueError("uniform_mix must be in the interval (0, 1]")
        if self.semantic_power <= 0.0:
            raise ValueError("semantic_power must be positive")
        if self.max_weight_ratio < 1.0:
            raise ValueError("max_weight_ratio must be at least 1")


@dataclass(frozen=True, slots=True)
class SemanticMatchPrior:
    """Full-length proposal weights and auditable semantic support.

    Invalid match rows retain zero weight.  Every valid row has positive
    weight, including rows outside all semantic regions, so the global
    geometric fallback remains reachable.
    """

    query_text: str
    query_class_ids: tuple[int, ...]
    used_class_ids: tuple[int, ...]
    sampling_weights: np.ndarray
    semantic_support: np.ndarray
    valid_mask: np.ndarray
    supported_match_indices: np.ndarray
    effective_sample_size: float
    max_to_min_valid_weight_ratio: float
    fallback_reason: str | None
    interface: str = SEMANTIC_MATCH_PRIOR_INTERFACE

    def __post_init__(self) -> None:
        weights = np.array(self.sampling_weights, dtype=np.float64, copy=True)
        support = np.array(self.semantic_support, dtype=np.float64, copy=True)
        valid = np.array(self.valid_mask, dtype=bool, copy=True)
        supported = np.array(self.supported_match_indices, dtype=np.int64, copy=True)
        if (
            weights.ndim != 1
            or support.shape != weights.shape
            or valid.shape != weights.shape
        ):
            raise ValueError(
                "sampling_weights, semantic_support, and valid_mask must align"
            )
        if supported.ndim != 1:
            raise ValueError("supported_match_indices must have shape (N,)")
        if np.any(supported < 0) or np.any(supported >= len(weights)):
            raise ValueError("supported_match_indices are outside the match array")
        if len(np.unique(supported)) != len(supported):
            raise ValueError("supported_match_indices must not contain duplicates")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sampling_weights must be finite and non-negative")
        if not np.all(np.isfinite(support)) or np.any(
            (support < 0.0) | (support > 1.0)
        ):
            raise ValueError("semantic_support must be finite and within [0, 1]")
        if np.any(weights[~valid] != 0.0):
            raise ValueError("invalid match rows must have zero sampling weight")
        if np.any(weights[valid] <= 0.0):
            raise ValueError("every valid match row must have positive sampling weight")
        expected_supported = np.flatnonzero(valid & (support > 0.0))
        if not np.array_equal(supported, expected_supported):
            raise ValueError("supported_match_indices must identify positive support")
        if (
            not math.isfinite(self.effective_sample_size)
            or self.effective_sample_size < 0
        ):
            raise ValueError("effective_sample_size must be finite and non-negative")
        if (
            not math.isfinite(self.max_to_min_valid_weight_ratio)
            or self.max_to_min_valid_weight_ratio < 0
        ):
            raise ValueError(
                "max_to_min_valid_weight_ratio must be finite and non-negative"
            )
        weights.setflags(write=False)
        support.setflags(write=False)
        valid.setflags(write=False)
        supported.setflags(write=False)
        object.__setattr__(self, "sampling_weights", weights)
        object.__setattr__(self, "semantic_support", support)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "supported_match_indices", supported)

    @property
    def weights(self) -> np.ndarray:
        """Alias matching the bearing-correspondence weight vocabulary."""

        return self.sampling_weights

    def to_bearing_correspondences(
        self, matches: SphericalFeatureMatches
    ) -> SphericalBearingCorrespondences:
        """Attach this proposal prior without changing match validity."""

        if len(matches) != len(self.sampling_weights):
            raise ValueError("prior and matches must have the same row count")
        if not np.array_equal(matches.valid, self.valid_mask):
            raise ValueError("prior and matches must have the same validity mask")
        return SphericalBearingCorrespondences(
            bearings_a=np.array(matches.bearings_a, copy=True),
            bearings_b=np.array(matches.bearings_b, copy=True),
            weights=np.array(self.sampling_weights, copy=True),
            valid=np.array(self.valid_mask, copy=True),
        )

    def describe(self) -> dict[str, object]:
        return {
            "interface": self.interface,
            "query_text": self.query_text,
            "query_class_ids": self.query_class_ids,
            "used_class_ids": self.used_class_ids,
            "match_count": len(self.sampling_weights),
            "valid_count": int(self.valid_mask.sum()),
            "supported_count": len(self.supported_match_indices),
            "effective_sample_size": self.effective_sample_size,
            "max_to_min_valid_weight_ratio": self.max_to_min_valid_weight_ratio,
            "fallback_reason": self.fallback_reason,
            "prefilters_correspondences": False,
        }


def _class_feature_evidence(
    regions: tuple[SemanticRegionObservation, ...],
    *,
    class_id: int,
) -> dict[int, float]:
    evidence: dict[int, float] = {}
    for region in regions:
        if region.class_id != class_id or not len(region.feature_indices):
            continue
        values = region.semantic_score * region.membership_weights
        for index, value in zip(region.feature_indices, values, strict=True):
            feature_index = int(index)
            evidence[feature_index] = max(
                evidence.get(feature_index, 0.0), float(value)
            )
    return evidence


def _evidence_at_matches(
    evidence: dict[int, float], feature_indices: np.ndarray
) -> np.ndarray:
    return np.fromiter(
        (evidence.get(int(index), 0.0) for index in feature_indices),
        dtype=np.float64,
        count=len(feature_indices),
    )


def build_semantic_match_prior(
    query: SemanticQuery,
    regions_a: tuple[SemanticRegionObservation, ...],
    regions_b: tuple[SemanticRegionObservation, ...],
    matches: SphericalFeatureMatches,
    config: SemanticMatchPriorConfig | None = None,
) -> SemanticMatchPrior:
    """Build query-conditioned weights from feature-indexed semantic regions.

    Multiple regions or query classes cannot inflate a match by duplication:
    evidence is combined with a maximum both within a class and across query
    classes.  The geometric mean requires support in both views.
    """

    if not isinstance(query, SemanticQuery):
        raise TypeError("query must be a SemanticQuery")
    policy = config or SemanticMatchPriorConfig()
    if any(region.view_id != matches.panorama_id_a for region in regions_a):
        raise ValueError("every regions_a observation must belong to match view A")
    if any(region.view_id != matches.panorama_id_b for region in regions_b):
        raise ValueError("every regions_b observation must belong to match view B")
    if np.any(matches.feature_indices_a < 0) or np.any(matches.feature_indices_b < 0):
        raise ValueError("match feature indices must be non-negative")

    valid = np.asarray(matches.valid, dtype=bool)
    support = np.zeros(len(matches), dtype=np.float64)
    used_classes: list[int] = []
    for class_id in query.class_ids:
        evidence_a = _class_feature_evidence(regions_a, class_id=class_id)
        evidence_b = _class_feature_evidence(regions_b, class_id=class_id)
        if not evidence_a or not evidence_b:
            continue
        class_support = np.sqrt(
            _evidence_at_matches(evidence_a, matches.feature_indices_a)
            * _evidence_at_matches(evidence_b, matches.feature_indices_b)
        )
        if np.any(valid & (class_support > 0.0)):
            used_classes.append(class_id)
            support = np.maximum(support, class_support)
    support[~valid] = 0.0

    weights = np.zeros(len(matches), dtype=np.float64)
    fallback_reason: str | None = None
    if not np.any(valid):
        fallback_reason = "no-valid-matches"
    else:
        maximum = float(support[valid].max(initial=0.0))
        if maximum <= 0.0:
            weights[valid] = 1.0
            fallback_reason = "no-shared-query-support"
        elif policy.uniform_mix == 1.0:
            weights[valid] = 1.0
            fallback_reason = "uniform-only-configured"
        else:
            normalized = np.power(support[valid] / maximum, policy.semantic_power)
            mixed = policy.uniform_mix + (1.0 - policy.uniform_mix) * normalized
            minimum = float(mixed.min())
            mixed = np.minimum(mixed, minimum * policy.max_weight_ratio)
            weights[valid] = mixed / float(mixed.mean())

    valid_weights = weights[valid]
    if len(valid_weights):
        effective_sample_size = float(
            valid_weights.sum() ** 2 / np.square(valid_weights).sum()
        )
        ratio = float(valid_weights.max() / valid_weights.min())
    else:
        effective_sample_size = 0.0
        ratio = 0.0
    return SemanticMatchPrior(
        query_text=query.text,
        query_class_ids=query.class_ids,
        used_class_ids=tuple(used_classes),
        sampling_weights=weights,
        semantic_support=support,
        valid_mask=valid,
        supported_match_indices=np.flatnonzero(valid & (support > 0.0)),
        effective_sample_size=effective_sample_size,
        max_to_min_valid_weight_ratio=ratio,
        fallback_reason=fallback_reason,
    )
