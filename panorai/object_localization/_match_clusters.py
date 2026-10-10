"""Joint spherical match clusters for splitting broad semantic support."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real

import numpy as np

from ._models import (
    PairObjectLocalizationInput,
    RegionAssociation,
    SemanticRegionObservation,
)


JOINT_MATCH_CLUSTERS_INTERFACE = "panorai-joint-match-clusters/v1"


@dataclass(frozen=True, slots=True)
class JointMatchClusterConfig:
    """Conservative gates for grouping matches in both spherical views.

    Two matches are neighbours only when their bearing separation is within
    ``max_neighbor_angle_deg`` in *both* views. Connected components smaller
    than ``min_matches`` are reported as discarded support.
    """

    max_neighbor_angle_deg: float = 5.0
    min_matches: int = 3
    max_clusters: int | None = None
    pose_inliers_only: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_neighbor_angle_deg, bool)
            or not isinstance(self.max_neighbor_angle_deg, Real)
            or not math.isfinite(float(self.max_neighbor_angle_deg))
            or not 0.0 < float(self.max_neighbor_angle_deg) <= 180.0
        ):
            raise ValueError("max_neighbor_angle_deg must be finite and in (0, 180]")
        if (
            isinstance(self.min_matches, bool)
            or not isinstance(self.min_matches, Integral)
            or self.min_matches < 1
        ):
            raise ValueError("min_matches must be a positive integer")
        if self.max_clusters is not None and (
            isinstance(self.max_clusters, bool)
            or not isinstance(self.max_clusters, Integral)
            or self.max_clusters < 1
        ):
            raise ValueError("max_clusters must be a positive integer or None")
        if not isinstance(self.pose_inliers_only, bool):
            raise TypeError("pose_inliers_only must be boolean")


@dataclass(frozen=True, slots=True)
class JointMatchCluster:
    """One paired candidate instance supported by matched features."""

    cluster_id: str
    parent_association_id: str
    match_indices: np.ndarray
    region_a: SemanticRegionObservation
    region_b: SemanticRegionObservation
    diameter_a_deg: float
    diameter_b_deg: float

    def __post_init__(self) -> None:
        if not self.cluster_id or not self.parent_association_id:
            raise ValueError("cluster and parent association IDs must be non-empty")
        indices = np.array(self.match_indices, dtype=np.int64, copy=True)
        if indices.ndim != 1 or np.any(indices < 0):
            raise ValueError("match_indices must be a non-negative (N,) array")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("match_indices must not contain duplicates")
        indices.sort()
        indices.setflags(write=False)
        for name in ("diameter_a_deg", "diameter_b_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 180.0:
                raise ValueError(f"{name} must be finite and within [0, 180]")
        object.__setattr__(self, "match_indices", indices)


@dataclass(frozen=True, slots=True)
class JointMatchClusterResult:
    """Retained paired regions and explicit clustering diagnostics."""

    parent_association_id: str
    clusters: tuple[JointMatchCluster, ...]
    source_match_count: int
    raw_cluster_count: int
    retained_cluster_count: int
    discarded_small_cluster_count: int
    discarded_match_count: int
    truncated_cluster_count: int
    max_neighbor_angle_deg: float
    pose_inliers_only: bool
    interface: str = JOINT_MATCH_CLUSTERS_INTERFACE

    def __post_init__(self) -> None:
        if not self.parent_association_id:
            raise ValueError("parent_association_id must be non-empty")
        if self.retained_cluster_count != len(self.clusters):
            raise ValueError("retained_cluster_count must match clusters")
        counts = (
            self.source_match_count,
            self.raw_cluster_count,
            self.retained_cluster_count,
            self.discarded_small_cluster_count,
            self.discarded_match_count,
            self.truncated_cluster_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in counts
        ):
            raise ValueError("cluster diagnostics must be non-negative integers")

    @property
    def regions_a(self) -> tuple[SemanticRegionObservation, ...]:
        return tuple(cluster.region_a for cluster in self.clusters)

    @property
    def regions_b(self) -> tuple[SemanticRegionObservation, ...]:
        return tuple(cluster.region_b for cluster in self.clusters)

    def describe(self) -> dict[str, object]:
        return {
            "interface": self.interface,
            "parent_association_id": self.parent_association_id,
            "source_match_count": self.source_match_count,
            "raw_cluster_count": self.raw_cluster_count,
            "retained_cluster_count": self.retained_cluster_count,
            "discarded_small_cluster_count": self.discarded_small_cluster_count,
            "discarded_match_count": self.discarded_match_count,
            "truncated_cluster_count": self.truncated_cluster_count,
            "max_neighbor_angle_deg": self.max_neighbor_angle_deg,
            "pose_inliers_only": self.pose_inliers_only,
        }


def _angular_distance_matrix_deg(bearings: np.ndarray) -> np.ndarray:
    dots = np.clip(bearings @ bearings.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _connected_components(adjacency: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(len(adjacency), dtype=bool)
    components: list[np.ndarray] = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = np.flatnonzero(adjacency[current] & ~visited)
            visited[neighbours] = True
            stack.extend(int(value) for value in neighbours[::-1])
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _parent_region(
    regions: tuple[SemanticRegionObservation, ...], region_id: str
) -> SemanticRegionObservation:
    for region in regions:
        if region.region_id == region_id:
            return region
    raise ValueError(f"association references unknown region {region_id!r}")


def _membership_for_features(
    parent: SemanticRegionObservation, feature_indices: np.ndarray
) -> np.ndarray:
    weights_by_index = {
        int(index): float(weight)
        for index, weight in zip(
            parent.feature_indices, parent.membership_weights, strict=True
        )
    }
    try:
        return np.asarray(
            [weights_by_index[int(index)] for index in feature_indices],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError(
            "clustered match endpoint is outside its parent semantic region"
        ) from error


def _centroid(bearings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    value = np.sum(bearings * weights[:, None], axis=0)
    if np.linalg.norm(value) <= 1e-12:
        value = np.sum(bearings, axis=0)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("cluster bearing centroid is undefined")
    return value / norm


def _stable_cluster_id(
    association: RegionAssociation,
    feature_pairs: tuple[tuple[int, int], ...],
) -> str:
    payload = json.dumps(
        [
            JOINT_MATCH_CLUSTERS_INTERFACE,
            association.association_id,
            feature_pairs,
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    return f"match_cluster_{hashlib.sha256(payload).hexdigest()[:32]}"


def _diameter_deg(bearings: np.ndarray) -> float:
    if len(bearings) < 2:
        return 0.0
    return float(np.max(_angular_distance_matrix_deg(bearings)))


def cluster_region_association_matches(
    association: RegionAssociation,
    evidence: PairObjectLocalizationInput,
    config: JointMatchClusterConfig | None = None,
) -> JointMatchClusterResult:
    """Split one accepted broad region pair using match proximity in both views.

    This function does not assign a persistent identity. It returns paired
    :class:`SemanticRegionObservation` candidates that can be passed back to
    :class:`ObjectLocalizationPipeline`. Angular distances are computed from
    3D bearings, so neighbourhoods cross the ERP longitude seam correctly.
    """

    if association.state != "accepted":
        raise ValueError("only accepted region associations can be clustered")
    policy = config or JointMatchClusterConfig()
    parent_a = _parent_region(evidence.regions_a, association.region_id_a)
    parent_b = _parent_region(evidence.regions_b, association.region_id_b)
    source_indices = np.asarray(
        (
            association.inlier_match_indices
            if policy.pose_inliers_only
            else association.match_indices
        ),
        dtype=np.int64,
    )
    if len(source_indices) == 0:
        return JointMatchClusterResult(
            parent_association_id=association.association_id,
            clusters=(),
            source_match_count=0,
            raw_cluster_count=0,
            retained_cluster_count=0,
            discarded_small_cluster_count=0,
            discarded_match_count=0,
            truncated_cluster_count=0,
            max_neighbor_angle_deg=float(policy.max_neighbor_angle_deg),
            pose_inliers_only=policy.pose_inliers_only,
        )

    bearings_a = np.asarray(evidence.matches.bearings_a[source_indices])
    bearings_b = np.asarray(evidence.matches.bearings_b[source_indices])
    radius = float(policy.max_neighbor_angle_deg)
    adjacency = (_angular_distance_matrix_deg(bearings_a) <= radius) & (
        _angular_distance_matrix_deg(bearings_b) <= radius
    )
    components = _connected_components(adjacency)
    feature_indices_a = np.asarray(evidence.matches.feature_indices_a)
    feature_indices_b = np.asarray(evidence.matches.feature_indices_b)

    component_records = []
    for local_indices in components:
        match_indices = source_indices[local_indices]
        feature_pairs = tuple(
            sorted(
                (
                    int(feature_indices_a[index]),
                    int(feature_indices_b[index]),
                )
                for index in match_indices
            )
        )
        component_records.append((local_indices, match_indices, feature_pairs))
    component_records.sort(key=lambda item: (-len(item[1]), item[2]))

    small = [item for item in component_records if len(item[1]) < policy.min_matches]
    retained = [
        item for item in component_records if len(item[1]) >= policy.min_matches
    ]
    truncated = []
    if policy.max_clusters is not None:
        truncated = retained[policy.max_clusters :]
        retained = retained[: policy.max_clusters]

    clusters: list[JointMatchCluster] = []
    for local_indices, match_indices, feature_pairs in retained:
        cluster_id = _stable_cluster_id(association, feature_pairs)
        indices_a = np.unique(feature_indices_a[match_indices]).astype(np.int64)
        indices_b = np.unique(feature_indices_b[match_indices]).astype(np.int64)
        weights_a = _membership_for_features(parent_a, indices_a)
        weights_b = _membership_for_features(parent_b, indices_b)
        cluster_bearings_a = np.asarray(evidence.features_a.bearings[indices_a])
        cluster_bearings_b = np.asarray(evidence.features_b.bearings[indices_b])
        suffix = cluster_id.removeprefix("match_cluster_")
        region_a = SemanticRegionObservation(
            region_id=f"{parent_a.region_id}#joint-{suffix}",
            view_id=evidence.view_id_a,
            class_id=association.class_id,
            class_name=association.class_name,
            semantic_score=parent_a.semantic_score,
            feature_indices=indices_a,
            membership_weights=weights_a,
            centroid_bearing=_centroid(cluster_bearings_a, weights_a),
            source_id=(
                f"{parent_a.source_id}#joint-match-cluster"
                if parent_a.source_id is not None
                else JOINT_MATCH_CLUSTERS_INTERFACE
            ),
        )
        region_b = SemanticRegionObservation(
            region_id=f"{parent_b.region_id}#joint-{suffix}",
            view_id=evidence.view_id_b,
            class_id=association.class_id,
            class_name=association.class_name,
            semantic_score=parent_b.semantic_score,
            feature_indices=indices_b,
            membership_weights=weights_b,
            centroid_bearing=_centroid(cluster_bearings_b, weights_b),
            source_id=(
                f"{parent_b.source_id}#joint-match-cluster"
                if parent_b.source_id is not None
                else JOINT_MATCH_CLUSTERS_INTERFACE
            ),
        )
        clusters.append(
            JointMatchCluster(
                cluster_id=cluster_id,
                parent_association_id=association.association_id,
                match_indices=match_indices,
                region_a=region_a,
                region_b=region_b,
                diameter_a_deg=_diameter_deg(bearings_a[local_indices]),
                diameter_b_deg=_diameter_deg(bearings_b[local_indices]),
            )
        )

    discarded_match_count = sum(len(item[1]) for item in (*small, *truncated))
    return JointMatchClusterResult(
        parent_association_id=association.association_id,
        clusters=tuple(clusters),
        source_match_count=len(source_indices),
        raw_cluster_count=len(component_records),
        retained_cluster_count=len(clusters),
        discarded_small_cluster_count=len(small),
        discarded_match_count=discarded_match_count,
        truncated_cluster_count=len(truncated),
        max_neighbor_angle_deg=radius,
        pose_inliers_only=policy.pose_inliers_only,
    )
