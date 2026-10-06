"""High-level descriptor matching with spherical result objects."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._config import FeatureMatcherConfig
from ._models import MatchProvenance, SphericalFeatureMatches, SphericalFeatureSet
from .backends.opencv import OpenCVFeatureBackend


class FeatureMatcher:
    """Match PanorAi feature sets through an encapsulated OpenCV matcher."""

    def __init__(
        self,
        config: FeatureMatcherConfig | None = None,
        *,
        backend: OpenCVFeatureBackend | None = None,
        minimum_opencv_version: str = "4.9.0",
    ) -> None:
        self.config = config or FeatureMatcherConfig()
        self.backend = backend or OpenCVFeatureBackend()
        self.minimum_opencv_version = minimum_opencv_version

    def match(
        self,
        features_a: SphericalFeatureSet,
        features_b: SphericalFeatureSet,
    ) -> SphericalFeatureMatches:
        self.backend.require_version(self.minimum_opencv_version)
        if features_a.descriptor_metric != features_b.descriptor_metric:
            raise ValueError("feature sets must use the same descriptor metric")
        if features_a.descriptor_type != features_b.descriptor_type:
            raise ValueError("feature sets must use the same descriptor type")
        raw, resolved_method = self.backend.match(
            features_a.descriptors,
            features_b.descriptors,
            features_a.descriptor_metric,
            self.config,
        )
        if self.config.deduplicate_matches:
            raw = _deduplicate_spherical_matches(
                raw,
                features_a,
                features_b,
                math.radians(self.config.angular_dedup_threshold_deg),
            )
        else:
            for item in raw:
                item["face_pair_group"] = (
                    (
                        features_a.features[item["query_idx"]].face_id,
                        features_b.features[item["train_idx"]].face_id,
                    ),
                )
        raw.sort(
            key=lambda item: (item["query_idx"], item["train_idx"], item["distance"])
        )

        return _build_spherical_feature_matches(
            raw,
            features_a,
            features_b,
            config=self.config,
            matcher_name=resolved_method,
            backend_name=self.backend.name,
            backend_version=self.backend.version,
        )


def _build_spherical_feature_matches(
    raw: list[dict[str, Any]],
    features_a: SphericalFeatureSet,
    features_b: SphericalFeatureSet,
    *,
    config: FeatureMatcherConfig,
    matcher_name: str,
    backend_name: str,
    backend_version: str,
) -> SphericalFeatureMatches:
    """Materialize PanorAi result arrays from backend match records.

    This package-private seam lets experimental routing policies reuse the
    exact public result construction without implementing another descriptor
    matcher or changing :class:`FeatureMatcher` behaviour.
    """

    indices_a = np.asarray([item["query_idx"] for item in raw], dtype=np.int64)
    indices_b = np.asarray([item["train_idx"] for item in raw], dtype=np.int64)
    bearings_a = (
        features_a.bearings[indices_a]
        if len(raw)
        else np.empty((0, 3), dtype=np.float64)
    )
    bearings_b = (
        features_b.bearings[indices_b]
        if len(raw)
        else np.empty((0, 3), dtype=np.float64)
    )
    distances = np.asarray([item["distance"] for item in raw], dtype=np.float32)
    ratios = None
    if raw and any(item["ratio_score"] is not None for item in raw):
        ratios = np.asarray(
            [
                np.nan if item["ratio_score"] is None else item["ratio_score"]
                for item in raw
            ],
            dtype=np.float32,
        )
    mutual = None
    if raw and any(item["mutual"] is not None for item in raw):
        mutual = np.asarray([bool(item["mutual"]) for item in raw], dtype=bool)
    valid = (
        np.isfinite(bearings_a).all(axis=1)
        & np.isfinite(bearings_b).all(axis=1)
        & np.isfinite(distances)
    )
    responses = (
        np.stack(
            (features_a.responses[indices_a], features_b.responses[indices_b]),
            axis=1,
        )
        if len(raw)
        else np.empty((0, 2), dtype=np.float32)
    )
    face_ids_a = features_a.face_ids[indices_a]
    face_ids_b = features_b.face_ids[indices_b]
    face_pairs = tuple(
        (str(left), str(right)) for left, right in zip(face_ids_a, face_ids_b)
    )
    interface = (
        features_a.interface
        if features_a.interface == features_b.interface
        else "panorai-spherical-features/v1"
    )
    stability = (
        "stable"
        if features_a.stability == features_b.stability == "stable"
        else "experimental"
    )
    provenance = MatchProvenance(
        interface=interface,
        source_checksums=(
            features_a.panorama_checksum,
            features_b.panorama_checksum,
        ),
        face_pairs=face_pairs,
        face_pair_groups=tuple(item["face_pair_group"] for item in raw),
        deduplicated=config.deduplicate_matches,
    )
    return SphericalFeatureMatches(
        panorama_id_a=features_a.panorama_id,
        panorama_id_b=features_b.panorama_id,
        feature_indices_a=indices_a,
        feature_indices_b=indices_b,
        bearings_a=bearings_a,
        bearings_b=bearings_b,
        descriptor_distances=distances,
        ratio_scores=ratios,
        mutual=mutual,
        valid=valid,
        matcher_name=matcher_name,
        matcher_config=config.to_dict(),
        backend_name=backend_name,
        backend_version=backend_version,
        provenance=provenance,
        keypoint_responses=responses,
        face_ids_a=face_ids_a,
        face_ids_b=face_ids_b,
        interface=interface,
        stability=stability,
    )


def match_opencv_features(
    features_a: SphericalFeatureSet,
    features_b: SphericalFeatureSet,
    matcher: Any,
    *,
    knn: int = 2,
) -> SphericalFeatureMatches:
    """Advanced route using an injected OpenCV-compatible matcher."""

    if knn not in {1, 2}:
        raise ValueError("knn must be 1 or 2")
    config = FeatureMatcherConfig(ratio_test=0.75 if knn == 2 else None)
    return FeatureMatcher(
        config,
        backend=OpenCVFeatureBackend(matcher=matcher),
    ).match(features_a, features_b)


def _deduplicate_spherical_matches(
    raw: list[dict[str, Any]],
    features_a: SphericalFeatureSet,
    features_b: SphericalFeatureSet,
    angular_threshold_rad: float,
) -> list[dict[str, Any]]:
    """Apply deterministic bilateral angular NMS to OpenCV-produced matches."""

    if not raw:
        return []
    bearings_a = np.stack(
        [features_a.features[item["query_idx"]].bearing_xyz for item in raw]
    ).astype(np.float64, copy=False)
    bearings_b = np.stack(
        [features_b.features[item["train_idx"]].bearing_xyz for item in raw]
    ).astype(np.float64, copy=False)
    bearings_a /= np.linalg.norm(bearings_a, axis=1, keepdims=True)
    bearings_b /= np.linalg.norm(bearings_b, axis=1, keepdims=True)
    threshold_cosine = math.cos(angular_threshold_rad)
    rank = sorted(
        range(len(raw)),
        key=lambda index: (
            raw[index]["distance"],
            raw[index]["query_idx"],
            raw[index]["train_idx"],
            index,
        ),
    )
    available = np.ones(len(raw), dtype=bool)
    selected: list[dict[str, Any]] = []
    for selected_index in rank:
        if not available[selected_index]:
            continue
        near_a = bearings_a @ bearings_a[selected_index] >= threshold_cosine - 1e-15
        near_b = bearings_b @ bearings_b[selected_index] >= threshold_cosine - 1e-15
        group_indices = np.flatnonzero(available & near_a & near_b)
        available[group_indices] = False
        group_order = sorted(
            (int(index) for index in group_indices),
            key=lambda index: (
                0 if index == selected_index else 1,
                raw[index]["distance"],
                raw[index]["query_idx"],
                raw[index]["train_idx"],
                index,
            ),
        )
        item = dict(raw[selected_index])
        item["face_pair_group"] = tuple(
            (
                features_a.features[raw[index]["query_idx"]].face_id,
                features_b.features[raw[index]["train_idx"]].face_id,
            )
            for index in group_order
        )
        selected.append(item)
    return selected
