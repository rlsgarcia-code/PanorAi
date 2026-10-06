"""Experimental multiscale visual-context routing for spherical features.

The module deliberately keeps three responsibilities separate:

* an embedding provider describes appearance but never labels a region;
* OpenCV remains responsible for local detection, description and matching;
* PanorAi owns virtual-camera scale, spherical association, routing,
  provenance and proposal weights for the existing robust estimator.

The global descriptor match is always retained.  Context nodes can add local
matches and alter RANSAC proposal mass, but they never remove a
correspondence from geometric scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import math
from numbers import Integral, Real
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from panorai.geometry import (
    GnomonicProjector,
    GnomonicSpec,
    equirectangular_to_gnomonic,
    gnomonic_pixels_to_rays,
)

from ._config import FaceSetSpec
from ._extractor import FeatureExtractor, _as_panorama, _is_torch, _spatial_shape
from ._matcher import (
    _build_spherical_feature_matches,
    _deduplicate_spherical_matches,
)
from ._models import SphericalBearingCorrespondences, SphericalFeatureMatches
from ._pipeline import SphericalFeaturePipeline


@runtime_checkable
class VisualEmbeddingProvider(Protocol):
    """Caller-injectable, batch visual embedding contract.

    Providers return one finite floating vector per image.  They may use a
    learned model, but model selection, weights and licensing remain explicit
    caller concerns and are recorded through ``name`` and ``version``.
    """

    name: str
    version: str

    def embed(
        self,
        images: Sequence[np.ndarray],
        *,
        masks: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class MultiscaleEmbeddingConfig:
    """Serializable policy for adaptive wide/local spherical views."""

    local_fov_deg: tuple[float, float] = (42.0, 42.0)
    local_grid_size: int = 2
    max_local_views_per_root: int = 2
    context_similarity_threshold: float = 0.82
    scale_similarity_threshold: float = 0.94
    node_match_similarity_threshold: float = 0.55
    node_match_top_k: int = 2
    cross_scale_nms_threshold_deg: float = 0.20
    fallback_weight: float = 0.35
    regional_weight_gain: float = 1.0
    embedding_shape_hw: tuple[int, int] = (64, 64)
    interface: str = "panorai-multiscale-visual-features/v1"

    def __post_init__(self) -> None:
        if self.interface != "panorai-multiscale-visual-features/v1":
            raise ValueError(
                "interface must be 'panorai-multiscale-visual-features/v1'"
            )
        object.__setattr__(
            self,
            "local_fov_deg",
            _pair(self.local_fov_deg, "local_fov_deg", lower=0.0, upper=180.0),
        )
        object.__setattr__(
            self,
            "embedding_shape_hw",
            _int_pair(self.embedding_shape_hw, "embedding_shape_hw"),
        )
        for name in (
            "local_grid_size",
            "max_local_views_per_root",
            "node_match_top_k",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "context_similarity_threshold",
            "scale_similarity_threshold",
            "node_match_similarity_threshold",
        ):
            value = _finite(getattr(self, name), name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
            object.__setattr__(self, name, value)
        threshold = _finite(
            self.cross_scale_nms_threshold_deg,
            "cross_scale_nms_threshold_deg",
        )
        if not 0.0 <= threshold <= 180.0:
            raise ValueError("cross_scale_nms_threshold_deg must be in [0, 180]")
        object.__setattr__(self, "cross_scale_nms_threshold_deg", threshold)
        for name in ("fallback_weight", "regional_weight_gain"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenCVContextEmbedding:
    """Small deterministic visual-context descriptor with no model weights.

    This is a reference provider for testing the routing mechanics, not a
    claim of learned semantic understanding.  It combines coarse luminance,
    colour, gradient-orientation and low-frequency DCT evidence.  Production
    experiments may inject a learned provider through the same contract.
    """

    name = "opencv-context"
    version = "1"

    def __init__(self, *, shape_hw: tuple[int, int] = (64, 64)) -> None:
        self.shape_hw = _int_pair(shape_hw, "shape_hw")

    def embed(
        self,
        images: Sequence[np.ndarray],
        *,
        masks: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        import cv2

        if masks is not None and len(masks) != len(images):
            raise ValueError("masks and images must have the same length")
        rows: list[np.ndarray] = []
        for index, image in enumerate(images):
            array = _to_hwc_uint8(image)
            mask = (
                np.ones(array.shape[:2], dtype=bool)
                if masks is None
                else np.asarray(masks[index], dtype=bool)
            )
            if mask.shape != array.shape[:2]:
                raise ValueError("embedding mask must match image spatial shape")
            array = _fill_invalid_nearest(array, mask)
            height, width = self.shape_hw
            resized = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
            resized_mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if not resized_mask.any():
                rows.append(np.zeros(112, dtype=np.float32))
                continue
            rgb = (
                np.repeat(resized[..., None], 3, axis=2)
                if resized.ndim == 2
                else resized[..., :3]
            )
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            # Coarse spatial luminance retains layout while remaining compact.
            pooled = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA).ravel()
            dct = cv2.dct(gray)[:8, :8].ravel()
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hist_parts = []
            for channel, limit in ((0, 180), (1, 256), (2, 256)):
                hist = cv2.calcHist(
                    [hsv], [channel], resized_mask.astype(np.uint8), [8], [0, limit]
                ).ravel()
                hist_parts.append(hist / max(float(hist.sum()), 1.0))
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
            gradient_mask = cv2.erode(
                resized_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
            ).astype(bool)
            orientation = np.zeros(8, dtype=np.float32)
            bins = np.floor((angle % (2.0 * np.pi)) * (8.0 / (2.0 * np.pi))).astype(int)
            for bin_index in range(8):
                active = gradient_mask & (bins == bin_index)
                orientation[bin_index] = float(magnitude[active].sum())
            orientation /= max(float(orientation.sum()), 1e-12)
            row = np.concatenate((pooled, dct, *hist_parts, orientation)).astype(
                np.float32
            )
            norm = float(np.linalg.norm(row))
            rows.append(row / norm if norm > 0.0 else row)
        return np.stack(rows) if rows else np.empty((0, 112), dtype=np.float32)


@dataclass(slots=True)
class VisualContextNode:
    node_id: str
    parent_id: str | None
    level: int
    spec: GnomonicSpec
    embedding: np.ndarray
    selected: bool
    global_local_similarity: float | None = None
    same_support_similarity: float | None = None
    novelty: float = 0.0
    solid_angle_sr: float = 0.0
    feature_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    feature_density_per_sr: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "level": self.level,
            "selected": self.selected,
            "spec": {
                "center_lat_deg": self.spec.center_lat_deg,
                "center_lon_deg": self.spec.center_lon_deg,
                "hfov_deg": self.spec.hfov_deg,
                "vfov_deg": self.spec.vfov_deg,
                "roll_deg": self.spec.roll_deg,
                "output_shape_hw": self.spec.output_shape_hw,
            },
            "embedding_dimension": int(self.embedding.size),
            "embedding_sha256": hashlib.sha256(
                np.ascontiguousarray(self.embedding).tobytes()
            ).hexdigest(),
            "global_local_similarity": self.global_local_similarity,
            "same_support_similarity": self.same_support_similarity,
            "novelty": self.novelty,
            "solid_angle_sr": self.solid_angle_sr,
            "feature_count": int(self.feature_indices.size),
            "feature_density_per_sr": self.feature_density_per_sr,
        }


@dataclass(slots=True)
class MultiscaleFeatureSet:
    """Base spherical features plus scale/context evidence."""

    base: Any
    nodes: tuple[VisualContextNode, ...]
    feature_guidance: np.ndarray
    feature_scale_states: np.ndarray
    provider_name: str
    provider_version: str
    config: MultiscaleEmbeddingConfig

    def __post_init__(self) -> None:
        self.feature_guidance = np.asarray(self.feature_guidance, dtype=np.float32)
        self.feature_scale_states = np.asarray(self.feature_scale_states, dtype=object)
        if self.feature_guidance.shape != (len(self.base),):
            raise ValueError("feature_guidance must have shape (feature_count,)")
        if self.feature_scale_states.shape != (len(self.base),):
            raise ValueError("feature_scale_states must have shape (feature_count,)")
        if not np.isfinite(self.feature_guidance).all() or np.any(
            self.feature_guidance <= 0.0
        ):
            raise ValueError("feature_guidance must be finite and positive")

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError as exc:  # supports copy/pickle reconstruction
            raise AttributeError(name) from exc
        return getattr(base, name)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.config.interface,
            "base": self.base.describe(),
            "embedding_provider": {
                "name": self.provider_name,
                "version": self.provider_version,
            },
            "config": self.config.to_dict(),
            "selected_node_count": sum(node.selected for node in self.nodes),
            "candidate_node_count": len(self.nodes),
            "nodes": [node.describe() for node in self.nodes],
            "scale_evidence": {
                "minimum": float(self.feature_guidance.min())
                if self.feature_guidance.size
                else 1.0,
                "maximum": float(self.feature_guidance.max())
                if self.feature_guidance.size
                else 1.0,
                "mean": float(self.feature_guidance.mean())
                if self.feature_guidance.size
                else 1.0,
                "state_counts": {
                    str(name): int(np.sum(self.feature_scale_states == name))
                    for name in np.unique(self.feature_scale_states)
                },
            },
        }


@dataclass(slots=True)
class MultiscaleFeatureMatches:
    """Normal PanorAi matches with explicit proposal-only guidance."""

    base: SphericalFeatureMatches
    sampling_weights: np.ndarray
    match_sources: np.ndarray
    node_similarities: np.ndarray
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.base)
        self.sampling_weights = np.asarray(self.sampling_weights, dtype=np.float32)
        self.match_sources = np.asarray(self.match_sources, dtype=object)
        self.node_similarities = np.asarray(self.node_similarities, dtype=np.float32)
        for name, value in (
            ("sampling_weights", self.sampling_weights),
            ("match_sources", self.match_sources),
            ("node_similarities", self.node_similarities),
        ):
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (match_count,)")
        if not np.isfinite(self.sampling_weights).all() or np.any(
            self.sampling_weights <= 0.0
        ):
            raise ValueError("sampling_weights must be finite and positive")

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name: str) -> Any:
        try:
            base = object.__getattribute__(self, "base")
        except AttributeError as exc:  # supports copy/pickle reconstruction
            raise AttributeError(name) from exc
        return getattr(base, name)

    def to_bearing_correspondences(self) -> SphericalBearingCorrespondences:
        weights = self.sampling_weights * self.base.valid.astype(np.float32)
        return SphericalBearingCorrespondences(
            self.base.bearings_a.copy(),
            self.base.bearings_b.copy(),
            weights,
            self.base.valid.copy(),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "interface": "panorai-multiscale-visual-matches/v1",
            "base": self.base.describe(),
            "sampling_weight_range": (
                float(self.sampling_weights.min())
                if self.sampling_weights.size
                else 1.0,
                float(self.sampling_weights.max())
                if self.sampling_weights.size
                else 1.0,
            ),
            "source_counts": {
                str(name): int(np.sum(self.match_sources == name))
                for name in np.unique(self.match_sources)
            },
            "diagnostics": dict(self.diagnostics),
        }


class MultiscaleSphericalFeaturePipeline:
    """Adaptive wide/local feature pipeline with embedding graph routing."""

    def __init__(
        self,
        base_pipeline: SphericalFeaturePipeline,
        config: MultiscaleEmbeddingConfig | None = None,
        *,
        embedding_provider: VisualEmbeddingProvider | None = None,
    ) -> None:
        self.base_pipeline = base_pipeline
        self.config = config or MultiscaleEmbeddingConfig()
        self.embedding_provider = embedding_provider or OpenCVContextEmbedding(
            shape_hw=self.config.embedding_shape_hw
        )
        if not isinstance(self.embedding_provider, VisualEmbeddingProvider):
            raise TypeError(
                "embedding_provider must expose name, version and embed(images, masks=...)"
            )

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        multiscale_config: MultiscaleEmbeddingConfig | None = None,
        embedding_provider: VisualEmbeddingProvider | None = None,
        **base_options: Any,
    ) -> "MultiscaleSphericalFeaturePipeline":
        return cls(
            SphericalFeaturePipeline.from_preset(name, **base_options),
            multiscale_config,
            embedding_provider=embedding_provider,
        )

    def extract(
        self,
        panorama: Any,
        *,
        panorama_id: str | None = None,
        validity_mask: Any | None = None,
    ) -> MultiscaleFeatureSet:
        panorama = _as_panorama(panorama)
        face_spec = self.base_pipeline.config.face_set
        wide_views = panorama.views(
            face_spec.sampler,
            size=face_spec.shape_hw,
            fov=face_spec.effective_fov_deg,
            count=face_spec.count,
            subdivisions=face_spec.subdivisions,
        )
        wide_faces = list(wide_views)
        nodes: list[VisualContextNode] = []
        selected_local_faces: list[Any] = []
        for root_index, face in enumerate(wide_faces):
            root_image, root_mask = _embedding_input(
                face,
                projected_validity=_project_embedding_validity(
                    validity_mask, face.spec
                ),
            )
            root_embedding = self._embed([root_image], [root_mask])[0]
            root_node = VisualContextNode(
                node_id=str(face.face_id),
                parent_id=None,
                level=0,
                spec=face.spec,
                embedding=root_embedding,
                selected=True,
                solid_angle_sr=_rectangular_solid_angle(face.spec),
            )
            nodes.append(root_node)
            candidates: list[tuple[float, VisualContextNode, Any]] = []
            for local_index, local_spec in enumerate(
                _local_specs(face.spec, self.config)
            ):
                local_id = f"{face.face_id}:local-{local_index:02d}"
                local_face = _materialize_face(panorama, local_spec, local_id)
                local_image, local_mask = _embedding_input(
                    local_face,
                    projected_validity=_project_embedding_validity(
                        validity_mask, local_spec
                    ),
                )
                low_spec = GnomonicSpec(
                    center_lat_deg=local_spec.center_lat_deg,
                    center_lon_deg=local_spec.center_lon_deg,
                    hfov_deg=local_spec.hfov_deg,
                    vfov_deg=local_spec.vfov_deg,
                    roll_deg=local_spec.roll_deg,
                    output_shape_hw=_same_density_shape(face.spec, local_spec),
                )
                low_face = _materialize_face(panorama, low_spec, local_id + ":density")
                low_image, low_mask = _embedding_input(
                    low_face,
                    projected_validity=_project_embedding_validity(
                        validity_mask, low_spec
                    ),
                )
                local_embedding, low_embedding = self._embed(
                    [local_image, low_image], [local_mask, low_mask]
                )
                context_similarity = _cosine(root_embedding, local_embedding)
                scale_similarity = _cosine(low_embedding, local_embedding)
                novelty = max(1.0 - context_similarity, 1.0 - scale_similarity)
                selected = (
                    context_similarity < self.config.context_similarity_threshold
                    or scale_similarity < self.config.scale_similarity_threshold
                )
                node = VisualContextNode(
                    node_id=local_id,
                    parent_id=str(face.face_id),
                    level=1,
                    spec=local_spec,
                    embedding=local_embedding,
                    selected=selected,
                    global_local_similarity=context_similarity,
                    same_support_similarity=scale_similarity,
                    novelty=novelty,
                    solid_angle_sr=_rectangular_solid_angle(local_spec),
                )
                candidates.append((novelty, node, local_face))
            ranked = sorted(
                candidates,
                key=lambda item: (-item[0], item[1].node_id),
            )
            selected_ids = set(
                [
                    item[1].node_id
                    for item in ranked
                    if item[1].selected
                ][: self.config.max_local_views_per_root]
            )
            for _, node, local_face in candidates:
                node.selected = node.node_id in selected_ids
                nodes.append(node)
                if node.selected:
                    selected_local_faces.append(local_face)

        from panorai.data.gnomonic_imageset import GnomonicFaceSet

        combined = GnomonicFaceSet(wide_faces + selected_local_faces)
        # One NMS operates over wide and local faces together.  It is the same
        # spherical implementation used by the normal extractor, with an
        # explicit cross-scale radius rather than an implicit second pass.
        cross_scale_extractor = FeatureExtractor(
            replace(
                self.base_pipeline.config.extractor,
                angular_dedup_threshold_deg=self.config.cross_scale_nms_threshold_deg,
            ),
            backend=self.base_pipeline.backend,
            minimum_opencv_version=self.base_pipeline.config.minimum_opencv_version,
        )
        base = cross_scale_extractor.extract(
            panorama,
            face_set=combined,
            face_set_spec=FaceSetSpec(
                sampler=f"multiscale-{face_spec.sampler}",
                shape_hw=face_spec.shape_hw,
                fov_deg=face_spec.fov_deg,
                overlap_deg=face_spec.overlap_deg,
                count=face_spec.count,
                subdivisions=face_spec.subdivisions,
            ),
            panorama_id=panorama_id,
            validity_mask=validity_mask,
        )
        by_id = {node.node_id: node for node in nodes}
        feature_nodes: list[set[str]] = []
        for index, feature in enumerate(base.features):
            identities = {feature.face_id, *feature.provenance.alternative_face_ids}
            identities &= by_id.keys()
            feature_nodes.append(identities)
            for node_id in identities:
                node = by_id[node_id]
                node.feature_indices = np.append(node.feature_indices, index)
        for node in nodes:
            node.feature_indices = np.unique(node.feature_indices).astype(np.int64)
            node.feature_density_per_sr = float(node.feature_indices.size) / max(
                node.solid_angle_sr, 1e-12
            )
        guidance = _feature_scale_guidance(base, nodes, feature_nodes)
        scale_states = _feature_scale_states(nodes, feature_nodes)
        return MultiscaleFeatureSet(
            base=base,
            nodes=tuple(nodes),
            feature_guidance=guidance,
            feature_scale_states=scale_states,
            provider_name=str(self.embedding_provider.name),
            provider_version=str(self.embedding_provider.version),
            config=self.config,
        )

    def match(
        self,
        features_a: MultiscaleFeatureSet,
        features_b: MultiscaleFeatureSet,
    ) -> MultiscaleFeatureMatches:
        if not isinstance(features_a, MultiscaleFeatureSet) or not isinstance(
            features_b, MultiscaleFeatureSet
        ):
            raise TypeError("multiscale match requires two MultiscaleFeatureSet objects")
        left = features_a.base
        right = features_b.base
        if left.descriptor_metric != right.descriptor_metric:
            raise ValueError("feature sets must use the same descriptor metric")
        raw, method = self.base_pipeline.backend.match(
            left.descriptors,
            right.descriptors,
            left.descriptor_metric,
            self.base_pipeline.config.matcher,
        )
        evidence: dict[tuple[int, int], dict[str, Any]] = {}
        for item in raw:
            key = (item["query_idx"], item["train_idx"])
            evidence[key] = {
                "source": "global-fallback",
                "node_similarity": 0.0,
                "weight": self.config.fallback_weight
                * math.sqrt(
                    float(features_a.feature_guidance[key[0]])
                    * float(features_b.feature_guidance[key[1]])
                ),
            }
        node_pairs = _candidate_node_pairs(
            features_a.nodes,
            features_b.nodes,
            threshold=self.config.node_match_similarity_threshold,
            top_k=self.config.node_match_top_k,
        )
        regional_added = 0
        for node_a, node_b, similarity in node_pairs:
            indices_a = node_a.feature_indices
            indices_b = node_b.feature_indices
            if indices_a.size == 0 or indices_b.size < 2:
                continue
            regional, _ = self.base_pipeline.backend.match(
                left.descriptors[indices_a],
                right.descriptors[indices_b],
                left.descriptor_metric,
                self.base_pipeline.config.matcher,
            )
            for item in regional:
                mapped = dict(item)
                mapped["query_idx"] = int(indices_a[item["query_idx"]])
                mapped["train_idx"] = int(indices_b[item["train_idx"]])
                key = (mapped["query_idx"], mapped["train_idx"])
                guidance = (
                    self.config.fallback_weight
                    + self.config.regional_weight_gain * max(similarity, 0.0)
                ) * math.sqrt(
                    float(features_a.feature_guidance[key[0]])
                    * float(features_b.feature_guidance[key[1]])
                )
                previous = evidence.get(key)
                if previous is None:
                    raw.append(mapped)
                    regional_added += 1
                    evidence[key] = {
                        "source": "embedding-region",
                        "node_similarity": similarity,
                        "weight": guidance,
                    }
                elif guidance > previous["weight"]:
                    previous.update(
                        source="global+embedding-region",
                        node_similarity=max(
                            float(previous["node_similarity"]), similarity
                        ),
                        weight=guidance,
                    )
        # Remove exact duplicate indices before bilateral spherical NMS.
        exact: dict[tuple[int, int], dict[str, Any]] = {}
        for item in raw:
            key = (int(item["query_idx"]), int(item["train_idx"]))
            previous = exact.get(key)
            if previous is None or float(item["distance"]) < float(previous["distance"]):
                exact[key] = item
        raw = list(exact.values())
        if self.base_pipeline.config.matcher.deduplicate_matches:
            raw = _deduplicate_spherical_matches(
                raw,
                left,
                right,
                math.radians(
                    self.base_pipeline.config.matcher.angular_dedup_threshold_deg
                ),
            )
        else:
            for item in raw:
                item["face_pair_group"] = (
                    (
                        left.features[item["query_idx"]].face_id,
                        right.features[item["train_idx"]].face_id,
                    ),
                )
        raw.sort(key=lambda item: (item["query_idx"], item["train_idx"], item["distance"]))
        base_matches = _build_spherical_feature_matches(
            raw,
            left,
            right,
            config=self.base_pipeline.config.matcher,
            matcher_name=f"{method}+embedding-routing",
            backend_name=self.base_pipeline.backend.name,
            backend_version=self.base_pipeline.backend.version,
        )
        weights = []
        sources = []
        similarities = []
        for item in raw:
            key = (int(item["query_idx"]), int(item["train_idx"]))
            detail = evidence.get(
                key,
                {
                    "source": "global-fallback",
                    "node_similarity": 0.0,
                    "weight": self.config.fallback_weight,
                },
            )
            weights.append(max(float(detail["weight"]), 1e-6))
            sources.append(detail["source"])
            similarities.append(detail["node_similarity"])
        weights_array = np.asarray(weights, dtype=np.float32)
        if weights_array.size:
            weights_array /= max(float(np.mean(weights_array)), 1e-6)
        return MultiscaleFeatureMatches(
            base=base_matches,
            sampling_weights=weights_array,
            match_sources=np.asarray(sources, dtype=object),
            node_similarities=np.asarray(similarities, dtype=np.float32),
            diagnostics={
                "candidate_node_pairs": len(node_pairs),
                "regional_matches_added_before_nms": regional_added,
                "global_fallback_retained": True,
                "all_matches_scored_by_estimator": True,
            },
        )

    def extract_and_match(
        self,
        panorama_a: Any,
        panorama_b: Any,
        *,
        panorama_id_a: str | None = None,
        panorama_id_b: str | None = None,
        validity_mask_a: Any | None = None,
        validity_mask_b: Any | None = None,
    ) -> MultiscaleFeatureMatches:
        first = self.extract(
            panorama_a,
            panorama_id=panorama_id_a,
            validity_mask=validity_mask_a,
        )
        second = self.extract(
            panorama_b,
            panorama_id=panorama_id_b,
            validity_mask=validity_mask_b,
        )
        return self.match(first, second)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.config.interface,
            "status": "Experimental",
            "config": self.config.to_dict(),
            "embedding_provider": {
                "name": str(self.embedding_provider.name),
                "version": str(self.embedding_provider.version),
            },
            "base_pipeline": self.base_pipeline.describe(),
            "invariants": {
                "opencv_owns_local_matching": True,
                "global_fallback_retained": True,
                "embedding_guidance_is_proposal_only": True,
                "semantic_labels_required": False,
            },
        }

    def _embed(
        self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]
    ) -> np.ndarray:
        result = np.asarray(self.embedding_provider.embed(images, masks=masks))
        if result.ndim != 2 or result.shape[0] != len(images) or result.shape[1] == 0:
            raise ValueError("embedding provider must return shape (N, D) with D > 0")
        if not np.issubdtype(result.dtype, np.floating):
            raise TypeError("embedding provider must return floating vectors")
        if not np.isfinite(result).all():
            raise ValueError("embedding provider returned non-finite values")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return np.divide(
            result,
            norms,
            out=np.zeros_like(result, dtype=np.float32),
            where=norms > 0.0,
        ).astype(np.float32, copy=False)


def _materialize_face(panorama: Any, spec: GnomonicSpec, face_id: str):
    from panorai.data.gnomonic_image import GnomonicFace

    value = panorama.image
    projection_input = value
    if _is_torch(value):
        if not value.dtype.is_floating_point:
            import torch

            projection_input = value.to(dtype=torch.float32)
    elif not np.issubdtype(np.asarray(value).dtype, np.floating):
        projection_input = np.asarray(value, dtype=np.float32)
    result = GnomonicProjector(spec).project(projection_input)
    face = GnomonicFace(
        result.data,
        spec.center_lat_deg,
        spec.center_lon_deg,
        spec.hfov_deg,
        hfov_deg=spec.hfov_deg,
        vfov_deg=spec.vfov_deg,
        roll_deg=spec.roll_deg,
        face_id=face_id,
    )
    face.spec = spec
    face.support_mask = result.support_mask
    face._erp_shape_hw = _spatial_shape(value)
    support = result.support_mask.clone() if _is_torch(result.support_mask) else result.support_mask.copy()
    face._workflow_metadata = {
        "image": {
            "kind": "image",
            "units": None,
            "validity": support.clone() if _is_torch(support) else support.copy(),
            "interpolation": "bilinear",
        }
    }
    face._workflow_support = {"image": support}
    return face


def _embedding_input(
    face: Any, *, projected_validity: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    mask = face.support_mask
    mask_np = mask.detach().cpu().numpy() if _is_torch(mask) else np.asarray(mask)
    image = face.image
    array = image.detach().cpu().numpy() if _is_torch(image) else np.asarray(image)
    if _is_torch(image) and array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    combined_mask = mask_np.astype(bool, copy=False)
    if projected_validity is not None:
        valid_np = (
            projected_validity.detach().cpu().numpy()
            if _is_torch(projected_validity)
            else np.asarray(projected_validity)
        )
        if valid_np.shape != combined_mask.shape or valid_np.dtype != np.bool_:
            raise TypeError(
                "projected embedding validity must be boolean with face shape"
            )
        combined_mask = combined_mask & valid_np
    working = np.array(array, copy=True)
    if working.ndim == 3:
        working[~combined_mask, :] = 0
    else:
        working[~combined_mask] = 0
    array = _to_hwc_uint8(working)
    return np.ascontiguousarray(array), combined_mask


def _fill_invalid_nearest(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill invalid pixels from nearest valid support without a black edge."""

    if mask.all():
        return np.array(array, copy=True)
    if not mask.any():
        return np.zeros_like(array)
    from scipy.ndimage import distance_transform_edt

    nearest = distance_transform_edt(
        ~mask, return_distances=False, return_indices=True
    )
    filled = np.array(array, copy=True)
    replacement = array[nearest[0], nearest[1]]
    filled[~mask] = replacement[~mask]
    return filled


def _project_embedding_validity(validity_mask: Any | None, spec: GnomonicSpec):
    if validity_mask is None:
        return None
    result = equirectangular_to_gnomonic(
        validity_mask, spec, interpolation="nearest"
    ).data
    return result


def _to_hwc_uint8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in {2, 3}:
        raise ValueError("embedding images must use HW or HWC layout")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise TypeError("embedding images must be uint8 or finite floating arrays")
        maximum = float(array.max(initial=0.0))
        if maximum <= 1.0:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif maximum <= 255.0 and float(array.min(initial=0.0)) >= 0.0:
            array = np.rint(array).astype(np.uint8)
        else:
            raise ValueError("floating embedding images must be scaled to [0,1] or [0,255]")
    return np.ascontiguousarray(array)


def _local_specs(
    parent: GnomonicSpec, config: MultiscaleEmbeddingConfig
) -> tuple[GnomonicSpec, ...]:
    height, width = parent.output_shape_hw
    grid = config.local_grid_size
    xs = (np.arange(grid, dtype=np.float64) + 0.5) * (width / grid) - 0.5
    ys = (np.arange(grid, dtype=np.float64) + 0.5) * (height / grid) - 0.5
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    rays = np.asarray(gnomonic_pixels_to_rays(points, parent).rays_xyz)
    specs = []
    for ray in rays:
        latitude = math.degrees(math.asin(float(np.clip(ray[1], -1.0, 1.0))))
        longitude = math.degrees(math.atan2(float(ray[0]), float(ray[2])))
        specs.append(
            GnomonicSpec(
                center_lat_deg=latitude,
                center_lon_deg=longitude,
                hfov_deg=config.local_fov_deg[0],
                vfov_deg=config.local_fov_deg[1],
                roll_deg=parent.roll_deg,
                output_shape_hw=parent.output_shape_hw,
            )
        )
    return tuple(specs)


def _same_density_shape(parent: GnomonicSpec, child: GnomonicSpec) -> tuple[int, int]:
    parent_h, parent_w = parent.output_shape_hw
    width = round(
        parent_w
        * math.tan(math.radians(child.hfov_deg) / 2.0)
        / math.tan(math.radians(parent.hfov_deg) / 2.0)
    )
    height = round(
        parent_h
        * math.tan(math.radians(child.vfov_deg) / 2.0)
        / math.tan(math.radians(parent.vfov_deg) / 2.0)
    )
    return max(16, height), max(16, width)


def _rectangular_solid_angle(spec: GnomonicSpec) -> float:
    horizontal = math.tan(math.radians(spec.hfov_deg) / 2.0)
    vertical = math.tan(math.radians(spec.vfov_deg) / 2.0)
    return 4.0 * math.atan2(
        horizontal * vertical,
        math.sqrt(1.0 + horizontal * horizontal + vertical * vertical),
    )


def _feature_scale_guidance(
    base: Any,
    nodes: list[VisualContextNode],
    feature_nodes: list[set[str]],
) -> np.ndarray:
    by_id = {node.node_id: node for node in nodes}
    result = np.ones(len(base), dtype=np.float32)
    for index, identities in enumerate(feature_nodes):
        selected = [by_id[item] for item in identities if by_id[item].selected]
        if not selected:
            continue
        levels = len({item.level for item in selected})
        persistence = 1.0 + 0.35 * max(levels - 1, 0)
        density_evidence = 1.0
        for node in selected:
            if node.parent_id is None:
                continue
            parent = by_id[node.parent_id]
            if parent.feature_density_per_sr > 0.0:
                ratio = node.feature_density_per_sr / parent.feature_density_per_sr
                density_evidence = max(density_evidence, math.sqrt(max(ratio, 1e-6)))
        result[index] = float(np.clip(persistence * density_evidence, 0.5, 4.0))
    if result.size:
        result /= max(float(result.mean()), 1e-6)
    return result


def _feature_scale_states(
    nodes: list[VisualContextNode], feature_nodes: list[set[str]]
) -> np.ndarray:
    by_id = {node.node_id: node for node in nodes}
    states: list[str] = []
    for identities in feature_nodes:
        levels = [by_id[item].level for item in identities if item in by_id]
        local_count = sum(level > 0 for level in levels)
        if 0 in levels and local_count:
            states.append("persistent-multiscale")
        elif local_count > 1:
            states.append("local-split-support")
        elif local_count == 1:
            states.append("local-birth")
        else:
            states.append("wide-only")
    return np.asarray(states, dtype=object)


def _candidate_node_pairs(
    nodes_a: tuple[VisualContextNode, ...],
    nodes_b: tuple[VisualContextNode, ...],
    *,
    threshold: float,
    top_k: int,
) -> list[tuple[VisualContextNode, VisualContextNode, float]]:
    active_a = [item for item in nodes_a if item.selected and item.feature_indices.size]
    active_b = [item for item in nodes_b if item.selected and item.feature_indices.size]
    if not active_a or not active_b:
        return []
    dimensions = {item.embedding.size for item in (*active_a, *active_b)}
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions changed between context nodes")
    matrix_a = np.stack([item.embedding for item in active_a])
    matrix_b = np.stack([item.embedding for item in active_b])
    similarity = matrix_a @ matrix_b.T
    left_choices = {
        (left, int(right))
        for left in range(len(active_a))
        for right in np.argsort(-similarity[left], kind="stable")[:top_k]
    }
    right_choices = {
        (int(left), right)
        for right in range(len(active_b))
        for left in np.argsort(-similarity[:, right], kind="stable")[:top_k]
    }
    mutual = left_choices & right_choices
    return [
        (active_a[left], active_b[right], float(similarity[left, right]))
        for left, right in sorted(mutual)
        if float(similarity[left, right]) >= threshold
    ]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pair(
    value: Any, name: str, *, lower: float, upper: float
) -> tuple[float, float]:
    try:
        result = tuple(_finite(item, name) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two real values") from exc
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    if any(not lower < item < upper for item in result):
        raise ValueError(f"{name} values must be in ({lower}, {upper})")
    return result  # type: ignore[return-value]


def _int_pair(value: Any, name: str) -> tuple[int, int]:
    try:
        result = tuple(_positive_int(item, name) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain two integers") from exc
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result  # type: ignore[return-value]


__all__ = [
    "MultiscaleEmbeddingConfig",
    "MultiscaleFeatureMatches",
    "MultiscaleFeatureSet",
    "MultiscaleSphericalFeaturePipeline",
    "OpenCVContextEmbedding",
    "VisualContextNode",
    "VisualEmbeddingProvider",
]
