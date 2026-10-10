"""Data contracts for Experimental spherical semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Literal

import numpy as np

SPHERICAL_SEMANTIC_SEGMENTATION_INTERFACE = "panorai-spherical-semantic-segmentation/v1"

SeedSource = Literal["semantic", "coverage", "frontier"]


def _readonly(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SphericalBinaryMask:
    """Bit-packed boolean mask in the canonical ERP pixel grid."""

    shape_hw: tuple[int, int]
    packed_bits: np.ndarray

    def __post_init__(self) -> None:
        height, width = self.shape_hw
        if height < 1 or width < 1:
            raise ValueError("shape_hw must contain positive dimensions")
        packed = _readonly(self.packed_bits, dtype=np.uint8).reshape(-1)
        expected = (height * width + 7) // 8
        if len(packed) != expected:
            raise ValueError(
                f"packed_bits has length {len(packed)}, expected {expected}"
            )
        object.__setattr__(self, "packed_bits", packed)

    @classmethod
    def from_array(cls, mask: np.ndarray) -> SphericalBinaryMask:
        values = np.asarray(mask)
        if values.ndim != 2:
            raise ValueError("mask must have shape (H,W)")
        packed = np.packbits(values.astype(bool).reshape(-1), bitorder="little")
        return cls(tuple(int(value) for value in values.shape), packed)

    def to_array(self) -> np.ndarray:
        """Return a writable boolean copy."""

        height, width = self.shape_hw
        return (
            np.unpackbits(self.packed_bits, bitorder="little", count=height * width)
            .reshape(height, width)
            .astype(bool, copy=False)
        )

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.to_array()))

    def solid_angle_area(self, support: np.ndarray | None = None) -> float:
        """Return the mask area in steradians using ERP pixel-center weights."""

        mask = self.to_array()
        if support is not None:
            support_array = np.asarray(support, dtype=bool)
            if support_array.shape != self.shape_hw:
                raise ValueError("support and mask shapes must match")
            mask &= support_array
        height, width = self.shape_hw
        latitude = math.pi / 2 - (np.arange(height) + 0.5) * math.pi / height
        pixel_area = (2 * math.pi / width) * (math.pi / height) * np.cos(latitude)
        return float(np.sum(mask * pixel_area[:, None]))

    def solid_angle_iou(
        self,
        other: SphericalBinaryMask,
        *,
        support: np.ndarray | None = None,
    ) -> float:
        """Return solid-angle-weighted intersection over union."""

        if self.shape_hw != other.shape_hw:
            raise ValueError("mask shapes must match")
        first = self.to_array()
        second = other.to_array()
        if support is not None:
            valid = np.asarray(support, dtype=bool)
            if valid.shape != self.shape_hw:
                raise ValueError("support and mask shapes must match")
            first &= valid
            second &= valid
        intersection = SphericalBinaryMask.from_array(first & second).solid_angle_area()
        union = SphericalBinaryMask.from_array(first | second).solid_angle_area()
        return intersection / union if union > 0 else 1.0


@dataclass(frozen=True, slots=True)
class SemanticProxyConcept:
    """One human-readable concept supported by explicit classifier classes."""

    concept_id: str
    display_name: str
    class_weights: tuple[tuple[int, str, float], ...]

    def __post_init__(self) -> None:
        if not self.concept_id.strip() or not self.display_name.strip():
            raise ValueError("concept identifiers and names must be non-empty")
        if not self.class_weights:
            raise ValueError("class_weights must not be empty")
        indices: list[int] = []
        total = 0.0
        for class_index, class_name, weight in self.class_weights:
            if isinstance(class_index, bool) or not isinstance(class_index, Integral):
                raise TypeError("class indices must be integers")
            if class_index < 0 or not class_name.strip():
                raise ValueError("class indices and names must be valid")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("class weights must be finite and positive")
            indices.append(int(class_index))
            total += float(weight)
        if len(indices) != len(set(indices)):
            raise ValueError("a concept must not repeat a class index")
        normalized = tuple(
            (int(index), name, float(weight) / total)
            for index, name, weight in self.class_weights
        )
        object.__setattr__(self, "class_weights", normalized)


@dataclass(frozen=True, slots=True)
class DenseSemanticEvidence:
    """Normalized concept evidence on a supported canonical ERP lattice."""

    concept_ids: tuple[str, ...]
    scores: np.ndarray
    support: np.ndarray
    vocabulary: str = "imagenet-proxy/v1"
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scores = _readonly(self.scores, dtype=np.float32)
        support = _readonly(self.support, dtype=bool)
        if scores.ndim != 3 or support.shape != scores.shape[-2:]:
            raise ValueError("scores must be CHW and support must match HW")
        if scores.shape[0] != len(self.concept_ids) or not self.concept_ids:
            raise ValueError("concept_ids must align with score channels")
        if len(set(self.concept_ids)) != len(self.concept_ids):
            raise ValueError("concept_ids must be unique")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("semantic scores must be finite and within [0,1]")
        if not np.any(support):
            raise ValueError("semantic support must contain valid samples")
        if np.any(scores[:, ~support] != 0):
            raise ValueError("scores outside support must be zero")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True, slots=True)
class SphericalSeed:
    seed_id: str
    longitude_degrees: float
    latitude_degrees: float
    source: SeedSource
    score: float
    concept_id: str | None = None

    def __post_init__(self) -> None:
        if not self.seed_id:
            raise ValueError("seed_id must be non-empty")
        if self.source not in {"semantic", "coverage", "frontier"}:
            raise ValueError("invalid seed source")
        if not -180 <= self.longitude_degrees < 180:
            raise ValueError("longitude must be in [-180,180)")
        if not -90 <= self.latitude_degrees <= 90:
            raise ValueError("latitude must be in [-90,90]")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("seed score must be finite and within [0,1]")


@dataclass(frozen=True, slots=True)
class SphericalMaskProposal:
    proposal_id: str
    seed: SphericalSeed
    alternatives: tuple[SphericalBinaryMask, SphericalBinaryMask, SphericalBinaryMask]
    consensus: SphericalBinaryMask
    envelope: SphericalBinaryMask
    predicted_iou: tuple[float, float, float]
    pairwise_iou: float
    stability_score: float
    semantic_score: float
    concept_id: str | None
    proxy_classes: tuple[str, ...] = ()
    parent_proposal_id: str | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("proposal_id must be non-empty")
        shapes = {
            mask.shape_hw
            for mask in (*self.alternatives, self.consensus, self.envelope)
        }
        if len(shapes) != 1:
            raise ValueError("all proposal masks must share one shape")
        if len(self.predicted_iou) != 3:
            raise ValueError("predicted_iou must contain three scores")
        for name in ("pairwise_iou", "stability_score", "semantic_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0,1]")

    @property
    def quality_score(self) -> float:
        scores = sorted((float(value) for value in self.predicted_iou), reverse=True)
        pair_score = 0.5 * (scores[0] + scores[1])
        return float(
            0.45 * self.pairwise_iou
            + 0.30 * pair_score
            + 0.15 * self.stability_score
            + 0.10 * self.semantic_score
        )


@dataclass(frozen=True, slots=True)
class SphericalSegment:
    segment_id: int
    concept_id: str | None
    concept_name: str
    proxy_classes: tuple[str, ...]
    consensus: SphericalBinaryMask
    envelope: SphericalBinaryMask
    quality_score: float
    semantic_score: float
    proposal_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.segment_id < 1 or not self.concept_name or not self.proposal_ids:
            raise ValueError("segment identity and proposal_ids must be non-empty")
        if self.consensus.shape_hw != self.envelope.shape_hw:
            raise ValueError("segment masks must share one shape")
        for name in ("quality_score", "semantic_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0,1]")


@dataclass(frozen=True, slots=True)
class SphericalSegmentationConfig:
    """Reproducible exhaustive operating point."""

    semantic_peaks_per_concept: int = 12
    semantic_peak_separation_degrees: float = 12.0
    semantic_relative_threshold: float = 0.30
    coverage_direction_count: int = 96
    discovery_fov_degrees: float = 70.0
    discovery_face_size: int = 1024
    prompt_grid_size: int = 8
    prompt_batch_size: int = 16
    minimum_pairwise_iou: float = 0.50
    minimum_predicted_iou: float = 0.70
    minimum_stability: float = 0.85
    minimum_pixels: int = 64
    expansion_fov_degrees: float = 30.0
    expansion_step_fraction: float = 0.65
    expansion_frontier_band_pixels: int = 32
    expansion_frontier_minimum_pixels: int = 128
    expansion_prior_retention: float = 0.80
    maximum_expansion_faces: int = 24
    duplicate_iou: float = 0.80
    merge_iou: float = 0.55
    merge_containment: float = 0.80
    semantic_minimum_mean: float = 0.25
    semantic_minimum_contrast: float = 0.05

    def __post_init__(self) -> None:
        integer_names = (
            "semantic_peaks_per_concept",
            "coverage_direction_count",
            "discovery_face_size",
            "prompt_grid_size",
            "prompt_batch_size",
            "minimum_pixels",
            "expansion_frontier_band_pixels",
            "expansion_frontier_minimum_pixels",
            "maximum_expansion_faces",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        unit_interval_names = (
            "semantic_relative_threshold",
            "minimum_pairwise_iou",
            "minimum_predicted_iou",
            "minimum_stability",
            "expansion_step_fraction",
            "expansion_prior_retention",
            "duplicate_iou",
            "merge_iou",
            "merge_containment",
            "semantic_minimum_mean",
            "semantic_minimum_contrast",
        )
        for name in unit_interval_names:
            value = getattr(self, name)
            if not isinstance(value, Real) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be within [0,1]")
        for name in (
            "semantic_peak_separation_degrees",
            "discovery_fov_degrees",
            "expansion_fov_degrees",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value < 180:
                raise ValueError(f"{name} must be finite and within (0,180)")


@dataclass(frozen=True, slots=True)
class SphericalSegmentationResult:
    segments: tuple[SphericalSegment, ...]
    proposals: tuple[SphericalMaskProposal, ...]
    panoptic_map: np.ndarray
    support: np.ndarray
    diagnostics: dict[str, object]
    interface: str = SPHERICAL_SEMANTIC_SEGMENTATION_INTERFACE

    def __post_init__(self) -> None:
        panoptic = _readonly(self.panoptic_map, dtype=np.uint32)
        support = _readonly(self.support, dtype=bool)
        if panoptic.ndim != 2 or support.shape != panoptic.shape:
            raise ValueError("panoptic_map and support must be matching HW arrays")
        if np.any(panoptic[~support] != 0):
            raise ValueError("unsupported pixels must have panoptic ID zero")
        valid_ids = {segment.segment_id for segment in self.segments}
        if not set(np.unique(panoptic)).issubset({0, *valid_ids}):
            raise ValueError("panoptic_map contains an unknown segment ID")
        object.__setattr__(self, "panoptic_map", panoptic)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def unknown_segments(self) -> tuple[SphericalSegment, ...]:
        """Segments retained without sufficient semantic evidence."""

        return tuple(segment for segment in self.segments if segment.concept_id is None)
