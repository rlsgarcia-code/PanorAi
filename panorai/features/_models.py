"""PanorAi-owned result objects for spherical features and matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from panorai.geometry import GnomonicSpec

from ._config import FEATURES_INTERFACE, FEATURES_STABILITY, FaceSetSpec


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    interface: str
    source_panorama_checksum: str
    projection_backend: str
    projection_backend_version: str
    face_id: str
    generating_commit: str | None = None
    alternative_face_ids: tuple[str, ...] = ()
    duplicate_descriptor_indices: tuple[int, ...] = ()
    duplicate_angular_distances_deg: tuple[float, ...] = ()
    selection_reason: str = "highest-response-then-source-order"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FaceDetectionDiagnostics:
    """Detection and validity counts for one projected face."""

    face_id: str
    detected_count: int
    valid_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultifaceDetectionDiagnostics:
    """Stage counts and overlap ambiguity for a multiface extraction."""

    face_count: int
    per_face_capacity: int
    per_face_capacity_hit_count: int
    detected_count: int
    valid_count: int
    unique_after_deduplication: int
    output_after_budget: int
    duplicate_group_count: int
    duplicate_candidate_count: int
    cross_face_ambiguous_group_count: int
    cross_face_duplicate_candidate_count: int
    same_face_duplicate_candidate_count: int
    maximum_group_multiplicity: int
    multiplicity_histogram: dict[str, int]
    maximum_unique_face_multiplicity: int
    unique_face_multiplicity_histogram: dict[str, int]
    descriptor_pair_count: int
    normalized_descriptor_l2_median: float | None
    normalized_descriptor_l2_p90: float | None
    normalized_descriptor_l2_maximum: float | None
    descriptor_cosine_similarity_median: float | None
    descriptor_cosine_similarity_p10: float | None
    duplicate_face_pair_histogram: dict[str, int]
    per_face: tuple[FaceDetectionDiagnostics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "per_face": [item.to_dict() for item in self.per_face],
        }


@dataclass(frozen=True, slots=True)
class MatchProvenance:
    interface: str
    source_checksums: tuple[str, str]
    face_pairs: tuple[tuple[str, str], ...]
    face_pair_groups: tuple[tuple[tuple[str, str], ...], ...]
    deduplicated: bool
    selection_reason: str = "minimum-distance-then-source-order"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SphericalFeature:
    feature_id: str
    panorama_id: str
    face_id: str
    pixel_xy: np.ndarray
    source_erp_xy: np.ndarray | None
    bearing_xyz: np.ndarray
    response: float
    scale: float
    angle_deg: float
    octave: int
    descriptor_index: int
    valid: bool
    projection_spec: GnomonicSpec
    provenance: FeatureProvenance


@dataclass(slots=True)
class SphericalFeatureSet:
    panorama_id: str
    features: list[SphericalFeature]
    descriptors: np.ndarray
    descriptor_type: str
    descriptor_metric: str
    extractor_name: str
    extractor_config: dict[str, Any]
    backend_name: str
    backend_version: str
    face_set_spec: FaceSetSpec
    projection_backend: str
    projection_backend_version: str
    panorama_checksum: str
    generating_commit: str | None = None
    interface: str = FEATURES_INTERFACE
    stability: str = FEATURES_STABILITY
    detection_diagnostics: MultifaceDetectionDiagnostics | None = None

    def __post_init__(self) -> None:
        self.descriptors = np.asarray(self.descriptors)
        if self.descriptors.ndim != 2:
            raise ValueError("descriptors must have shape (N, D)")
        if len(self.features) != self.descriptors.shape[0]:
            raise ValueError("feature and descriptor counts must match")
        for index, feature in enumerate(self.features):
            if feature.descriptor_index != index:
                raise ValueError("descriptor_index must match final descriptor row")

    def __len__(self) -> int:
        return len(self.features)

    @property
    def bearings(self) -> np.ndarray:
        return _feature_array(self.features, "bearing_xyz", 3)

    @property
    def pixels_xy(self) -> np.ndarray:
        return _feature_array(self.features, "pixel_xy", 2)

    @property
    def source_erp_xy(self) -> np.ndarray:
        if not self.features:
            return np.empty((0, 2), dtype=np.float64)
        values = [
            (
                np.full(2, np.nan, dtype=np.float64)
                if item.source_erp_xy is None
                else np.asarray(item.source_erp_xy)
            )
            for item in self.features
        ]
        return np.stack(values)

    @property
    def responses(self) -> np.ndarray:
        return np.asarray([item.response for item in self.features], dtype=np.float32)

    @property
    def face_ids(self) -> np.ndarray:
        return np.asarray([item.face_id for item in self.features], dtype=object)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "panorama_id": self.panorama_id,
            "feature_count": len(self),
            "descriptor_shape": tuple(self.descriptors.shape),
            "descriptor_dtype": str(self.descriptors.dtype),
            "descriptor_type": self.descriptor_type,
            "descriptor_metric": self.descriptor_metric,
            "extractor": self.extractor_name,
            "extractor_config": dict(self.extractor_config),
            "backend": {"name": self.backend_name, "version": self.backend_version},
            "face_set": self.face_set_spec.to_dict(),
            "projection": {
                "backend": self.projection_backend,
                "version": self.projection_backend_version,
            },
            "source": {
                "panorama_checksum": self.panorama_checksum,
                "generating_commit": self.generating_commit,
            },
            "detection_diagnostics": (
                None
                if self.detection_diagnostics is None
                else self.detection_diagnostics.to_dict()
            ),
        }


def _feature_array(
    features: list[SphericalFeature], name: str, width: int
) -> np.ndarray:
    if not features:
        return np.empty((0, width), dtype=np.float64)
    return np.stack([np.asarray(getattr(item, name)) for item in features])


@dataclass(frozen=True, slots=True)
class SphericalBearingCorrespondences:
    bearings_a: np.ndarray
    bearings_b: np.ndarray
    weights: np.ndarray
    valid: np.ndarray


@dataclass(slots=True)
class SphericalFeatureMatches:
    panorama_id_a: str
    panorama_id_b: str
    feature_indices_a: np.ndarray
    feature_indices_b: np.ndarray
    bearings_a: np.ndarray
    bearings_b: np.ndarray
    descriptor_distances: np.ndarray
    ratio_scores: np.ndarray | None
    mutual: np.ndarray | None
    valid: np.ndarray
    matcher_name: str
    matcher_config: dict[str, Any]
    backend_name: str
    backend_version: str
    provenance: MatchProvenance
    keypoint_responses: np.ndarray
    face_ids_a: np.ndarray
    face_ids_b: np.ndarray
    interface: str = FEATURES_INTERFACE
    stability: str = FEATURES_STABILITY

    def __post_init__(self) -> None:
        self.feature_indices_a = np.asarray(self.feature_indices_a, dtype=np.int64)
        self.feature_indices_b = np.asarray(self.feature_indices_b, dtype=np.int64)
        self.bearings_a = np.asarray(self.bearings_a)
        self.bearings_b = np.asarray(self.bearings_b)
        self.descriptor_distances = np.asarray(
            self.descriptor_distances, dtype=np.float32
        )
        self.valid = np.asarray(self.valid, dtype=bool)
        self.keypoint_responses = np.asarray(self.keypoint_responses, dtype=np.float32)
        self.face_ids_a = np.asarray(self.face_ids_a, dtype=object)
        self.face_ids_b = np.asarray(self.face_ids_b, dtype=object)
        if self.ratio_scores is not None:
            self.ratio_scores = np.asarray(self.ratio_scores, dtype=np.float32)
        if self.mutual is not None:
            self.mutual = np.asarray(self.mutual, dtype=bool)
        one_dimensional = (
            ("feature_indices_a", self.feature_indices_a),
            ("feature_indices_b", self.feature_indices_b),
            ("descriptor_distances", self.descriptor_distances),
            ("valid", self.valid),
            ("face_ids_a", self.face_ids_a),
            ("face_ids_b", self.face_ids_b),
        )
        for name, array in one_dimensional:
            if array.ndim != 1:
                raise ValueError(f"{name} must have shape (N,)")
        count = self.feature_indices_a.shape[0]
        arrays = (
            self.feature_indices_b,
            self.bearings_a,
            self.bearings_b,
            self.descriptor_distances,
            self.valid,
            self.keypoint_responses,
            self.face_ids_a,
            self.face_ids_b,
        )
        if any(array.shape[0] != count for array in arrays):
            raise ValueError("all match fields must have the same leading dimension")
        if self.bearings_a.shape != (count, 3) or self.bearings_b.shape != (count, 3):
            raise ValueError("match bearings must have shape (N, 3)")
        if self.keypoint_responses.shape != (count, 2):
            raise ValueError("keypoint_responses must have shape (N, 2)")
        if self.ratio_scores is not None and self.ratio_scores.shape != (count,):
            raise ValueError("ratio_scores must have shape (N,)")
        if self.mutual is not None and self.mutual.shape != (count,):
            raise ValueError("mutual must have shape (N,)")

    def __len__(self) -> int:
        return int(self.feature_indices_a.shape[0])

    def to_bearing_correspondences(self) -> SphericalBearingCorrespondences:
        """Return uniform-weight spherical correspondences for estimators.

        Descriptor metrics have incompatible scales, so PanorAi deliberately
        does not invent a cross-descriptor confidence calibration here.
        """

        weights = self.valid.astype(np.float32)
        return SphericalBearingCorrespondences(
            self.bearings_a.copy(), self.bearings_b.copy(), weights, self.valid.copy()
        )

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "panorama_ids": (self.panorama_id_a, self.panorama_id_b),
            "match_count": len(self),
            "valid_count": int(self.valid.sum()),
            "matcher": self.matcher_name,
            "matcher_config": dict(self.matcher_config),
            "backend": {"name": self.backend_name, "version": self.backend_version},
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    selected_indices: np.ndarray
    group_index_for_input: np.ndarray
    angular_distance_to_selected_rad: np.ndarray
    duplicate_groups: tuple[tuple[int, ...], ...]
    selection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GnomonicRigCamera:
    camera_id: str
    face_id: str
    width: int
    height: int
    K: np.ndarray
    R_panorama_from_face: np.ndarray
    translation_panorama_from_face: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class GnomonicRig:
    panorama_id: str
    cameras: tuple[GnomonicRigCamera, ...]
    face_set_spec: FaceSetSpec
    interface: str = "panorai-spherical-rig/v1"


@dataclass(frozen=True, slots=True)
class PyCOLMAPExportResult:
    database_path: str
    camera_ids: dict[str, int]
    image_ids: dict[str, int]
    feature_rows: dict[str, np.ndarray]
    panorama_rig_rotations: dict[str, np.ndarray]
    interface: str = "panorai-pycolmap-export/v1"
