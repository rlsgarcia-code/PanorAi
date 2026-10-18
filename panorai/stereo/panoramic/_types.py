"""Immutable contracts for end-to-end panoramic dense stereo."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from panorai.geometry import GnomonicSpec


def _readonly(
    value: Any, dtype: Any, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PanoramicStereoPose:
    """Relative pose using ``X_source = R @ X_reference + t``.

    ``translation`` is metric only when ``scale_status == 'metric'``.  A pose
    estimated from two central cameras has direction but no scale; PanorAi then
    uses a unit baseline and labels the result ``baseline-units``.
    """

    rotation_source_from_reference: np.ndarray
    translation_source_from_reference: np.ndarray
    baseline: float
    units: str
    scale_status: str
    provenance: str

    def __post_init__(self) -> None:
        rotation = _readonly(
            self.rotation_source_from_reference, np.float64, "rotation", (3, 3)
        )
        translation = _readonly(
            self.translation_source_from_reference, np.float64, "translation", (3,)
        )
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("pose arrays must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("rotation must have determinant +1")
        if not np.isfinite(self.baseline) or self.baseline <= 0.0:
            raise ValueError("baseline must be positive and finite")
        if not np.isclose(
            np.linalg.norm(translation), self.baseline, rtol=1e-7, atol=1e-9
        ):
            raise ValueError("translation norm must equal baseline")
        if self.scale_status not in {"metric", "scale-free"}:
            raise ValueError("scale_status must be 'metric' or 'scale-free'")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty string")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance must be a non-empty string")
        object.__setattr__(self, "rotation_source_from_reference", rotation)
        object.__setattr__(self, "translation_source_from_reference", translation)


@dataclass(frozen=True, slots=True)
class RectifiedStereoViewSpec:
    """One deterministically paired rectified virtual-camera specification."""

    pair_id: str
    reference_spec: GnomonicSpec
    source_spec: GnomonicSpec
    longitude_index: int


@dataclass(frozen=True, slots=True)
class StereoViewPlan:
    """A finite ring of rectified pinhole pairs around the stereo baseline."""

    pairs: tuple[RectifiedStereoViewSpec, ...]
    baseline_axis_reference: np.ndarray
    estimated_angular_coverage: float
    omitted_epipolar_cone_deg: float

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("a stereo view plan must contain at least one pair")
        axis = _readonly(
            self.baseline_axis_reference,
            np.float64,
            "baseline_axis_reference",
            (3,),
        )
        norm = np.linalg.norm(axis)
        if not np.isclose(norm, 1.0, atol=1e-8):
            raise ValueError("baseline_axis_reference must have unit norm")
        object.__setattr__(self, "baseline_axis_reference", axis)


@dataclass(frozen=True, slots=True)
class StereoViewDescriptor:
    """Geometry and support used to pair independently sampled views."""

    view_id: str
    spec: GnomonicSpec
    support_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.support_mask is not None:
            support = _readonly(
                self.support_mask,
                np.bool_,
                "support_mask",
                self.spec.output_shape_hw,
            )
            object.__setattr__(self, "support_mask", support)


@dataclass(frozen=True, slots=True)
class StereoViewPairScore:
    """Auditable score for one candidate pair of independently sampled views."""

    reference_view_id: str
    source_view_id: str
    score: float
    angular_overlap: float
    sparse_inlier_coverage: float
    epipolar_consistency: float
    support_fraction: float


@dataclass(frozen=True, slots=True)
class StereoPairDiagnostic:
    """Compact per-pair result retained after its cost volume is released."""

    pair_id: str
    valid_fraction: float
    median_confidence: float
    median_range: float
    method: str


@dataclass(frozen=True, slots=True)
class StereoViewSelectionDiagnostic:
    """Quality-gate decision made before constructing a dense cost volume."""

    pair_id: str
    accepted: bool
    joint_support_fraction: float
    texture_std: float
    pose_inliers: int
    median_epipolar_row_error_px: float
    disparity_interval_viable: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PanoramicStereoResult:
    """Dense radial range in the reference ERP and complete audit metadata."""

    radial_range: np.ndarray
    validity: np.ndarray
    confidence: np.ndarray
    disagreement: np.ndarray
    contribution_count: np.ndarray
    pose: PanoramicStereoPose
    view_pairs: tuple[RectifiedStereoViewSpec, ...]
    view_selection: tuple[StereoViewSelectionDiagnostic, ...]
    pair_diagnostics: tuple[StereoPairDiagnostic, ...]
    diagnostics: Mapping[str, Any]
    interface: str = "panorai-panoramic-dense-stereo/v1-experimental"
    matching_confidence: np.ndarray | None = None
    consistency_confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        radial = np.asarray(self.radial_range)
        if radial.ndim != 2:
            raise ValueError("radial_range must have shape (H, W)")
        shape = radial.shape
        radial = _readonly(radial, np.float32, "radial_range", shape)
        validity = _readonly(self.validity, np.bool_, "validity", shape)
        confidence = _readonly(self.confidence, np.float32, "confidence", shape)
        matching_confidence = _readonly(
            confidence
            if self.matching_confidence is None
            else self.matching_confidence,
            np.float32,
            "matching_confidence",
            shape,
        )
        consistency_confidence = _readonly(
            (
                np.where(validity, 1.0, 0.0)
                if self.consistency_confidence is None
                else self.consistency_confidence
            ),
            np.float32,
            "consistency_confidence",
            shape,
        )
        disagreement = _readonly(self.disagreement, np.float32, "disagreement", shape)
        count = _readonly(
            self.contribution_count, np.uint16, "contribution_count", shape
        )
        if np.any(validity & ~np.isfinite(radial)):
            raise ValueError("valid radial-range samples must be finite")
        if np.any(~validity & ~np.isnan(radial)):
            raise ValueError("invalid radial-range samples must be NaN")
        for name, value in (
            ("confidence", confidence),
            ("matching_confidence", matching_confidence),
            ("consistency_confidence", consistency_confidence),
        ):
            if np.any(~np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must be finite and in [0, 1]")
            if np.any(value[~validity] != 0.0):
                raise ValueError(f"invalid {name} samples must be zero")
        if not np.allclose(
            confidence,
            matching_confidence * consistency_confidence,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "confidence must equal matching_confidence * consistency_confidence"
            )
        object.__setattr__(self, "radial_range", radial)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "matching_confidence", matching_confidence)
        object.__setattr__(self, "consistency_confidence", consistency_confidence)
        object.__setattr__(self, "disagreement", disagreement)
        object.__setattr__(self, "contribution_count", count)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    @property
    def range(self) -> np.ndarray:
        """Compatibility alias for radial range."""

        return self.radial_range


__all__ = [
    "PanoramicStereoPose",
    "PanoramicStereoResult",
    "RectifiedStereoViewSpec",
    "StereoPairDiagnostic",
    "StereoViewDescriptor",
    "StereoViewPairScore",
    "StereoViewSelectionDiagnostic",
    "StereoViewPlan",
]
