"""Serializable configuration contracts for spherical feature workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping


FEATURES_INTERFACE = "panorai-spherical-features/v1"
FEATURES_STABILITY = "stable"


def _method(value: str, choices: set[str], name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return normalized


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    invalid = result < 0 if allow_zero else result <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _finite(value: float, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _angular_threshold_degrees(value: float, name: str) -> float:
    result = _finite(value, name, minimum=0.0)
    if result > 180.0:
        raise ValueError(f"{name} must be <= 180")
    return result


def _minimum_opencv_version(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("minimum_opencv_version must be a string")
    normalized = value.strip()
    if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", normalized) is None:
        raise ValueError("minimum_opencv_version must use numeric X.Y or X.Y.Z form")
    parts = tuple(int(item) for item in normalized.split("."))
    padded = parts + (0,) * (3 - len(parts))
    if padded < (4, 9, 0):
        raise ValueError("minimum_opencv_version must be at least 4.9.0")
    return normalized


def _parameter_items(value: Any) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(sorted((str(key), item) for key, item in items))
    for key, item in normalized:
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise TypeError(
                f"algorithm parameter {key!r} must be JSON-serializable scalar"
            )
    return normalized


@dataclass(frozen=True, slots=True)
class FeatureExtractorConfig:
    """Declarative OpenCV feature-extractor configuration."""

    method: str = "sift"
    max_features: int = 4096
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    edge_margin_px: int = 16
    deduplicate_overlaps: bool = True
    angular_dedup_threshold_deg: float = 0.15
    akaze_descriptor_type: str = "binary"
    parameters: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    validity_margin_px: int = 0
    validity_scale_margin: float = 0.0
    max_features_per_face: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "method", _method(self.method, {"sift", "orb", "akaze"}, "method")
        )
        object.__setattr__(
            self, "max_features", _positive_int(self.max_features, "max_features")
        )
        if self.max_features_per_face is not None:
            object.__setattr__(
                self,
                "max_features_per_face",
                _positive_int(self.max_features_per_face, "max_features_per_face"),
            )
        object.__setattr__(
            self,
            "contrast_threshold",
            _finite(self.contrast_threshold, "contrast_threshold", minimum=0.0),
        )
        object.__setattr__(
            self,
            "edge_threshold",
            _finite(self.edge_threshold, "edge_threshold", minimum=0.0),
        )
        object.__setattr__(
            self,
            "edge_margin_px",
            _positive_int(self.edge_margin_px, "edge_margin_px", allow_zero=True),
        )
        object.__setattr__(
            self,
            "validity_margin_px",
            _positive_int(
                self.validity_margin_px, "validity_margin_px", allow_zero=True
            ),
        )
        object.__setattr__(
            self,
            "validity_scale_margin",
            _finite(
                self.validity_scale_margin,
                "validity_scale_margin",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "angular_dedup_threshold_deg",
            _angular_threshold_degrees(
                self.angular_dedup_threshold_deg,
                "angular_dedup_threshold_deg",
            ),
        )
        object.__setattr__(
            self,
            "deduplicate_overlaps",
            _boolean(self.deduplicate_overlaps, "deduplicate_overlaps"),
        )
        object.__setattr__(
            self,
            "akaze_descriptor_type",
            _method(
                self.akaze_descriptor_type, {"binary", "float"}, "akaze_descriptor_type"
            ),
        )
        object.__setattr__(self, "parameters", _parameter_items(self.parameters))

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return dict(self.parameters)

    @property
    def effective_max_features_per_face(self) -> int:
        return (
            self.max_features
            if self.max_features_per_face is None
            else self.max_features_per_face
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameters"] = self.parameter_dict
        return result


@dataclass(frozen=True, slots=True)
class FeatureMatcherConfig:
    """Declarative OpenCV descriptor-matcher configuration."""

    method: str = "auto"
    ratio_test: float | None = 0.75
    cross_check: bool = False
    max_distance: float | None = None
    deduplicate_matches: bool = True
    angular_dedup_threshold_deg: float = 0.15
    parameters: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "method", _method(self.method, {"auto", "bf", "flann"}, "method")
        )
        if self.ratio_test is not None:
            ratio = _finite(self.ratio_test, "ratio_test")
            if not 0.0 < ratio < 1.0:
                raise ValueError("ratio_test must be in the open interval (0, 1)")
            object.__setattr__(self, "ratio_test", ratio)
        if self.max_distance is not None:
            object.__setattr__(
                self,
                "max_distance",
                _finite(self.max_distance, "max_distance", minimum=0.0),
            )
        object.__setattr__(
            self, "cross_check", _boolean(self.cross_check, "cross_check")
        )
        object.__setattr__(
            self,
            "deduplicate_matches",
            _boolean(self.deduplicate_matches, "deduplicate_matches"),
        )
        object.__setattr__(
            self,
            "angular_dedup_threshold_deg",
            _angular_threshold_degrees(
                self.angular_dedup_threshold_deg,
                "angular_dedup_threshold_deg",
            ),
        )
        object.__setattr__(self, "parameters", _parameter_items(self.parameters))

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return dict(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameters"] = self.parameter_dict
        return result


@dataclass(frozen=True, slots=True)
class FaceSetSpec:
    """Serializable materialization contract for virtual gnomonic cameras."""

    sampler: str = "icosahedron"
    shape_hw: tuple[int, int] = (1024, 1024)
    fov_deg: tuple[float, float] = (80.0, 80.0)
    overlap_deg: float = 0.0
    count: int | None = None
    subdivisions: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sampler, str) or not self.sampler.strip():
            raise TypeError("sampler must be a non-empty string")
        object.__setattr__(self, "sampler", self.sampler.strip().lower())
        try:
            shape = tuple(self.shape_hw)
        except TypeError as exc:
            raise TypeError("shape_hw must contain (height, width)") from exc
        if len(shape) != 2:
            raise ValueError("shape_hw must contain exactly (height, width)")
        shape = tuple(_positive_int(item, "shape_hw value") for item in shape)
        object.__setattr__(self, "shape_hw", shape)
        try:
            fov = tuple(self.fov_deg)
        except TypeError as exc:
            raise TypeError("fov_deg must contain (horizontal, vertical)") from exc
        if len(fov) != 2:
            raise ValueError("fov_deg must contain exactly (horizontal, vertical)")
        fov = tuple(_finite(item, "fov_deg") for item in fov)
        overlap = _finite(self.overlap_deg, "overlap_deg", minimum=0.0)
        effective = tuple(item + overlap for item in fov)
        if any(not 0.0 < item < 180.0 for item in effective):
            raise ValueError("fov_deg + overlap_deg must stay between 0 and 180")
        object.__setattr__(self, "fov_deg", fov)
        object.__setattr__(self, "overlap_deg", overlap)
        if self.count is not None:
            object.__setattr__(self, "count", _positive_int(self.count, "count"))
        object.__setattr__(
            self,
            "subdivisions",
            _positive_int(self.subdivisions, "subdivisions", allow_zero=True),
        )

    @property
    def effective_fov_deg(self) -> tuple[float, float]:
        """Return nominal FOV plus the declared overlap expansion."""

        return tuple(value + self.overlap_deg for value in self.fov_deg)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["effective_fov_deg"] = self.effective_fov_deg
        return result


@dataclass(frozen=True, slots=True)
class SphericalFeaturePipelineConfig:
    """Complete reproducible configuration for the high-level facade."""

    extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    matcher: FeatureMatcherConfig = field(default_factory=FeatureMatcherConfig)
    face_set: FaceSetSpec = field(default_factory=FaceSetSpec)
    interface: str = FEATURES_INTERFACE
    preset_name: str | None = None
    preset_version: int = 1
    minimum_opencv_version: str = "4.9.0"

    def __post_init__(self) -> None:
        if self.interface != FEATURES_INTERFACE:
            raise ValueError("interface must be 'panorai-spherical-features/v1'")
        object.__setattr__(
            self, "preset_version", _positive_int(self.preset_version, "preset_version")
        )
        object.__setattr__(
            self,
            "minimum_opencv_version",
            _minimum_opencv_version(self.minimum_opencv_version),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": FEATURES_STABILITY,
            "preset_name": self.preset_name,
            "preset_version": self.preset_version,
            "minimum_opencv_version": self.minimum_opencv_version,
            "extractor": self.extractor.to_dict(),
            "matcher": self.matcher.to_dict(),
            "face_set": self.face_set.to_dict(),
        }

    def replace(self, **changes: Any) -> "SphericalFeaturePipelineConfig":
        return replace(self, **changes)
