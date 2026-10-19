"""Typed, JSON-safe results for spherical evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


INTERFACE = "panorai-spherical-evaluation/v1-experimental"


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only finite JSON values") from exc
    return json.loads(encoded)


def _mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    copied = _json_copy(dict(value), name=name)
    return MappingProxyType(copied)


def _metric_mapping(
    value: Mapping[str, float | None], *, name: str
) -> Mapping[str, float | None]:
    result: dict[str, float | None] = {}
    for key, item in value.items():
        metric = str(key).strip()
        if not metric:
            raise ValueError(f"{name} names must be non-empty")
        if item is None:
            result[metric] = None
            continue
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{metric!r}] must be finite or None")
        result[metric] = number
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class BinaryRegionEvaluation:
    """Solid-angle and boundary comparison for two binary ERP regions."""

    iou: float
    dice: float
    precision: float
    recall: float
    coverage: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    supported_solid_angle_sr: float
    comparable_solid_angle_sr: float
    prediction_solid_angle_sr: float
    reference_solid_angle_sr: float
    intersection_solid_angle_sr: float
    union_solid_angle_sr: float
    boundary_tolerance_deg: float
    interface: str = INTERFACE

    @property
    def metrics(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "iou": self.iou,
                "dice": self.dice,
                "precision": self.precision,
                "recall": self.recall,
                "coverage": self.coverage,
                "boundary_precision": self.boundary_precision,
                "boundary_recall": self.boundary_recall,
                "boundary_f1": self.boundary_f1,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureEvaluation:
    """One-to-one angular repeatability result for spherical keypoints."""

    predicted_count: int
    reference_count: int
    matched_count: int
    precision: float
    recall: float
    repeatability: float
    median_angular_error_deg: float | None
    maximum_angular_error_deg: float | None
    median_scale_error_levels: float | None
    maximum_scale_error_levels: float | None
    threshold_deg: float
    matched_indices: tuple[tuple[int, int], ...]
    interface: str = INTERFACE

    @property
    def metrics(self) -> Mapping[str, float | None]:
        return _metric_mapping(
            {
                "precision": self.precision,
                "recall": self.recall,
                "repeatability": self.repeatability,
                "median_angular_error_deg": self.median_angular_error_deg,
                "maximum_angular_error_deg": self.maximum_angular_error_deg,
                "median_scale_error_levels": self.median_scale_error_levels,
                "maximum_scale_error_levels": self.maximum_scale_error_levels,
            },
            name="feature metrics",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PoseEvaluation:
    """SO(3) and optional translation-direction error."""

    rotation_error_deg: float
    translation_direction_error_deg: float | None
    translation_sign_invariant: bool
    interface: str = INTERFACE

    @property
    def metrics(self) -> Mapping[str, float | None]:
        return _metric_mapping(
            {
                "rotation_error_deg": self.rotation_error_deg,
                "translation_direction_error_deg": (
                    self.translation_direction_error_deg
                ),
            },
            name="pose metrics",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DepthEvaluation:
    """Area-weighted radial-range accuracy, coverage and confidence evidence."""

    coverage: float
    mae: float | None
    rmse: float | None
    abs_rel: float | None
    delta1: float | None
    delta2: float | None
    delta3: float | None
    relative_coverage: float
    confidence_error_spearman: float | None
    top_confidence_quartile_mae: float | None
    bottom_confidence_quartile_mae: float | None
    comparable_count: int
    supported_count: int
    relative_count: int
    comparable_solid_angle_sr: float
    supported_solid_angle_sr: float
    units: str
    interface: str = INTERFACE

    @property
    def metrics(self) -> Mapping[str, float | None]:
        return _metric_mapping(
            {
                "coverage": self.coverage,
                "mae": self.mae,
                "rmse": self.rmse,
                "abs_rel": self.abs_rel,
                "delta1": self.delta1,
                "delta2": self.delta2,
                "delta3": self.delta3,
                "relative_coverage": self.relative_coverage,
                "confidence_error_spearman": self.confidence_error_spearman,
                "top_confidence_quartile_mae": self.top_confidence_quartile_mae,
                "bottom_confidence_quartile_mae": (self.bottom_confidence_quartile_mae),
            },
            name="depth metrics",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Frozen evaluator identity and parameters selected before evaluation."""

    protocol_id: str
    version: int
    evaluator: str
    metric_names: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    interface: str = INTERFACE

    def __post_init__(self) -> None:
        for name in ("protocol_id", "evaluator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an integer")
        if self.version < 1:
            raise ValueError("version must be positive")
        names = tuple(str(name).strip() for name in self.metric_names)
        if (
            not names
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("metric_names must be unique non-empty strings")
        object.__setattr__(self, "metric_names", names)
        object.__setattr__(
            self, "parameters", _mapping(self.parameters, name="parameters")
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "protocol_id": self.protocol_id,
            "version": self.version,
            "evaluator": self.evaluator,
            "metric_names": list(self.metric_names),
            "parameters": _json_copy(dict(self.parameters), name="parameters"),
        }


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One JSON-safe result bound to data, algorithm, protocol and artifact."""

    protocol: EvaluationProtocol
    subject_id: str
    input_id: str
    reference_id: str
    split: str
    algorithm_id: str
    algorithm_config: Mapping[str, Any]
    package_version: str
    backend: str
    metrics: Mapping[str, float | None]
    status: str = "evaluated"
    reasons: tuple[str, ...] = ()
    artifact_sha256: str | None = None
    geometry_contract: str = "geometry-v1"
    environment: Mapping[str, Any] = field(default_factory=dict)
    interface: str = INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol")
        for name in (
            "subject_id",
            "input_id",
            "reference_id",
            "split",
            "algorithm_id",
            "package_version",
            "backend",
            "geometry_contract",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.status not in {"evaluated", "abstained", "failed"}:
            raise ValueError("status must be 'evaluated', 'abstained', or 'failed'")
        reasons = tuple(str(reason).strip() for reason in self.reasons)
        if any(not reason for reason in reasons):
            raise ValueError("reasons must contain non-empty strings")
        metrics = _metric_mapping(self.metrics, name="metrics")
        if set(metrics) != set(self.protocol.metric_names):
            raise ValueError("metrics must exactly match protocol.metric_names")
        if self.status == "evaluated" and reasons:
            raise ValueError("evaluated records cannot contain failure reasons")
        if self.status != "evaluated" and not reasons:
            raise ValueError("abstained and failed records require reasons")
        if self.artifact_sha256 is not None:
            digest = self.artifact_sha256.lower()
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("artifact_sha256 must be one lowercase SHA-256 digest")
            object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "algorithm_config",
            _mapping(self.algorithm_config, name="algorithm_config"),
        )
        object.__setattr__(
            self, "environment", _mapping(self.environment, name="environment")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "protocol": self.protocol.to_dict(),
            "protocol_fingerprint": self.protocol.fingerprint,
            "subject_id": self.subject_id,
            "input_id": self.input_id,
            "reference_id": self.reference_id,
            "split": self.split,
            "algorithm_id": self.algorithm_id,
            "algorithm_config": _json_copy(
                dict(self.algorithm_config), name="algorithm_config"
            ),
            "package_version": self.package_version,
            "artifact_sha256": self.artifact_sha256,
            "backend": self.backend,
            "geometry_contract": self.geometry_contract,
            "status": self.status,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "environment": _json_copy(dict(self.environment), name="environment"),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvaluationRecord":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("payload must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("payload must contain one JSON object")
        protocol_value = value.get("protocol")
        if not isinstance(protocol_value, dict):
            raise ValueError("payload.protocol must be an object")
        protocol = EvaluationProtocol(
            protocol_id=protocol_value.get("protocol_id"),
            version=protocol_value.get("version"),
            evaluator=protocol_value.get("evaluator"),
            metric_names=tuple(protocol_value.get("metric_names", ())),
            parameters=protocol_value.get("parameters", {}),
        )
        expected = value.get("protocol_fingerprint")
        if expected != protocol.fingerprint:
            raise ValueError("protocol_fingerprint does not match protocol")
        return cls(
            protocol=protocol,
            subject_id=value.get("subject_id"),
            input_id=value.get("input_id"),
            reference_id=value.get("reference_id"),
            split=value.get("split"),
            algorithm_id=value.get("algorithm_id"),
            algorithm_config=value.get("algorithm_config", {}),
            package_version=value.get("package_version"),
            artifact_sha256=value.get("artifact_sha256"),
            backend=value.get("backend"),
            geometry_contract=value.get("geometry_contract"),
            status=value.get("status"),
            reasons=tuple(value.get("reasons", ())),
            metrics=value.get("metrics", {}),
            environment=value.get("environment", {}),
        )
