#!/usr/bin/env python3
"""Portable inference for the frozen spherical two-view probability models.

This module is a research deployment artifact.  It is deliberately kept out of
the stable ``panorai`` package API until prospective confirmation succeeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


BUNDLE_SCHEMA = "panorai-two-view-probability-model-bundle/v1"
BUNDLE_ID = "panorai-3.5.0-two-view-probability-authorized-v1"
CAPTURE_ACCEPT_MODEL = "capture-accept-overlap-baseline"
CAPTURE_PRECISE_MODEL = "capture-precise-given-accept-overlap-baseline"
POST_PRECISE_MODEL = "post-precise-aligned-orientation"
EXPECTED_MODEL_IDS = {
    CAPTURE_ACCEPT_MODEL,
    CAPTURE_PRECISE_MODEL,
    POST_PRECISE_MODEL,
}
DEFAULT_BUNDLE_PATH = (
    Path(__file__).with_name("model_artifacts")
    / "panorai-3.5.0-authorized-v1.json"
)
# Updated only when the reviewed, tracked payload changes.
DEFAULT_BUNDLE_SHA256 = (
    "4c481d6b84e9507c8656b1b57fb24359b3db30ff639839df5c510abcae9a7f62"
)


@dataclass(frozen=True)
class CaptureProbabilities:
    """Capture-geometry probabilities for one explicit-overlap pair."""

    registered_cloud_overlap_min: float
    baseline_m: float
    p_accept: float
    p_precise_given_accept: float
    p_usable: float
    inside_supported_envelope: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairDecision:
    """Complete capture/post probability result for one two-view estimate."""

    capture: CaptureProbabilities
    returned: bool
    public_quality_accepted: bool
    p_precise_post: float | None
    selected: bool
    reason: str
    model_status: str
    bundle_id: str
    bundle_sha256: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capture"] = self.capture.to_dict()
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _nested_value(row: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"required model feature is missing: {dotted_path}")
        value = value[part]
    return value


def _transform(value: Any, name: str, feature_path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{feature_path} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{feature_path} must be finite")
    if name == "identity":
        return number
    if name == "log1p":
        if number < 0.0:
            raise ValueError(f"{feature_path} must be non-negative")
        return math.log1p(number)
    if name == "logit":
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{feature_path} must be in [0, 1]")
        clipped = min(max(number, 1e-4), 1.0 - 1e-4)
        return math.log(clipped / (1.0 - clipped))
    raise ValueError(f"unsupported feature transform: {name}")


class TwoViewProbabilityModels:
    """Validated scorer for the frozen capture and post-processing models."""

    def __init__(
        self,
        bundle: Mapping[str, Any],
        *,
        bundle_sha256: str,
    ) -> None:
        if bundle.get("schema") != BUNDLE_SCHEMA:
            raise ValueError("unsupported probability-model bundle schema")
        if bundle.get("bundle_id") != BUNDLE_ID:
            raise ValueError("unsupported probability-model bundle identity")
        models: dict[str, Mapping[str, Any]] = {}
        for model in bundle.get("models", []):
            if not isinstance(model, Mapping):
                raise ValueError("model bundle contains a non-object model")
            model_id = str(model.get("model_id", ""))
            if not model_id or model_id in models:
                raise ValueError("model bundle contains a missing or duplicate model ID")
            self._validate_model(model)
            models[model_id] = model
        if set(models) != EXPECTED_MODEL_IDS:
            raise ValueError("model bundle does not contain the exact frozen model set")

        envelope = bundle.get("capture_envelope")
        thresholds = bundle.get("operating_thresholds")
        if not isinstance(envelope, Mapping) or not isinstance(thresholds, Mapping):
            raise ValueError("model bundle lacks its operating contract")
        baseline = envelope.get("baseline_m_closed_interval")
        if not isinstance(baseline, list) or len(baseline) != 2:
            raise ValueError("capture baseline interval is malformed")
        self._baseline_low = float(baseline[0])
        self._baseline_high = float(baseline[1])
        self._overlap_minimum = float(envelope["registered_cloud_overlap_min"])
        self._capture_threshold = float(thresholds["p_usable_capture_min"])
        self._post_threshold = float(thresholds["p_precise_post_min"])
        if not (
            0.0 <= self._overlap_minimum <= 1.0
            and 0.0 <= self._capture_threshold <= 1.0
            and 0.0 <= self._post_threshold <= 1.0
            and 0.0 <= self._baseline_low <= self._baseline_high
        ):
            raise ValueError("model bundle operating contract is invalid")

        self._bundle = dict(bundle)
        self._models = models
        self.bundle_sha256 = bundle_sha256
        self.bundle_id = str(bundle["bundle_id"])
        self.status = str(bundle["status"])

    @classmethod
    def load_default(cls) -> TwoViewProbabilityModels:
        """Load the reviewed PanorAi 3.5.0 model bundle and verify its hash."""

        return cls.load(DEFAULT_BUNDLE_PATH, expected_sha256=DEFAULT_BUNDLE_SHA256)

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str | None = None
    ) -> TwoViewProbabilityModels:
        """Load a JSON bundle, optionally requiring an exact SHA-256 identity."""

        bundle_path = Path(path)
        digest = _sha256(bundle_path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("probability-model bundle SHA-256 mismatch")
        value = json.loads(bundle_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("probability-model bundle must be one JSON object")
        return cls(value, bundle_sha256=digest)

    @staticmethod
    def _validate_model(model: Mapping[str, Any]) -> None:
        features = model.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("serialized model has no features")
        dimension = len(features)
        vectors = {
            "parameters": dimension + 1,
            "platt_parameters": 2,
            "scaler_mean": dimension,
            "scaler_scale": dimension,
        }
        for field, length in vectors.items():
            values = model.get(field)
            if not isinstance(values, list) or len(values) != length:
                raise ValueError(f"serialized model has invalid {field}")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"serialized model has non-finite {field}")
        if any(float(value) <= 0.0 for value in model["scaler_scale"]):
            raise ValueError("serialized model has a non-positive scaler scale")
        for feature in features:
            if not isinstance(feature, Mapping):
                raise ValueError("serialized model feature is not an object")
            if feature.get("transform") not in {"identity", "log1p", "logit"}:
                raise ValueError("serialized model uses an unsupported transform")
            if not str(feature.get("path", "")):
                raise ValueError("serialized model feature has no path")

    def _score(self, model_id: str, row: Mapping[str, Any]) -> float:
        model = self._models[model_id]
        transformed = [
            _transform(
                _nested_value(row, str(feature["path"])),
                str(feature["transform"]),
                str(feature["path"]),
            )
            for feature in model["features"]
        ]
        standardized = [
            (value - float(mean)) / float(scale)
            for value, mean, scale in zip(
                transformed,
                model["scaler_mean"],
                model["scaler_scale"],
                strict=True,
            )
        ]
        parameters = [float(value) for value in model["parameters"]]
        raw_logit = parameters[0] + sum(
            coefficient * value
            for coefficient, value in zip(
                parameters[1:], standardized, strict=True
            )
        )
        raw_probability = _sigmoid(raw_logit)
        clipped = min(max(raw_probability, 1e-6), 1.0 - 1e-6)
        calibration_input = math.log(clipped / (1.0 - clipped))
        platt = [float(value) for value in model["platt_parameters"]]
        probability = _sigmoid(platt[0] + platt[1] * calibration_input)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"model produced an invalid probability: {model_id}")
        return probability

    def score_capture(
        self, *, registered_cloud_overlap_min: float, baseline_m: float
    ) -> CaptureProbabilities:
        """Score explicit registered-cloud overlap and physical baseline.

        ``registered_cloud_overlap_min`` is a fraction in ``[0, 1]`` and
        ``baseline_m`` is expressed in metres.
        """

        overlap = float(registered_cloud_overlap_min)
        baseline = float(baseline_m)
        if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise ValueError("registered_cloud_overlap_min must be in [0, 1]")
        if not math.isfinite(baseline) or baseline < 0.0:
            raise ValueError("baseline_m must be a finite non-negative value")
        row = {
            "capture": {
                "registered_cloud_overlap_min": overlap,
                "baseline_m": baseline,
            }
        }
        p_accept = self._score(CAPTURE_ACCEPT_MODEL, row)
        p_conditional = self._score(CAPTURE_PRECISE_MODEL, row)
        return CaptureProbabilities(
            registered_cloud_overlap_min=overlap,
            baseline_m=baseline,
            p_accept=p_accept,
            p_precise_given_accept=p_conditional,
            p_usable=p_accept * p_conditional,
            inside_supported_envelope=(
                overlap >= self._overlap_minimum
                and self._baseline_low <= baseline <= self._baseline_high
            ),
        )

    def score_post(self, post_features: Mapping[str, Any]) -> float:
        """Score diagnostics for a pose that was actually returned."""

        nested = post_features.get("post")
        flat = nested if isinstance(nested, Mapping) else post_features
        ambiguous = flat.get("translation_orientation_ambiguous")
        if not isinstance(ambiguous, bool):
            raise TypeError("translation_orientation_ambiguous must be boolean")
        counts = {}
        for field in ("match_count", "inlier_count"):
            value = flat.get(field)
            if isinstance(value, bool):
                raise TypeError(f"{field} must be an integer count")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} must be an integer count") from error
            if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
                raise ValueError(f"{field} must be a non-negative integer count")
            counts[field] = int(numeric)
        if counts["inlier_count"] > counts["match_count"]:
            raise ValueError("inlier_count cannot exceed match_count")
        for field in ("coverage_entropy_a", "coverage_entropy_b"):
            try:
                value = float(flat.get(field))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} must be in [0, 1]") from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
        row = post_features if isinstance(nested, Mapping) else {"post": flat}
        return self._score(POST_PRECISE_MODEL, row)

    def decide(
        self,
        *,
        registered_cloud_overlap_min: float,
        baseline_m: float,
        returned: bool,
        public_quality_accepted: bool,
        post_features: Mapping[str, Any] | None,
    ) -> PairDecision:
        """Apply the complete frozen selective policy to one two-view pair."""

        if not isinstance(returned, bool) or not isinstance(
            public_quality_accepted, bool
        ):
            raise TypeError("returned and public_quality_accepted must be booleans")
        if public_quality_accepted and not returned:
            raise ValueError("a non-returned pose cannot be quality accepted")
        capture = self.score_capture(
            registered_cloud_overlap_min=registered_cloud_overlap_min,
            baseline_m=baseline_m,
        )
        p_post = None
        if returned:
            if post_features is None:
                raise ValueError("post_features are required for a returned pose")
            p_post = self.score_post(post_features)

        if not capture.inside_supported_envelope:
            selected, reason = False, "outside-supported-capture-envelope"
        elif not returned:
            selected, reason = False, "pose-not-returned"
        elif not public_quality_accepted:
            selected, reason = False, "public-quality-rejected"
        elif capture.p_usable < self._capture_threshold:
            selected, reason = False, "capture-probability-below-threshold"
        elif p_post is None or p_post < self._post_threshold:
            selected, reason = False, "post-probability-below-threshold"
        else:
            selected, reason = True, "selected"
        return PairDecision(
            capture=capture,
            returned=returned,
            public_quality_accepted=public_quality_accepted,
            p_precise_post=p_post,
            selected=selected,
            reason=reason,
            model_status=self.status,
            bundle_id=self.bundle_id,
            bundle_sha256=self.bundle_sha256,
        )

    def decide_from_pair_result(
        self,
        *,
        registered_cloud_overlap_min: float,
        baseline_m: float,
        pair_result: Mapping[str, Any],
        require_frozen_panorai_identity: bool = True,
    ) -> PairDecision:
        """Extract the exact aligned-route diagnostics and apply ``decide``."""

        if require_frozen_panorai_identity:
            package = pair_result.get("package")
            expected = self._bundle["panorai"]
            if not isinstance(package, Mapping):
                raise ValueError("pair result has no PanorAi package identity")
            if package.get("version") != expected["version"]:
                raise ValueError("pair result PanorAi version differs from the model")
            if package.get("expected_source_commit") != expected["source_commit"]:
                raise ValueError("pair result source commit differs from the model")
        pose = pair_result.get("pose")
        if not isinstance(pose, Mapping):
            raise ValueError("pair result has no pose object")
        returned = pose.get("returned")
        accepted = pose.get("quality_accepted")
        if not isinstance(returned, bool) or not isinstance(accepted, bool):
            raise ValueError("pair result has invalid pose decisions")
        post = extract_post_features(pair_result) if returned else None
        return self.decide(
            registered_cloud_overlap_min=registered_cloud_overlap_min,
            baseline_m=baseline_m,
            returned=returned,
            public_quality_accepted=accepted,
            post_features=post,
        )


def extract_post_features(pair_result: Mapping[str, Any]) -> dict[str, Any]:
    """Map a ``panorai-unified-optimized-pair/v2`` result to model features."""

    pose = pair_result.get("pose")
    counts = pair_result.get("counts")
    if not isinstance(pose, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("pair result lacks pose or count diagnostics")
    quality = pose.get("quality_report")
    if not isinstance(quality, Mapping):
        raise ValueError("returned pose lacks its quality report")
    stability = quality.get("stability")
    competition = quality.get("model_competition")
    orientation = quality.get("translation_orientation")
    if not all(
        isinstance(value, Mapping)
        for value in (stability, competition, orientation)
    ):
        raise ValueError("quality report lacks required nested diagnostics")
    return {
        "raw_quality_score": quality.get("raw_quality_score"),
        "match_count": counts.get("matches"),
        "inlier_count": pose.get("inlier_count"),
        "inlier_ratio": quality.get("inlier_ratio"),
        "median_parallax_deg": quality.get("median_parallax_deg"),
        "cheirality_ratio": quality.get("cheirality_ratio"),
        "coverage_entropy_a": quality.get("coverage_entropy_a"),
        "coverage_entropy_b": quality.get("coverage_entropy_b"),
        "stability_translation_p90_deg": stability.get("translation_p90_deg"),
        "essential_score_margin": competition.get("essential_score_margin"),
        "translation_orientation_cheirality_margin": orientation.get(
            "cheirality_margin"
        ),
        "translation_orientation_weighted_margin": orientation.get(
            "weighted_cheirality_margin"
        ),
        "translation_orientation_median_triangulation_angle_deg": orientation.get(
            "median_triangulation_angle_deg"
        ),
        "translation_orientation_ambiguous": orientation.get("ambiguous"),
    }


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="score capture geometry")
    capture.add_argument("--overlap", type=float, required=True)
    capture.add_argument("--baseline-m", type=float, required=True)
    pair = subparsers.add_parser("pair", help="score an optimized-pair result JSON")
    pair.add_argument("--overlap", type=float, required=True)
    pair.add_argument("--baseline-m", type=float, required=True)
    pair.add_argument("--result", type=Path, required=True)
    pair.add_argument(
        "--allow-unverified-panorai-identity",
        action="store_true",
        help="accept a result without the exact frozen package identity",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    expected_hash = DEFAULT_BUNDLE_SHA256 if args.bundle == DEFAULT_BUNDLE_PATH else None
    models = TwoViewProbabilityModels.load(
        args.bundle, expected_sha256=expected_hash
    )
    if args.command == "capture":
        result = models.score_capture(
            registered_cloud_overlap_min=args.overlap,
            baseline_m=args.baseline_m,
        ).to_dict()
    else:
        result = models.decide_from_pair_result(
            registered_cloud_overlap_min=args.overlap,
            baseline_m=args.baseline_m,
            pair_result=_read_object(args.result),
            require_frozen_panorai_identity=(
                not args.allow_unverified_panorai_identity
            ),
        ).to_dict()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
