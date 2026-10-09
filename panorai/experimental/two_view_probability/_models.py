"""Portable probability models for the Experimental two-view workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from panorai.estimators import RelativePoseResult
from panorai.features import SphericalFeatureMatches

_BUNDLE_SHA256 = "4c481d6b84e9507c8656b1b57fb24359b3db30ff639839df5c510abcae9a7f62"
_OVERLAP_SHA256 = "e72a53d17c303361503777a38eb986132a2340ae2b86b7e24043557da71c7e60"


@dataclass(frozen=True, slots=True)
class BaselineEstimate:
    """Physical camera-centre distance and its one-sigma uncertainty, in metres."""

    mean_m: float
    standard_deviation_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean_m) or self.mean_m <= 0.0:
            raise ValueError("mean_m must be finite and positive")
        if (
            not math.isfinite(self.standard_deviation_m)
            or self.standard_deviation_m < 0.0
        ):
            raise ValueError("standard_deviation_m must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MatchEvidence:
    """Pre-pose observations available from two EQR images and their matches."""

    keypoint_count_min: int
    match_count: int
    descriptor_distance_median: float | None
    descriptor_distance_p90: float | None
    ratio_score_median: float | None
    valid_fraction_min: float


@dataclass(frozen=True, slots=True)
class OverlapPosterior:
    """Discrete posterior over spatial-overlap bins inferred without depth."""

    bin_edges: tuple[float, ...]
    bin_representatives: tuple[float, ...]
    probabilities: tuple[float, ...]
    expected_overlap: float
    probability_at_least_0_5: float
    inside_training_envelope: bool
    model_id: str
    model_sha256: str
    status: str


@dataclass(frozen=True, slots=True)
class CaptureAdvisory:
    """Advisory probability before interpreting the estimated pose."""

    p_accept: float
    p_precise_given_accept: float
    p_usable: float
    supported_probability_mass: float
    action: str
    status: str


@dataclass(frozen=True, slots=True)
class ExplicitCaptureProbabilities:
    """Capture probabilities for a measured overlap and physical baseline."""

    overlap: float
    baseline_m: float
    p_accept: float
    p_precise_given_accept: float
    p_usable: float
    inside_supported_envelope: bool


def evidence_from_matches(
    matches: SphericalFeatureMatches,
    *,
    keypoint_counts: tuple[int, int],
    valid_fractions: tuple[float, float] = (1.0, 1.0),
) -> MatchEvidence:
    """Summarize only observations available before R,t estimation."""

    if len(keypoint_counts) != 2 or min(keypoint_counts) < 0:
        raise ValueError("keypoint_counts must contain two non-negative counts")
    if len(valid_fractions) != 2 or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in valid_fractions
    ):
        raise ValueError("valid_fractions must contain two fractions in [0, 1]")
    valid = np.asarray(matches.valid, dtype=bool)
    distances = np.asarray(matches.descriptor_distances, dtype=np.float64)[valid]
    ratios = (
        None
        if matches.ratio_scores is None
        else np.asarray(matches.ratio_scores, dtype=np.float64)[valid]
    )
    return MatchEvidence(
        keypoint_count_min=int(min(keypoint_counts)),
        match_count=int(valid.sum()),
        descriptor_distance_median=(
            None if distances.size == 0 else float(np.median(distances))
        ),
        descriptor_distance_p90=(
            None if distances.size == 0 else float(np.quantile(distances, 0.9))
        ),
        ratio_score_median=(
            None if ratios is None or ratios.size == 0 else float(np.median(ratios))
        ),
        valid_fraction_min=float(min(valid_fractions)),
    )


def post_features_from_pose(
    pose: RelativePoseResult,
    *,
    match_count: int,
) -> dict[str, Any]:
    """Map a public pose result directly to the frozen post-model contract."""

    quality = pose.quality_report
    return {
        "raw_quality_score": quality.raw_quality_score,
        "match_count": int(match_count),
        "inlier_count": pose.num_inliers,
        "inlier_ratio": quality.inlier_ratio,
        "median_parallax_deg": quality.median_parallax_deg,
        "cheirality_ratio": quality.cheirality_ratio,
        "coverage_entropy_a": quality.coverage_entropy_a,
        "coverage_entropy_b": quality.coverage_entropy_b,
        "stability_translation_p90_deg": quality.stability.translation_p90_deg,
        "essential_score_margin": quality.model_competition.essential_score_margin,
        "translation_orientation_cheirality_margin": (
            quality.translation_orientation.cheirality_margin
        ),
        "translation_orientation_weighted_margin": (
            quality.translation_orientation.weighted_cheirality_margin
        ),
        "translation_orientation_median_triangulation_angle_deg": (
            quality.translation_orientation.median_triangulation_angle_deg
        ),
        "translation_orientation_ambiguous": (
            quality.translation_orientation.ambiguous
        ),
    }


def _sigmoid(value: float | np.ndarray) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -700.0, 700.0)))


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-4), 1.0 - 1e-4)
    return math.log(clipped / (1.0 - clipped))


class OverlapProxyModel:
    """Reviewed ordinal model from match evidence to an overlap posterior."""

    def __init__(
        self, artifact: Mapping[str, Any], *, sha256: str = "unverified"
    ) -> None:
        if artifact.get("schema") != "panorai-rgb-overlap-proxy/v1":
            raise ValueError("unsupported overlap-proxy schema")
        self._artifact = dict(artifact)
        self.sha256 = sha256

    @classmethod
    def load_default(cls) -> OverlapProxyModel:
        resource = files(__package__).joinpath("data/rgb-overlap-proxy-v1.json")
        payload = resource.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != _OVERLAP_SHA256:
            raise ValueError("overlap-proxy artifact SHA-256 mismatch")
        return cls(json.loads(payload), sha256=digest)

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self._artifact["metrics"]

    @property
    def data_provenance(self) -> Mapping[str, Any]:
        return self._artifact["data_provenance"]

    def score(self, evidence: MatchEvidence) -> OverlapPosterior:
        keypoints = max(float(evidence.keypoint_count_min), 0.0)
        matches = max(float(evidence.match_count), 0.0)
        optional = (
            evidence.descriptor_distance_median,
            evidence.descriptor_distance_p90,
            evidence.ratio_score_median,
        )
        raw = np.asarray(
            [
                math.log1p(keypoints),
                math.log1p(matches),
                _logit(matches / max(keypoints, 1.0)),
                *[math.nan if value is None else float(value) for value in optional],
                _logit(evidence.valid_fraction_min),
                *[1.0 if value is None else 0.0 for value in optional],
            ],
            dtype=np.float64,
        )
        medians = np.asarray(self._artifact["imputation_medians"], dtype=np.float64)
        raw = np.where(np.isnan(raw), medians, raw)
        mean = np.asarray(self._artifact["scaler_mean"], dtype=np.float64)
        scale = np.asarray(self._artifact["scaler_scale"], dtype=np.float64)
        beta = np.asarray(self._artifact["coefficients"], dtype=np.float64)
        cuts = np.asarray(self._artifact["cumulative_thresholds"], dtype=np.float64)
        temperature = float(self._artifact["calibration_temperature"])
        cumulative = _sigmoid(
            (cuts - float(((raw - mean) / scale) @ beta)) / temperature
        )
        probabilities = np.concatenate(
            (cumulative[:1], np.diff(cumulative), 1.0 - cumulative[-1:])
        )
        representatives = np.asarray(
            self._artifact["overlap_bin_representatives"], dtype=np.float64
        )
        ranges = self._artifact["supported_runtime_ranges"]
        inside = all(
            float(ranges[name][0]) <= raw[index] <= float(ranges[name][1])
            for index, name in enumerate(self._artifact["feature_names"][:7])
        )
        return OverlapPosterior(
            bin_edges=tuple(
                float(value) for value in self._artifact["overlap_bin_edges"]
            ),
            bin_representatives=tuple(float(value) for value in representatives),
            probabilities=tuple(float(value) for value in probabilities),
            expected_overlap=float(probabilities @ representatives),
            probability_at_least_0_5=float(probabilities[3:].sum()),
            inside_training_envelope=inside,
            model_id=str(self._artifact["model_id"]),
            model_sha256=self.sha256,
            status=str(self._artifact["status"]),
        )


class FrozenPoseProbabilityModels:
    """Portable scorer for the reviewed PanorAi 3.5.0 capture/post bundle."""

    def __init__(self, bundle: Mapping[str, Any], *, sha256: str) -> None:
        if bundle.get("schema") != "panorai-two-view-probability-model-bundle/v1":
            raise ValueError("unsupported pose-probability bundle schema")
        self.bundle = dict(bundle)
        self.models = {str(model["model_id"]): model for model in bundle["models"]}
        self.sha256 = sha256

    @classmethod
    def load_default(cls) -> FrozenPoseProbabilityModels:
        resource = files(__package__).joinpath("data/panorai-3.5.0-authorized-v1.json")
        payload = resource.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != _BUNDLE_SHA256:
            raise ValueError("pose-probability bundle SHA-256 mismatch")
        return cls(json.loads(payload), sha256=digest)

    @staticmethod
    def _transform(value: Any, transform: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("model feature must be finite")
        if transform == "identity":
            return number
        if transform == "log1p":
            if number < 0.0:
                raise ValueError("log1p model feature must be non-negative")
            return math.log1p(number)
        if transform == "logit":
            if not 0.0 <= number <= 1.0:
                raise ValueError("logit model feature must be in [0, 1]")
            return _logit(number)
        raise ValueError(f"unsupported transform: {transform}")

    def _score(self, model_id: str, flat: Mapping[str, Any]) -> float:
        model = self.models[model_id]
        transformed = np.asarray(
            [
                self._transform(
                    flat[str(feature["path"]).split(".")[-1]], feature["transform"]
                )
                for feature in model["features"]
            ]
        )
        standardized = (
            transformed - np.asarray(model["scaler_mean"], dtype=np.float64)
        ) / np.asarray(model["scaler_scale"], dtype=np.float64)
        parameters = np.asarray(model["parameters"], dtype=np.float64)
        raw_probability = float(_sigmoid(parameters[0] + standardized @ parameters[1:]))
        calibration_input = _logit(raw_probability)
        platt = model["platt_parameters"]
        return float(_sigmoid(float(platt[0]) + float(platt[1]) * calibration_input))

    def score_capture_explicit(
        self, *, overlap: float, baseline_m: float
    ) -> ExplicitCaptureProbabilities:
        """Score a measured registered-cloud overlap for offline analysis."""

        if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise ValueError("overlap must be finite and in [0, 1]")
        if not math.isfinite(baseline_m) or baseline_m <= 0.0:
            raise ValueError("baseline_m must be finite and positive")
        features = {
            "registered_cloud_overlap_min": overlap,
            "baseline_m": baseline_m,
        }
        p_accept = self._score("capture-accept-overlap-baseline", features)
        p_conditional = self._score(
            "capture-precise-given-accept-overlap-baseline", features
        )
        envelope = self.bundle["capture_envelope"]
        low, high = envelope["baseline_m_closed_interval"]
        return ExplicitCaptureProbabilities(
            overlap=overlap,
            baseline_m=baseline_m,
            p_accept=p_accept,
            p_precise_given_accept=p_conditional,
            p_usable=p_accept * p_conditional,
            inside_supported_envelope=(
                overlap >= envelope["registered_cloud_overlap_min"]
                and low <= baseline_m <= high
            ),
        )

    def score_post(self, features: Mapping[str, Any]) -> float:
        return self._score("post-precise-aligned-orientation", features)

    def capture_advisory(
        self,
        overlap: OverlapPosterior,
        baseline: BaselineEstimate,
    ) -> CaptureAdvisory:
        if baseline.standard_deviation_m == 0.0:
            baseline_nodes = np.asarray([baseline.mean_m])
            baseline_weights = np.asarray([1.0])
        else:
            nodes, weights = np.polynomial.hermite.hermgauss(11)
            baseline_nodes = (
                baseline.mean_m + math.sqrt(2.0) * baseline.standard_deviation_m * nodes
            )
            keep = baseline_nodes > 0.0
            baseline_nodes = baseline_nodes[keep]
            baseline_weights = weights[keep] / weights[keep].sum()
        representatives = np.asarray(overlap.bin_representatives)
        overlap_weights = np.asarray(overlap.probabilities)
        total_accept = total_conditional_joint = total_usable = support = 0.0
        envelope = self.bundle["capture_envelope"]
        low, high = envelope["baseline_m_closed_interval"]
        for overlap_value, overlap_weight in zip(
            representatives, overlap_weights, strict=True
        ):
            for baseline_value, baseline_weight in zip(
                baseline_nodes, baseline_weights, strict=True
            ):
                weight = float(overlap_weight * baseline_weight)
                point = self.score_capture_explicit(
                    overlap=float(overlap_value), baseline_m=float(baseline_value)
                )
                p_accept = point.p_accept
                p_conditional = point.p_precise_given_accept
                total_accept += weight * p_accept
                total_conditional_joint += weight * p_accept * p_conditional
                total_usable += weight * p_accept * p_conditional
                if (
                    overlap_value >= envelope["registered_cloud_overlap_min"]
                    and low <= baseline_value <= high
                ):
                    support += weight
        conditional = (
            total_conditional_joint / total_accept if total_accept > 0.0 else 0.0
        )
        capture_threshold = self.bundle["operating_thresholds"]["p_usable_capture_min"]
        if support < 0.95 or not overlap.inside_training_envelope:
            action = "unsupported"
        elif total_usable >= capture_threshold:
            action = "attempt"
        else:
            action = "recapture"
        return CaptureAdvisory(
            p_accept=total_accept,
            p_precise_given_accept=conditional,
            p_usable=total_usable,
            supported_probability_mass=support,
            action=action,
            status=str(self.bundle["status"]),
        )

    @property
    def post_threshold(self) -> float:
        return float(self.bundle["operating_thresholds"]["p_precise_post_min"])


def as_serializable(value: Any) -> dict[str, Any]:
    """Return dataclass evidence as a JSON-compatible dictionary."""

    return asdict(value)
