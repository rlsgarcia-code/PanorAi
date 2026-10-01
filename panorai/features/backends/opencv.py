"""OpenCV implementation adapter for PanorAi's spherical feature facade."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .._config import FeatureExtractorConfig, FeatureMatcherConfig


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - clean optional environment
        raise ImportError(
            "Spherical features require OpenCV; install PanorAi with "
            "`pip install panorai[features]`."
        ) from exc
    return cv2


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(item) for item in parts[:3])


class OpenCVFeatureBackend:
    """Encapsulate OpenCV detectors, descriptors, and matchers.

    Injected OpenCV-compatible objects are an advanced route. The normal API
    never returns ``cv2.KeyPoint`` or ``cv2.DMatch`` instances.
    """

    name = "opencv"

    def __init__(self, *, extractor: Any | None = None, matcher: Any | None = None):
        self._extractor = extractor
        self._matcher = matcher

    @property
    def version(self) -> str:
        return str(_cv2().__version__)

    def require_version(self, minimum: str) -> None:
        if _version_tuple(self.version) < _version_tuple(minimum):
            raise RuntimeError(f"OpenCV>={minimum} is required; found {self.version}")

    def create_extractor(self, config: FeatureExtractorConfig):
        if self._extractor is not None:
            if not hasattr(self._extractor, "detectAndCompute"):
                raise TypeError("injected extractor must provide detectAndCompute")
            return self._extractor
        cv2 = _cv2()
        parameters = config.parameter_dict
        if config.method == "sift":
            defaults = {
                "nfeatures": config.max_features,
                "contrastThreshold": config.contrast_threshold,
                "edgeThreshold": config.edge_threshold,
            }
            defaults.update(parameters)
            return cv2.SIFT_create(**defaults)
        if config.method == "orb":
            defaults = {"nfeatures": config.max_features}
            defaults.update(parameters)
            return cv2.ORB_create(**defaults)
        descriptor_type = (
            cv2.AKAZE_DESCRIPTOR_MLDB
            if config.akaze_descriptor_type == "binary"
            else cv2.AKAZE_DESCRIPTOR_KAZE
        )
        defaults = {"descriptor_type": descriptor_type}
        defaults.update(parameters)
        return cv2.AKAZE_create(**defaults)

    def descriptor_metadata(
        self, config: FeatureExtractorConfig, extractor: Any
    ) -> dict[str, Any]:
        cv2 = _cv2()
        method = _extractor_method(extractor) or config.method
        length = int(extractor.descriptorSize())
        descriptor_type = int(extractor.descriptorType())
        floating = descriptor_type in {cv2.CV_32F, cv2.CV_64F}
        if method == "orb" and hasattr(extractor, "getWTA_K"):
            metric = "hamming2" if int(extractor.getWTA_K()) > 2 else "hamming"
        else:
            metric = "l2" if floating else "hamming"
        return {
            "type": f"{method}-float32" if floating else f"{method}-binary",
            "metric": metric,
            "length": length,
            "extractor_name": method,
        }

    def detect_and_describe(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: FeatureExtractorConfig,
    ) -> dict[str, Any]:
        extractor = self.create_extractor(config)
        metadata = self.descriptor_metadata(config, extractor)
        keypoints, descriptors = extractor.detectAndCompute(image, mask)
        keypoints = [] if keypoints is None else list(keypoints)
        if descriptors is None:
            dtype = (
                np.uint8
                if metadata["metric"] in {"hamming", "hamming2"}
                else np.float32
            )
            descriptors = np.empty((0, metadata["length"]), dtype=dtype)
        descriptors = np.asarray(descriptors)
        if len(keypoints) != descriptors.shape[0]:
            raise RuntimeError("OpenCV returned misaligned keypoints and descriptors")
        return {
            "pixels_xy": np.asarray(
                [item.pt for item in keypoints], dtype=np.float64
            ).reshape(-1, 2),
            "responses": np.asarray(
                [item.response for item in keypoints], dtype=np.float32
            ),
            "scales": np.asarray([item.size for item in keypoints], dtype=np.float32),
            "angles_deg": np.asarray(
                [item.angle for item in keypoints], dtype=np.float32
            ),
            "octaves": np.asarray([item.octave for item in keypoints], dtype=np.int32),
            "descriptors": descriptors,
            "metadata": metadata,
        }

    def create_matcher(
        self,
        descriptor_metric: str,
        config: FeatureMatcherConfig,
    ) -> tuple[Any, str]:
        method = config.method
        if method == "auto":
            method = "flann" if descriptor_metric == "l2" else "bf"
        if self._matcher is not None:
            if not (
                hasattr(self._matcher, "match") or hasattr(self._matcher, "knnMatch")
            ):
                raise TypeError("injected matcher must provide match or knnMatch")
            return self._matcher, "injected"
        cv2 = _cv2()
        if method == "bf":
            norm = {
                "l2": cv2.NORM_L2,
                "hamming": cv2.NORM_HAMMING,
                "hamming2": cv2.NORM_HAMMING2,
            }.get(descriptor_metric)
            if norm is None:
                raise ValueError(f"unsupported descriptor metric: {descriptor_metric}")
            return cv2.BFMatcher(normType=norm, crossCheck=False), method
        if descriptor_metric == "hamming2":
            raise ValueError("FLANN is not supported for Hamming2 descriptors")
        parameters = config.parameter_dict
        if descriptor_metric == "l2":
            index = {"algorithm": 1, "trees": int(parameters.get("trees", 5))}
        else:
            index = {
                "algorithm": 6,
                "table_number": int(parameters.get("table_number", 12)),
                "key_size": int(parameters.get("key_size", 20)),
                "multi_probe_level": int(parameters.get("multi_probe_level", 2)),
            }
        search = {"checks": int(parameters.get("checks", 50))}
        return cv2.FlannBasedMatcher(index, search), method

    def match(
        self,
        descriptors_a: np.ndarray,
        descriptors_b: np.ndarray,
        descriptor_metric: str,
        config: FeatureMatcherConfig,
    ) -> tuple[list[dict[str, Any]], str]:
        left = np.asarray(descriptors_a)
        right = np.asarray(descriptors_b)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
            raise ValueError("descriptor arrays must have compatible shape (N, D)")
        matcher, method = self.create_matcher(descriptor_metric, config)
        if left.shape[0] == 0 or right.shape[0] == 0:
            return [], method
        matching_method = method
        if method == "injected":
            matching_method = "flann" if descriptor_metric == "l2" else "bf"
        if matching_method == "flann" and descriptor_metric == "l2":
            left = left.astype(np.float32, copy=False)
            right = right.astype(np.float32, copy=False)
        elif descriptor_metric in {"hamming", "hamming2"}:
            left = left.astype(np.uint8, copy=False)
            right = right.astype(np.uint8, copy=False)

        forward = self._candidate_matches(matcher, left, right, config)
        if config.cross_check:
            reverse_matcher, _ = self.create_matcher(descriptor_metric, config)
            reverse = self._candidate_matches(reverse_matcher, right, left, config)
            reverse_pairs = {(item["query_idx"], item["train_idx"]) for item in reverse}
            for item in forward:
                item["mutual"] = (item["train_idx"], item["query_idx"]) in reverse_pairs
            forward = [item for item in forward if item["mutual"]]
        else:
            for item in forward:
                item["mutual"] = None
        return forward, method

    @staticmethod
    def _candidate_matches(
        matcher: Any,
        descriptors_a: np.ndarray,
        descriptors_b: np.ndarray,
        config: FeatureMatcherConfig,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if config.ratio_test is not None:
            if descriptors_b.shape[0] < 2:
                return candidates
            pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
            for pair in pairs:
                if len(pair) < 2:
                    continue
                best, second = pair[0], pair[1]
                best_distance = float(best.distance)
                second_distance = float(second.distance)
                if not (
                    np.isfinite(best_distance)
                    and np.isfinite(second_distance)
                    and second_distance > 0.0
                    and best_distance < config.ratio_test * second_distance
                ):
                    continue
                ratio = best_distance / second_distance
                if (
                    config.max_distance is not None
                    and best_distance > config.max_distance
                ):
                    continue
                candidates.append(
                    {
                        "query_idx": int(best.queryIdx),
                        "train_idx": int(best.trainIdx),
                        "distance": best_distance,
                        "ratio_score": ratio,
                    }
                )
        else:
            for match in matcher.match(descriptors_a, descriptors_b):
                if (
                    config.max_distance is not None
                    and float(match.distance) > config.max_distance
                ):
                    continue
                candidates.append(
                    {
                        "query_idx": int(match.queryIdx),
                        "train_idx": int(match.trainIdx),
                        "distance": float(match.distance),
                        "ratio_score": None,
                    }
                )
        return candidates


def _extractor_method(extractor: Any) -> str | None:
    if not hasattr(extractor, "getDefaultName"):
        return None
    name = str(extractor.getDefaultName()).lower()
    for method in ("sift", "orb", "akaze"):
        if name.endswith(method) or f".{method}" in name:
            return method
    return None
