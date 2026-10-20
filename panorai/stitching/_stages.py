"""Composable stages for central rotation-only panorama stitching."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from panorai.calibration import (
    PinholeIntrinsics,
    RotationCalibrationOptions,
    calibrate_rotation_pair,
    pinhole_rays_to_pixels,
)
from panorai.data import EquirectangularImage
from panorai.geometry import erp_pixels_to_rays
from panorai.model_fitting import RobustFitOptions, fit_homography
from panorai.pipeline import StageResult
from panorai.pipeline._protocols import stage_details

from ._types import (
    ImagePairMatches,
    ImagePairRegistration,
    PanoramaStitchingResult,
    RegisteredImageSet,
    StitchingDiagnostics,
    StitchingFeatureSet,
    StitchingFeatures,
    StitchingInput,
    StitchingMatches,
    StitchingOptions,
    StitchingRecoveryOptions,
    WarpedImageSet,
)


class StitchingRegistrationError(RuntimeError):
    """The image set cannot support the requested central panorama model."""


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - mandatory project dependency
        raise ImportError("panorama stitching requires OpenCV") from exc
    return cv2


def _feature_image(values: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    image = np.asarray(values)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[..., 0]
        else:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    if image.dtype == np.uint8:
        return image
    numeric = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(numeric)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)
    minimum = float(np.min(numeric[finite]))
    maximum = float(np.max(numeric[finite]))
    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = np.zeros(image.shape, dtype=np.float64)
    normalized[finite] = (numeric[finite] - minimum) * 255.0 / (maximum - minimum)
    return np.rint(normalized).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class DetectStitchingFeatures:
    """Extract planar local evidence while retaining source-image identity."""

    options: StitchingOptions = StitchingOptions()

    input_kind = "stitching-input"
    output_kind = "stitching-features"
    interface = "panorai-stitching-detect-features/v1-experimental"
    stability = "experimental"

    def execute(self, value: StitchingInput) -> StageResult:
        if not isinstance(value, StitchingInput):
            raise TypeError("DetectStitchingFeatures expects StitchingInput")
        cv2 = _cv2()
        if self.options.feature_method == "sift":
            extractor = cv2.SIFT_create(nfeatures=self.options.max_features)
        else:
            extractor = cv2.ORB_create(nfeatures=self.options.max_features)
        results = []
        for image in value.images:
            mask = np.where(image.support & image.validity, 255, 0).astype(np.uint8)
            keypoints, descriptors = extractor.detectAndCompute(
                _feature_image(image.values), mask
            )
            keypoints = [] if keypoints is None else list(keypoints)
            if descriptors is None:
                length = int(extractor.descriptorSize())
                dtype = (
                    np.float32 if self.options.feature_method == "sift" else np.uint8
                )
                descriptors = np.empty((0, length), dtype=dtype)
            results.append(
                StitchingFeatureSet(
                    image.image_id,
                    np.asarray(
                        [item.pt for item in keypoints], dtype=np.float64
                    ).reshape(-1, 2),
                    np.asarray(descriptors),
                    np.asarray([item.response for item in keypoints], dtype=np.float64),
                    self.options.feature_method,
                )
            )
        output = StitchingFeatures(value, tuple(results))
        return StageResult(
            output,
            stage_details(
                domain="planar-local-evidence",
                method=self.options.feature_method,
                feature_counts=tuple(len(item.pixels_xy) for item in results),
            ),
        )


def _ratio_matches(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    *,
    method: str,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv2 = _cv2()
    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    norm = cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING

    def candidates(left: np.ndarray, right: np.ndarray) -> dict[tuple[int, int], float]:
        matcher = cv2.BFMatcher(normType=norm, crossCheck=False)
        result: dict[tuple[int, int], float] = {}
        for pair in matcher.knnMatch(left, right, k=2):
            if len(pair) < 2:
                continue
            first, second = pair
            if second.distance > 0 and first.distance < ratio * second.distance:
                result[(int(first.queryIdx), int(first.trainIdx))] = float(
                    first.distance
                )
        return result

    forward = candidates(descriptors_a, descriptors_b)
    reverse = candidates(descriptors_b, descriptors_a)
    accepted = sorted(
        (left, right, distance)
        for (left, right), distance in forward.items()
        if (right, left) in reverse
    )
    if not accepted:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    left, right, distance = zip(*accepted, strict=True)
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(distance, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class DiscoverOverlaps:
    """Match every unordered image pair to discover an overlap graph."""

    options: StitchingOptions = StitchingOptions()

    input_kind = "stitching-features"
    output_kind = "stitching-matches"
    interface = "panorai-stitching-discover-overlaps/v1-experimental"
    stability = "experimental"

    def execute(self, value: StitchingFeatures) -> StageResult:
        if not isinstance(value, StitchingFeatures):
            raise TypeError("DiscoverOverlaps expects StitchingFeatures")
        pairs = []
        for index_a, index_b in itertools.combinations(range(len(value.features)), 2):
            left = value.features[index_a]
            right = value.features[index_b]
            a, b, distance = _ratio_matches(
                left.descriptors,
                right.descriptors,
                method=self.options.feature_method,
                ratio=self.options.ratio_test,
            )
            if len(a) >= self.options.min_pair_matches:
                pairs.append(ImagePairMatches(index_a, index_b, a, b, distance))
        output = StitchingMatches(value, tuple(pairs))
        return StageResult(
            output,
            stage_details(
                domain="descriptor-and-overlap-graph",
                candidate_pairs=len(pairs),
                possible_pairs=len(value.features) * (len(value.features) - 1) // 2,
            ),
        )


@dataclass(frozen=True, slots=True)
class DiscoverRobustOverlaps:
    """Discover a strong graph, then recover only weak missing pair evidence.

    The primary matcher is byte-for-byte the v1 mutual-ratio route.  A pair
    below that descriptor count may be reconsidered with the recovery ratio
    and its own minimum geometric consensus.  Feature density is checked per
    source lattice before recovery so low-texture image sets do not become
    acceptable merely because descriptor ambiguity was relaxed.
    """

    options: StitchingOptions = StitchingOptions()
    recovery: StitchingRecoveryOptions = StitchingRecoveryOptions()

    input_kind = "stitching-features"
    output_kind = "stitching-matches"
    interface = "panorai-stitching-discover-overlaps/v2-experimental"
    stability = "experimental"

    def execute(self, value: StitchingFeatures) -> StageResult:
        if not isinstance(value, StitchingFeatures):
            raise TypeError("DiscoverRobustOverlaps expects StitchingFeatures")
        if self.recovery.ratio_test < self.options.ratio_test:
            raise ValueError(
                "recovery ratio_test must not be stricter than the primary ratio"
            )
        densities = tuple(
            len(features.pixels_xy)
            / float(np.prod(value.source.images[index].shape_hw))
            for index, features in enumerate(value.features)
        )
        minimum_counts = tuple(
            min(
                self.recovery.feature_density_count_cap,
                int(
                    math.ceil(
                        self.recovery.min_feature_density
                        * float(np.prod(value.source.images[index].shape_hw))
                    )
                ),
            )
            for index in range(len(value.features))
        )
        sparse = [
            value.source.images[index].image_id
            for index, features in enumerate(value.features)
            if len(features.pixels_xy) < minimum_counts[index]
        ]
        if sparse:
            raise StitchingRegistrationError(
                "insufficient feature density for robust rotation registration: "
                + ", ".join(sparse)
            )

        pairs = []
        primary_count = 0
        recovery_count = 0
        for index_a, index_b in itertools.combinations(range(len(value.features)), 2):
            left = value.features[index_a]
            right = value.features[index_b]
            a, b, distance = _ratio_matches(
                left.descriptors,
                right.descriptors,
                method=self.options.feature_method,
                ratio=self.options.ratio_test,
            )
            if len(a) >= self.options.min_pair_matches:
                pairs.append(
                    ImagePairMatches(
                        index_a,
                        index_b,
                        a,
                        b,
                        distance,
                        self.options.min_pair_inliers,
                        "primary",
                    )
                )
                primary_count += 1
                continue
            a, b, distance = _ratio_matches(
                left.descriptors,
                right.descriptors,
                method=self.options.feature_method,
                ratio=self.recovery.ratio_test,
            )
            if len(a) >= self.recovery.min_pair_matches:
                pairs.append(
                    ImagePairMatches(
                        index_a,
                        index_b,
                        a,
                        b,
                        distance,
                        self.recovery.min_pair_inliers,
                        "recovery",
                    )
                )
                recovery_count += 1
        output = StitchingMatches(value, tuple(pairs))
        return StageResult(
            output,
            stage_details(
                domain="descriptor-and-observability-aware-overlap-graph",
                candidate_pairs=len(pairs),
                primary_pairs=primary_count,
                recovery_pairs=recovery_count,
                feature_densities=densities,
                minimum_feature_counts=minimum_counts,
                possible_pairs=len(value.features) * (len(value.features) - 1) // 2,
            ),
        )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    cross = _skew(vector)
    if angle < 1e-12:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return (
        np.eye(3)
        + math.sin(angle) / angle * cross
        + (1.0 - math.cos(angle)) / angle**2 * cross @ cross
    )


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if angle < 1e-10:
        return 0.5 * vee
    if math.pi - angle < 1e-6:
        values, vectors = np.linalg.eigh((rotation + np.eye(3)) / 2.0)
        axis = vectors[:, int(np.argmax(values))]
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * vee


def _initialize_rotations(
    count: int, edges: tuple[ImagePairRegistration, ...], reference: int
) -> list[np.ndarray]:
    rotations: dict[int, np.ndarray] = {reference: np.eye(3)}
    while len(rotations) < count:
        candidates = [
            edge
            for edge in edges
            if (edge.index_a in rotations) != (edge.index_b in rotations)
        ]
        if not candidates:
            missing = sorted(set(range(count)) - set(rotations))
            raise StitchingRegistrationError(
                "overlap graph is disconnected; unregistered image indices: "
                + ", ".join(map(str, missing))
            )
        edge = min(
            candidates, key=lambda item: (-item.num_inliers, item.index_a, item.index_b)
        )
        if edge.index_a in rotations:
            rotations[edge.index_b] = rotations[edge.index_a] @ edge.rotation_b_from_a.T
        else:
            rotations[edge.index_a] = rotations[edge.index_b] @ edge.rotation_b_from_a
    return [rotations[index] for index in range(count)]


def _average_rotations(
    initial: list[np.ndarray],
    edges: tuple[ImagePairRegistration, ...],
    reference: int,
) -> tuple[list[np.ndarray], float, float]:
    from scipy.optimize import least_squares  # type: ignore[import-untyped]

    variable = [index for index in range(len(initial)) if index != reference]
    offsets = {index: 3 * position for position, index in enumerate(variable)}

    def unpack(parameters: np.ndarray) -> list[np.ndarray]:
        result = [rotation.copy() for rotation in initial]
        result[reference] = initial[reference]
        for index in variable:
            start = offsets[index]
            result[index] = (
                _rotation_exp(parameters[start : start + 3]) @ initial[index]
            )
        return result

    median = float(np.median([edge.num_inliers for edge in edges]))

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotations = unpack(parameters)
        values = []
        for edge in edges:
            predicted = rotations[edge.index_b].T @ rotations[edge.index_a]
            error = _rotation_log(edge.rotation_b_from_a.T @ predicted)
            values.extend(math.sqrt(edge.num_inliers / max(median, 1.0)) * error)
        return np.asarray(values, dtype=np.float64)

    start = np.zeros(3 * len(variable), dtype=np.float64)
    initial_cost = float(np.mean(residual(start) ** 2))
    solved = least_squares(
        residual,
        start,
        loss="cauchy",
        f_scale=math.radians(1.0),
        max_nfev=200,
    )
    final = unpack(solved.x)
    final_cost = float(np.mean(residual(solved.x) ** 2))
    return final, initial_cost, final_cost


@dataclass(frozen=True, slots=True)
class CalibrateAndRegister:
    """Fit pair models, calibrate FOV and optimize one connected SO(3) graph."""

    options: StitchingOptions = StitchingOptions()

    input_kind = "stitching-matches"
    output_kind = "registered-images"
    interface = "panorai-stitching-calibrate-register/v1-experimental"
    stability = "experimental"

    def execute(self, value: StitchingMatches) -> StageResult:
        if not isinstance(value, StitchingMatches):
            raise TypeError("CalibrateAndRegister expects StitchingMatches")
        registrations = []
        rejected = []
        feature_sets = value.features.features
        images = value.features.source.images
        for pair in value.pairs:
            pixels_a = feature_sets[pair.index_a].pixels_xy[pair.feature_indices_a]
            pixels_b = feature_sets[pair.index_b].pixels_xy[pair.feature_indices_b]
            label = f"{images[pair.index_a].image_id}<->{images[pair.index_b].image_id}"
            if pair.matching_tier == "recovery":
                label += " [recovery]"
            minimum_inliers = (
                self.options.min_pair_inliers
                if pair.minimum_inliers is None
                else pair.minimum_inliers
            )
            try:
                homography = fit_homography(
                    pixels_a,
                    pixels_b,
                    options=RobustFitOptions(
                        residual_threshold=self.options.max_transfer_error_px,
                        min_trials=32,
                        max_trials=1000,
                        min_inliers=minimum_inliers,
                        random_seed=self.options.random_seed
                        + pair.index_a * 1009
                        + pair.index_b,
                    ),
                )
                selected = np.flatnonzero(homography.inlier_mask)
                calibration = calibrate_rotation_pair(
                    pixels_a[selected],
                    pixels_b[selected],
                    images[pair.index_a].shape_hw,
                    images[pair.index_b].shape_hw,
                    horizontal_fov_a_deg=images[pair.index_a].horizontal_fov_deg,
                    horizontal_fov_b_deg=images[pair.index_b].horizontal_fov_deg,
                    options=RotationCalibrationOptions(
                        min_horizontal_fov_deg=self.options.min_horizontal_fov_deg,
                        max_horizontal_fov_deg=self.options.max_horizontal_fov_deg,
                        max_angular_error_deg=self.options.max_angular_error_deg,
                        max_median_angular_error_deg=(
                            self.options.max_median_angular_error_deg
                        ),
                        min_inliers=minimum_inliers,
                        random_seed=self.options.random_seed
                        + pair.index_a * 1009
                        + pair.index_b,
                        min_spatial_coverage=self.options.min_spatial_coverage,
                    ),
                )
                if not calibration.observable:
                    raise ValueError(
                        "unobservable calibration: "
                        + ",".join(calibration.degeneracy_reasons)
                    )
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                rejected.append(f"{label}: {exc}")
                continue
            registrations.append(
                ImagePairRegistration(
                    pair.index_a,
                    pair.index_b,
                    calibration.rotation_b_from_a,
                    calibration,
                    selected,
                    homography.model,
                    homography.num_inliers,
                    pair.matching_tier,
                )
            )
        edges = tuple(registrations)
        if not edges:
            raise StitchingRegistrationError(
                "no image pair passed homography, calibration and rotation gates"
            )
        count = len(images)
        degrees = np.zeros(count, dtype=np.int64)
        for edge in edges:
            degrees[edge.index_a] += edge.num_inliers
            degrees[edge.index_b] += edge.num_inliers
        reference = int(np.argmax(degrees))
        initial = _initialize_rotations(count, edges, reference)
        rotations, initial_cost, final_cost = _average_rotations(
            initial, edges, reference
        )

        intrinsics: list[PinholeIntrinsics | None] = [None] * count
        candidates: list[list[PinholeIntrinsics]] = [[] for _ in range(count)]
        for edge in edges:
            candidates[edge.index_a].append(edge.calibration.intrinsics_a)
            candidates[edge.index_b].append(edge.calibration.intrinsics_b)
        for index, options in enumerate(candidates):
            if not options:
                raise StitchingRegistrationError(
                    f"image {index} has no accepted calibration"
                )
            fov = float(np.median([item.horizontal_fov_deg for item in options]))
            intrinsics[index] = PinholeIntrinsics.from_horizontal_fov(
                images[index].shape_hw, fov
            )
        output = RegisteredImageSet(
            value.features.source,
            tuple(item for item in intrinsics if item is not None),
            tuple(rotations),
            edges,
            tuple(len(item.pixels_xy) for item in feature_sets),
            reference,
            initial_cost,
            final_cost,
            tuple(rejected),
        )
        return StageResult(
            output,
            stage_details(
                domain="pinhole-calibration-and-so3-registration",
                accepted_pairs=len(edges),
                rejected_pairs=len(rejected),
                reference_image=images[reference].image_id,
                rotation_initial_cost=initial_cost,
                rotation_final_cost=final_cost,
            ),
        )


def _remap_mask(
    mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: str,
) -> np.ndarray:
    cv2 = _cv2()
    if interpolation == "nearest":
        return cv2.remap(
            np.asarray(mask, dtype=np.uint8),
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    coverage = cv2.remap(
        np.asarray(mask, dtype=np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Strict propagation: a bilinear output is valid only when every source
    # neighbour with non-zero interpolation weight is valid and supported.
    return coverage >= 1.0 - 1e-6


@dataclass(frozen=True, slots=True)
class WarpRegisteredImages:
    """Inverse-warp registered pinhole sources onto one canonical ERP grid."""

    options: StitchingOptions = StitchingOptions()

    input_kind = "registered-images"
    output_kind = "warped-image-set"
    interface = "panorai-stitching-warp/v1-experimental"
    stability = "experimental"

    def execute(self, value: RegisteredImageSet) -> StageResult:
        if not isinstance(value, RegisteredImageSet):
            raise TypeError("WarpRegisteredImages expects RegisteredImageSet")
        cv2 = _cv2()
        height, width = self.options.output_shape_hw
        rows, columns = np.indices((height, width), dtype=np.float64)
        world_rays = erp_pixels_to_rays(
            np.stack((columns, rows), axis=-1), (height, width)
        )
        values = []
        support = []
        validity = []
        weights = []
        interpolation = (
            cv2.INTER_LINEAR
            if self.options.interpolation == "bilinear"
            else cv2.INTER_NEAREST
        )
        for image, intrinsics, camera_to_world in zip(
            value.source.images, value.intrinsics, value.camera_to_world, strict=True
        ):
            camera_rays = world_rays @ camera_to_world
            pixels, positive = pinhole_rays_to_pixels(camera_rays, intrinsics)
            map_x = pixels[..., 0].astype(np.float32)
            map_y = pixels[..., 1].astype(np.float32)
            source_height, source_width = image.shape_hw
            geometric = (
                positive
                & (map_x >= -0.5)
                & (map_x <= source_width - 0.5)
                & (map_y >= -0.5)
                & (map_y <= source_height - 0.5)
            )
            warped = cv2.remap(
                np.asarray(image.values),
                map_x,
                map_y,
                interpolation=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(np.float64)
            warped_support = geometric & _remap_mask(
                image.support,
                map_x,
                map_y,
                interpolation=self.options.interpolation,
            )
            warped_validity = warped_support & _remap_mask(
                image.validity,
                map_x,
                map_y,
                interpolation=self.options.interpolation,
            )
            weight = np.where(
                warped_validity, np.clip(camera_rays[..., 2], 0.0, 1.0) ** 4, 0.0
            )
            values.append(warped)
            support.append(warped_support)
            validity.append(warped_validity)
            weights.append(weight)
        output = WarpedImageSet(
            value,
            tuple(values),
            tuple(support),
            tuple(validity),
            tuple(weights),
            (1.0,) * len(values),
        )
        return StageResult(
            output,
            stage_details(
                domain="inverse-pinhole-to-erp",
                output_shape_hw=self.options.output_shape_hw,
                interpolation=self.options.interpolation,
            ),
        )


def _luminance(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 or image.shape[2] == 1:
        return image if image.ndim == 2 else image[..., 0]
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


@dataclass(frozen=True, slots=True)
class CompensateExposure:
    """Estimate robust global gains from valid ERP overlaps."""

    options: StitchingOptions = StitchingOptions()

    input_kind = "warped-image-set"
    output_kind = "warped-image-set"
    interface = "panorai-stitching-exposure/v1-experimental"
    stability = "experimental"

    def execute(self, value: WarpedImageSet) -> StageResult:
        if not isinstance(value, WarpedImageSet):
            raise TypeError("CompensateExposure expects WarpedImageSet")
        count = len(value.values)
        if self.options.exposure_compensation == "none":
            return StageResult(value, stage_details(domain="photometric", mode="none"))
        equations = []
        targets = []
        for left, right in itertools.combinations(range(count), 2):
            overlap = value.validity[left] & value.validity[right]
            if np.count_nonzero(overlap) < 32:
                continue
            first = _luminance(value.values[left])[overlap]
            second = _luminance(value.values[right])[overlap]
            usable = (
                (first > 1e-6)
                & (second > 1e-6)
                & np.isfinite(first)
                & np.isfinite(second)
            )
            if np.count_nonzero(usable) < 32:
                continue
            ratio = float(np.median(np.log(first[usable]) - np.log(second[usable])))
            row = np.zeros(count - 1, dtype=np.float64)
            reference = value.registered.reference_index
            if left != reference:
                row[left - (left > reference)] = -1.0
            if right != reference:
                row[right - (right > reference)] = 1.0
            equations.append(row)
            targets.append(ratio)
        if equations:
            logarithms, *_ = np.linalg.lstsq(
                np.stack(equations), np.asarray(targets), rcond=None
            )
        else:
            logarithms = np.zeros(count - 1, dtype=np.float64)
        gains = np.ones(count, dtype=np.float64)
        reference = value.registered.reference_index
        for index in range(count):
            if index != reference:
                gains[index] = math.exp(float(logarithms[index - (index > reference)]))
        gains = np.clip(gains, 0.25, 4.0)
        images = tuple(
            image * gain for image, gain in zip(value.values, gains, strict=True)
        )
        output = WarpedImageSet(
            value.registered,
            images,
            value.support,
            value.validity,
            value.weights,
            tuple(float(item) for item in gains),
        )
        return StageResult(
            output,
            stage_details(domain="photometric", mode="gain", gains=tuple(gains)),
        )


@dataclass(frozen=True, slots=True)
class BlendStitchingPanorama:
    """Blend valid warped samples and return a typed ERP plus diagnostics."""

    options: StitchingOptions = StitchingOptions()

    input_kind = "warped-image-set"
    output_kind = "panorama-stitching-result"
    interface = "panorai-stitching-blend/v1-experimental"
    stability = "experimental"

    def execute(self, value: WarpedImageSet) -> StageResult:
        if not isinstance(value, WarpedImageSet):
            raise TypeError("BlendStitchingPanorama expects WarpedImageSet")
        shape = value.values[0].shape
        numerator = np.zeros(shape, dtype=np.float64)
        denominator = np.zeros(shape[:2], dtype=np.float64)
        active_weights = []
        for image, validity, feather in zip(
            value.values, value.validity, value.weights, strict=True
        ):
            weight = (
                validity.astype(np.float64)
                if self.options.blend == "average"
                else feather
            )
            active_weights.append(weight)
            numerator += np.where(
                validity[..., None] if image.ndim == 3 else validity, image, 0.0
            ) * (weight[..., None] if image.ndim == 3 else weight)
            denominator += weight
        valid = denominator > 0.0
        output_values = np.zeros_like(numerator)
        divisor = denominator[..., None] if numerator.ndim == 3 else denominator
        np.divide(numerator, divisor, out=output_values, where=divisor > 0.0)
        source_dtype = value.registered.source.images[0].values.dtype
        if np.issubdtype(source_dtype, np.integer):
            bounds = np.iinfo(source_dtype)
            output_values = np.rint(
                np.clip(output_values, bounds.min, bounds.max)
            ).astype(source_dtype)
        else:
            output_values = output_values.astype(source_dtype)
        support = np.logical_or.reduce(np.stack(value.support), axis=0)
        support_count = np.sum(np.stack(value.support), axis=0, dtype=np.int32)
        stacked_weights = np.stack(active_weights)
        contribution = np.argmax(stacked_weights, axis=0).astype(np.int32)
        contribution[~valid] = -1
        panorama = EquirectangularImage(
            output_values,
            support_mask=support,
            valid=valid,
        )
        diagnostics = StitchingDiagnostics(
            input_image_count=len(value.registered.source.images),
            feature_counts=value.registered.feature_counts,
            candidate_pair_count=len(value.registered.pairwise)
            + len(value.registered.rejected_pairs),
            accepted_pair_count=len(value.registered.pairwise),
            rejected_pairs=value.registered.rejected_pairs,
            reference_image_id=value.registered.source.images[
                value.registered.reference_index
            ].image_id,
            estimated_fov_count=sum(
                image.horizontal_fov_deg is None
                for image in value.registered.source.images
            ),
            rotation_initial_cost=value.registered.rotation_initial_cost,
            rotation_final_cost=value.registered.rotation_final_cost,
            coverage=float(np.mean(valid)),
            overlap_fraction=float(np.mean(support_count >= 2)),
            gains=value.gains,
            recovery_pair_count=sum(
                edge.matching_tier == "recovery" for edge in value.registered.pairwise
            ),
        )
        result = PanoramaStitchingResult(
            panorama,
            value.registered,
            contribution,
            support_count,
            diagnostics,
        )
        return StageResult(
            result,
            stage_details(
                domain="erp-fusion",
                blend=self.options.blend,
                coverage=diagnostics.coverage,
                overlap_fraction=diagnostics.overlap_fraction,
            ),
        )


__all__ = [
    "BlendStitchingPanorama",
    "CalibrateAndRegister",
    "CompensateExposure",
    "DetectStitchingFeatures",
    "DiscoverOverlaps",
    "DiscoverRobustOverlaps",
    "StitchingRegistrationError",
    "WarpRegisteredImages",
]
