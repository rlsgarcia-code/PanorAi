"""Descriptor-neutral tangent patches for spherical keypoints.

The detector owns only bearing, response, and angular scale.  A
``TangentPatchRequest`` declares the visual context required by a downstream
consumer, while ``TangentPatchProvider`` delegates every projection and mask
operation to :mod:`panorai.geometry`.  Descriptor adapters consume the
materialized patches without knowing how the keypoints were detected.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from panorai.geometry import GnomonicProjector, GnomonicSpec, gnomonic_face_geometry

from ._config import FeatureExtractorConfig
from ._extractor import _array_checksum, _as_panorama, _opencv_image
from .backends.opencv import OpenCVFeatureBackend


TANGENT_PATCH_INTERFACE = "panorai-tangent-patches/v1"
TANGENT_OPENCV_DESCRIPTOR_INTERFACE = "panorai-tangent-opencv-descriptor/v1"
TANGENT_OPENCV_DESCRIPTOR_V2_INTERFACE = "panorai-tangent-opencv-descriptor/v2"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_real(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class TangentPatchRequest:
    """Descriptor-neutral request for one patch around every keypoint.

    ``radius_in_scales`` is the angular half-extent divided by the detector's
    scale sigma.  Consequently, the requested full FOV is
    ``2 * radius_in_scales * scale_deg``.  ``output_shape_hw`` independently
    controls sampling density.  No descriptor-specific window size is hidden
    in the provider.

    ``orientation_policy='upright'`` keeps north/up aligned with the patch.
    ``'per-keypoint-roll'`` requires one explicit roll angle per keypoint.
    """

    output_shape_hw: tuple[int, int] = (48, 48)
    radius_in_scales: float = 6.0
    minimum_fov_deg: float = 1.0
    maximum_fov_deg: float = 120.0
    interpolation: str = "bilinear"
    invalid_policy: str = "propagate"
    min_valid_weight: float | None = None
    minimum_valid_fraction: float = 1.0
    orientation_policy: str = "upright"
    fixed_roll_deg: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_shape_hw, tuple)
            or len(self.output_shape_hw) != 2
        ):
            raise TypeError("output_shape_hw must be a two-integer tuple")
        object.__setattr__(
            self,
            "output_shape_hw",
            tuple(
                _positive_integer(item, "output_shape_hw item")
                for item in self.output_shape_hw
            ),
        )
        object.__setattr__(
            self,
            "radius_in_scales",
            _finite_real(self.radius_in_scales, "radius_in_scales", minimum=0.25),
        )
        minimum_fov = _finite_real(
            self.minimum_fov_deg, "minimum_fov_deg", minimum=1e-6, maximum=179.0
        )
        maximum_fov = _finite_real(
            self.maximum_fov_deg, "maximum_fov_deg", minimum=1e-6, maximum=179.0
        )
        if minimum_fov > maximum_fov:
            raise ValueError("minimum_fov_deg must not exceed maximum_fov_deg")
        object.__setattr__(self, "minimum_fov_deg", minimum_fov)
        object.__setattr__(self, "maximum_fov_deg", maximum_fov)
        interpolation = str(self.interpolation).strip().lower()
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        object.__setattr__(self, "interpolation", interpolation)
        invalid_policy = str(self.invalid_policy).strip().lower()
        if invalid_policy not in {"propagate", "renormalize"}:
            raise ValueError("invalid_policy must be 'propagate' or 'renormalize'")
        object.__setattr__(self, "invalid_policy", invalid_policy)
        if self.min_valid_weight is not None:
            threshold = _finite_real(
                self.min_valid_weight,
                "min_valid_weight",
                minimum=0.0,
                maximum=1.0,
            )
            if invalid_policy != "renormalize":
                raise ValueError(
                    "min_valid_weight is only valid with invalid_policy='renormalize'"
                )
            object.__setattr__(self, "min_valid_weight", threshold)
        object.__setattr__(
            self,
            "minimum_valid_fraction",
            _finite_real(
                self.minimum_valid_fraction,
                "minimum_valid_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        policy = str(self.orientation_policy).strip().lower()
        if policy not in {"upright", "per-keypoint-roll"}:
            raise ValueError(
                "orientation_policy must be 'upright' or 'per-keypoint-roll'"
            )
        object.__setattr__(self, "orientation_policy", policy)
        object.__setattr__(
            self,
            "fixed_roll_deg",
            _finite_real(self.fixed_roll_deg, "fixed_roll_deg"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "interface": TANGENT_PATCH_INTERFACE,
                "projection": "gnomonic",
                "scale_units": "degrees-sigma",
                "fov_rule": "2 * radius_in_scales * detector_scale_deg",
                "validity_interpolation": "nearest",
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class TangentPatchGeometry:
    """Geometry and sampling metadata for a materialized tangent patch."""

    patch_id: str
    keypoint_index: int
    source_erp_xy: np.ndarray
    center_bearing_xyz: np.ndarray
    detector_scale_deg: float
    requested_radius_deg: float
    actual_radius_deg: float
    samples_per_scale_xy: tuple[float, float]
    projection_spec: GnomonicSpec
    K: np.ndarray
    R_panorama_from_patch: np.ndarray
    source_frame: str = "panorama-camera"
    patch_frame: str = "gnomonic-camera:+x-right,+y-down,+z-forward"


@dataclass(frozen=True, slots=True)
class TangentPatch:
    """One projected patch, with support and observed-data validity separated."""

    geometry: TangentPatchGeometry
    image: np.ndarray
    support_mask: np.ndarray
    validity_mask: np.ndarray
    valid_weight: np.ndarray | None
    valid_fraction: float
    valid: bool
    source_panorama_checksum: str
    interface: str = TANGENT_PATCH_INTERFACE


@dataclass(frozen=True, slots=True)
class TangentPatchSet:
    """Ordered tangent patches aligned to a descriptor-free keypoint set."""

    panorama_id: str
    source_shape_hw: tuple[int, int]
    source_checksum: str
    keypoint_interface: str
    request: TangentPatchRequest
    patches: tuple[TangentPatch, ...]
    interface: str = TANGENT_PATCH_INTERFACE
    stability: str = "experimental"

    def __len__(self) -> int:
        return len(self.patches)

    @property
    def valid_count(self) -> int:
        return sum(item.valid for item in self.patches)


class TangentPatchProvider:
    """Materialize descriptor-neutral patches through public PanorAi geometry."""

    def __init__(self, *, max_workers: int = 1) -> None:
        """Create a provider with deterministic ordered parallel materialization.

        ``max_workers`` changes execution only. Every worker owns its projector
        and patch objects, and :meth:`materialize` preserves input ordering.
        The default remains one worker for backward-compatible resource use.
        """

        self.max_workers = _positive_integer(max_workers, "max_workers")

    def materialize(
        self,
        panorama: Any,
        keypoints: Any,
        request: TangentPatchRequest,
        *,
        validity_mask: np.ndarray | None = None,
        rolls_deg: np.ndarray | None = None,
    ) -> TangentPatchSet:
        wrapped = _as_panorama(panorama)
        if not isinstance(wrapped.image, np.ndarray):
            raise TypeError("tangent patches currently require a NumPy panorama")
        image = wrapped.image
        if image.ndim not in {2, 3} or image.shape[0] < 2 or image.shape[1] < 4:
            raise ValueError("panorama must use HW/HWC layout and be at least 2x4")
        checksum = _array_checksum(image)
        if tuple(keypoints.source_shape_hw) != tuple(image.shape[:2]):
            raise ValueError("keypoints and panorama must have the same source shape")
        if keypoints.source_checksum != checksum:
            raise ValueError("keypoints were not detected on this panorama")
        validity = np.asarray(wrapped.validity("image"), dtype=bool)
        if wrapped.support_mask is not None:
            validity &= np.asarray(wrapped.support_mask, dtype=bool)
        if validity_mask is not None:
            if (
                not isinstance(validity_mask, np.ndarray)
                or validity_mask.dtype != np.bool_
            ):
                raise TypeError("validity_mask must be a boolean NumPy array")
            if validity_mask.shape != image.shape[:2]:
                raise ValueError(
                    f"validity_mask must have shape {image.shape[:2]}; got {validity_mask.shape}"
                )
            validity &= validity_mask
        rolls = self._resolve_rolls(request, len(keypoints.keypoints), rolls_deg)

        def materialize_index(index: int) -> TangentPatch:
            return self._materialize_one(
                image,
                validity,
                keypoints.keypoints[index],
                index,
                float(rolls[index]),
                request,
                checksum,
            )

        count = len(keypoints.keypoints)
        if self.max_workers == 1 or count < 2:
            patches = tuple(materialize_index(index) for index in range(count))
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, count),
                thread_name_prefix="panorai-tangent-patch",
            ) as executor:
                patches = tuple(executor.map(materialize_index, range(count)))
        return TangentPatchSet(
            panorama_id=str(keypoints.panorama_id),
            source_shape_hw=tuple(image.shape[:2]),
            source_checksum=checksum,
            keypoint_interface=str(keypoints.interface),
            request=request,
            patches=patches,
        )

    @staticmethod
    def _resolve_rolls(
        request: TangentPatchRequest, count: int, rolls_deg: np.ndarray | None
    ) -> np.ndarray:
        if request.orientation_policy == "upright":
            if rolls_deg is not None:
                raise ValueError("rolls_deg is only valid for per-keypoint-roll")
            return np.full(count, request.fixed_roll_deg, dtype=np.float64)
        if rolls_deg is None:
            raise ValueError("rolls_deg is required for per-keypoint-roll")
        values = np.asarray(rolls_deg, dtype=np.float64).reshape(-1)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError("rolls_deg must contain one finite value per keypoint")
        return values + request.fixed_roll_deg

    @staticmethod
    def _materialize_one(
        image: np.ndarray,
        validity: np.ndarray,
        keypoint: Any,
        index: int,
        roll_deg: float,
        request: TangentPatchRequest,
        checksum: str,
    ) -> TangentPatch:
        height, width = image.shape[:2]
        source_xy = np.asarray(keypoint.source_erp_xy, dtype=np.float64)
        bearing = np.asarray(keypoint.bearing_xyz, dtype=np.float64)
        scale_deg = _finite_real(
            keypoint.scale_deg, "keypoint scale_deg", minimum=1e-12
        )
        requested_radius = request.radius_in_scales * scale_deg
        fov_deg = min(
            request.maximum_fov_deg,
            max(request.minimum_fov_deg, 2.0 * requested_radius),
        )
        lon_deg = (source_xy[0] + 0.5) / width * 360.0 - 180.0
        lat_deg = 90.0 - (source_xy[1] + 0.5) / height * 180.0
        spec = GnomonicSpec(
            center_lat_deg=float(lat_deg),
            center_lon_deg=float(lon_deg),
            hfov_deg=float(fov_deg),
            vfov_deg=float(fov_deg),
            roll_deg=float(roll_deg),
            output_shape_hw=request.output_shape_hw,
        )
        image_projector = GnomonicProjector(
            spec,
            interpolation=request.interpolation,
            invalid_policy=request.invalid_policy,
            min_valid_weight=request.min_valid_weight,
        )
        projected = image_projector.project(
            image.astype(np.float32, copy=False),
            validity_mask=validity if request.invalid_policy == "renormalize" else None,
        )
        validity_projector = GnomonicProjector(spec, interpolation="nearest")
        projected_validity = validity_projector.project(validity).data.astype(bool)
        support = np.asarray(projected.support_mask, dtype=bool)
        if projected.validity_mask is not None:
            projected_validity &= np.asarray(projected.validity_mask, dtype=bool)
        projected_validity &= support
        support_count = int(np.count_nonzero(support))
        valid_fraction = (
            float(np.count_nonzero(projected_validity)) / support_count
            if support_count
            else 0.0
        )
        face = gnomonic_face_geometry(f"tangent-keypoint-{index:06d}", spec, support)
        actual_radius = 0.5 * fov_deg
        rows, columns = request.output_shape_hw
        samples_per_scale = (
            columns / max(2.0 * actual_radius / scale_deg, 1e-12),
            rows / max(2.0 * actual_radius / scale_deg, 1e-12),
        )
        geometry = TangentPatchGeometry(
            patch_id=face.face_id,
            keypoint_index=index,
            source_erp_xy=source_xy.copy(),
            center_bearing_xyz=bearing.copy(),
            detector_scale_deg=scale_deg,
            requested_radius_deg=float(requested_radius),
            actual_radius_deg=float(actual_radius),
            samples_per_scale_xy=(
                float(samples_per_scale[0]),
                float(samples_per_scale[1]),
            ),
            projection_spec=spec,
            K=np.asarray(face.K).copy(),
            R_panorama_from_patch=np.asarray(face.R_panorama_from_face).copy(),
        )
        return TangentPatch(
            geometry=geometry,
            image=np.asarray(projected.data),
            support_mask=support,
            validity_mask=projected_validity,
            valid_weight=(
                None
                if projected.valid_weight is None
                else np.asarray(projected.valid_weight).copy()
            ),
            valid_fraction=valid_fraction,
            valid=valid_fraction >= request.minimum_valid_fraction,
            source_panorama_checksum=checksum,
        )


@dataclass(frozen=True, slots=True)
class OpenCVTangentDescriptorConfig:
    """Descriptor semantics applied to already-materialized tangent patches."""

    method: str = "sift"
    keypoint_diameter_in_scales: float = 3.0
    orientation_policy: str = "dominant-gradient"
    orientation_bins: int = 36
    root_sift: bool = False
    algorithm_parameters: tuple[tuple[str, Any], ...] = ()
    minimum_opencv_version: str = "4.9.0"

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in {"sift", "orb", "akaze"}:
            raise ValueError("method must be 'sift', 'orb', or 'akaze'")
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "keypoint_diameter_in_scales",
            _finite_real(
                self.keypoint_diameter_in_scales,
                "keypoint_diameter_in_scales",
                minimum=0.1,
            ),
        )
        policy = str(self.orientation_policy).strip().lower()
        if policy not in {"dominant-gradient", "fixed-zero"}:
            raise ValueError(
                "orientation_policy must be 'dominant-gradient' or 'fixed-zero'"
            )
        object.__setattr__(self, "orientation_policy", policy)
        object.__setattr__(
            self,
            "orientation_bins",
            _positive_integer(self.orientation_bins, "orientation_bins"),
        )
        if not isinstance(self.root_sift, bool):
            raise TypeError("root_sift must be a boolean")
        if self.root_sift and method != "sift":
            raise ValueError("root_sift is only valid for method='sift'")
        parameters = tuple(
            sorted((str(key), value) for key, value in self.algorithm_parameters)
        )
        object.__setattr__(self, "algorithm_parameters", parameters)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "interface": TANGENT_OPENCV_DESCRIPTOR_INTERFACE,
                "patch_interface": TANGENT_PATCH_INTERFACE,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class TangentDescriptorSet:
    """Descriptors and metadata aligned to a subset of a tangent patch set."""

    patch_indices: np.ndarray
    descriptors: np.ndarray
    angles_deg: np.ndarray
    descriptor_type: str
    descriptor_metric: str
    extractor_name: str
    config: OpenCVTangentDescriptorConfig
    patch_interface: str = TANGENT_PATCH_INTERFACE
    interface: str = TANGENT_OPENCV_DESCRIPTOR_INTERFACE
    stability: str = "experimental"

    def __len__(self) -> int:
        return int(self.patch_indices.shape[0])


@dataclass(frozen=True, slots=True)
class OpenCVTangentDescriptorV2Config:
    """Experimental descriptor hypotheses over descriptor-neutral patches.

    The v2 adapter leaves patch geometry untouched and makes every additional
    source of invariance explicit. Defaults use the calibrated single-
    hypothesis RootSIFT profile consumed by the canonical spherical extractor.
    """

    method: str = "sift"
    keypoint_diameter_in_scales: float = 1.25
    scale_multipliers: tuple[float, ...] = (1.0,)
    orientation_policy: str = "fixed-zero"
    orientation_bins: int = 36
    orientation_peak_ratio: float = 0.8
    max_orientations: int = 2
    photometric_normalization: str = "local-standardization"
    minimum_descriptor_valid_fraction: float = 0.99
    root_sift: bool = True
    algorithm_parameters: tuple[tuple[str, Any], ...] = ()
    minimum_opencv_version: str = "4.9.0"

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in {"sift", "orb"}:
            raise ValueError("v2 method must be 'sift' or 'orb'")
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "keypoint_diameter_in_scales",
            _finite_real(
                self.keypoint_diameter_in_scales,
                "keypoint_diameter_in_scales",
                minimum=0.1,
            ),
        )
        multipliers = tuple(
            _finite_real(item, "scale multiplier", minimum=0.1)
            for item in self.scale_multipliers
        )
        if not multipliers:
            raise ValueError("scale_multipliers must not be empty")
        if len(set(multipliers)) != len(multipliers):
            raise ValueError("scale_multipliers must be unique")
        object.__setattr__(self, "scale_multipliers", multipliers)
        policy = str(self.orientation_policy).strip().lower()
        if policy not in {"fixed-zero", "dominant-gradient", "multi-peak-gradient"}:
            raise ValueError(
                "orientation_policy must be fixed-zero, dominant-gradient, "
                "or multi-peak-gradient"
            )
        object.__setattr__(self, "orientation_policy", policy)
        object.__setattr__(
            self,
            "orientation_bins",
            _positive_integer(self.orientation_bins, "orientation_bins"),
        )
        object.__setattr__(
            self,
            "orientation_peak_ratio",
            _finite_real(
                self.orientation_peak_ratio,
                "orientation_peak_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "max_orientations",
            _positive_integer(self.max_orientations, "max_orientations"),
        )
        normalization = str(self.photometric_normalization).strip().lower()
        if normalization not in {
            "none",
            "local-standardization",
            "robust-percentile-2-98",
        }:
            raise ValueError("unsupported photometric_normalization")
        object.__setattr__(self, "photometric_normalization", normalization)
        object.__setattr__(
            self,
            "minimum_descriptor_valid_fraction",
            _finite_real(
                self.minimum_descriptor_valid_fraction,
                "minimum_descriptor_valid_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if not isinstance(self.root_sift, bool):
            raise TypeError("root_sift must be a boolean")
        if self.root_sift and method != "sift":
            raise ValueError("root_sift is only valid for method='sift'")
        object.__setattr__(
            self,
            "algorithm_parameters",
            tuple(
                sorted((str(key), value) for key, value in self.algorithm_parameters)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "interface": TANGENT_OPENCV_DESCRIPTOR_V2_INTERFACE,
                "patch_interface": TANGENT_PATCH_INTERFACE,
                "hypothesis_identity": "physical-keypoint-id + scale + orientation",
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class TangentDescriptorHypothesisSet:
    """Descriptor hypotheses retaining their physical keypoint identity."""

    physical_keypoint_ids: np.ndarray
    patch_indices: np.ndarray
    scale_multipliers: np.ndarray
    orientation_ranks: np.ndarray
    orientation_confidences: np.ndarray
    descriptor_valid_fractions: np.ndarray
    descriptors: np.ndarray
    angles_deg: np.ndarray
    descriptor_type: str
    descriptor_metric: str
    extractor_name: str
    config: OpenCVTangentDescriptorV2Config
    patch_interface: str = TANGENT_PATCH_INTERFACE
    interface: str = TANGENT_OPENCV_DESCRIPTOR_V2_INTERFACE
    stability: str = "experimental"

    def __len__(self) -> int:
        return int(self.patch_indices.shape[0])


class OpenCVTangentDescriptor:
    """Apply SIFT, ORB, or AKAZE semantics to generic tangent patches."""

    def __init__(
        self,
        config: OpenCVTangentDescriptorConfig | None = None,
        *,
        backend: OpenCVFeatureBackend | None = None,
    ) -> None:
        self.config = config or OpenCVTangentDescriptorConfig()
        self.backend = backend or OpenCVFeatureBackend()

    def describe(
        self,
        patches: TangentPatchSet,
        *,
        responses: np.ndarray | None = None,
    ) -> TangentDescriptorSet:
        self.backend.require_version(self.config.minimum_opencv_version)
        count = len(patches)
        if responses is None:
            response_values = np.ones(count, dtype=np.float64)
        else:
            response_values = np.asarray(responses, dtype=np.float64).reshape(-1)
            if (
                response_values.shape != (count,)
                or not np.isfinite(response_values).all()
            ):
                raise ValueError("responses must contain one finite value per patch")
        extractor_config = FeatureExtractorConfig(
            method=self.config.method,
            max_features=max(1, count),
            edge_margin_px=0,
            deduplicate_overlaps=False,
            parameters=self.config.algorithm_parameters,
        )
        extractor = self.backend.create_extractor(extractor_config)
        metadata = self.backend.descriptor_metadata(extractor_config, extractor)
        indices: list[int] = []
        descriptors: list[np.ndarray] = []
        angles: list[float] = []
        for index, patch in enumerate(patches.patches):
            if not patch.valid:
                continue
            patch_u8 = _opencv_image(patch.image, patch.validity_mask)
            if patch_u8.ndim == 3:
                import cv2

                patch_u8 = cv2.cvtColor(patch_u8, cv2.COLOR_BGR2GRAY)
            angle = (
                _dominant_orientation(patch_u8, self.config.orientation_bins)
                if self.config.orientation_policy == "dominant-gradient"
                else 0.0
            )
            if angle is None:
                continue
            height, width = patch_u8.shape
            centre = np.asarray(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
            samples_per_scale = float(np.mean(patch.geometry.samples_per_scale_xy))
            size = self.config.keypoint_diameter_in_scales * samples_per_scale
            described = self.backend.describe_keypoints(
                patch_u8,
                centre.reshape(1, 2),
                np.asarray((size,), dtype=np.float64),
                np.asarray((angle,), dtype=np.float64),
                np.asarray((response_values[index],), dtype=np.float64),
                np.asarray((0,), dtype=np.int32),
                extractor_config,
                extractor=extractor,
            )
            if described["described_count"] != 1:
                continue
            descriptor = described["descriptors"][0]
            if self.config.root_sift:
                descriptor = _root_sift(descriptor)
            indices.append(index)
            descriptors.append(np.asarray(descriptor))
            angles.append(float(angle))
        dtype = np.float32 if metadata["metric"] == "l2" else np.uint8
        descriptor_array = (
            np.stack(descriptors).astype(dtype, copy=False)
            if descriptors
            else np.empty((0, int(metadata["length"])), dtype=dtype)
        )
        descriptor_type = str(metadata["type"])
        extractor_name = str(metadata["extractor_name"])
        if self.config.root_sift:
            descriptor_type = "root-sift-float32"
            extractor_name = "root-sift"
        return TangentDescriptorSet(
            patch_indices=np.asarray(indices, dtype=np.int64),
            descriptors=descriptor_array,
            angles_deg=np.asarray(angles, dtype=np.float32),
            descriptor_type=f"tangent-{descriptor_type}",
            descriptor_metric=str(metadata["metric"]),
            extractor_name=f"tangent-{extractor_name}",
            config=self.config,
        )


class OpenCVTangentDescriptorV2:
    """Describe one physical keypoint with one or more explicit hypotheses."""

    def __init__(
        self,
        config: OpenCVTangentDescriptorV2Config | None = None,
        *,
        backend: OpenCVFeatureBackend | None = None,
    ) -> None:
        self.config = config or OpenCVTangentDescriptorV2Config()
        self.backend = backend or OpenCVFeatureBackend()

    def describe(
        self,
        patches: TangentPatchSet,
        *,
        responses: np.ndarray | None = None,
    ) -> TangentDescriptorHypothesisSet:
        self.backend.require_version(self.config.minimum_opencv_version)
        count = len(patches)
        if responses is None:
            response_values = np.ones(count, dtype=np.float64)
        else:
            response_values = np.asarray(responses, dtype=np.float64).reshape(-1)
            if (
                response_values.shape != (count,)
                or not np.isfinite(response_values).all()
            ):
                raise ValueError("responses must contain one finite value per patch")
        extractor_config = FeatureExtractorConfig(
            method=self.config.method,
            max_features=max(1, count),
            edge_margin_px=0,
            deduplicate_overlaps=False,
            parameters=self.config.algorithm_parameters,
        )
        extractor = self.backend.create_extractor(extractor_config)
        metadata = self.backend.descriptor_metadata(extractor_config, extractor)
        physical_ids: list[int] = []
        patch_indices: list[int] = []
        scale_multipliers: list[float] = []
        orientation_ranks: list[int] = []
        orientation_confidences: list[float] = []
        descriptor_valid_fractions: list[float] = []
        descriptors: list[np.ndarray] = []
        angles: list[float] = []
        for index, patch in enumerate(patches.patches):
            if not patch.valid:
                continue
            patch_u8 = _opencv_image(patch.image, patch.validity_mask)
            patch_u8 = _normalize_patch_photometry(
                patch_u8,
                patch.validity_mask,
                self.config.photometric_normalization,
            )
            orientations = _descriptor_orientations(patch_u8, self.config)
            if not orientations:
                continue
            height, width = patch_u8.shape
            centre = np.asarray(((width - 1.0) / 2.0, (height - 1.0) / 2.0))
            samples_per_scale = float(np.mean(patch.geometry.samples_per_scale_xy))
            for multiplier in self.config.scale_multipliers:
                size = (
                    self.config.keypoint_diameter_in_scales
                    * multiplier
                    * samples_per_scale
                )
                valid_fraction = _descriptor_support_valid_fraction(
                    patch.validity_mask, centre, size
                )
                if valid_fraction < self.config.minimum_descriptor_valid_fraction:
                    continue
                for orientation_rank, (angle, confidence) in enumerate(orientations):
                    described = self.backend.describe_keypoints(
                        patch_u8,
                        centre.reshape(1, 2),
                        np.asarray((size,), dtype=np.float64),
                        np.asarray((angle,), dtype=np.float64),
                        np.asarray((response_values[index],), dtype=np.float64),
                        np.asarray((0,), dtype=np.int32),
                        extractor_config,
                        extractor=extractor,
                    )
                    if described["described_count"] != 1:
                        continue
                    descriptor = described["descriptors"][0]
                    if self.config.root_sift:
                        descriptor = _root_sift(descriptor)
                    physical_ids.append(int(patch.geometry.keypoint_index))
                    patch_indices.append(index)
                    scale_multipliers.append(float(multiplier))
                    orientation_ranks.append(orientation_rank)
                    orientation_confidences.append(float(confidence))
                    descriptor_valid_fractions.append(valid_fraction)
                    descriptors.append(np.asarray(descriptor))
                    angles.append(float(angle))
        dtype = np.float32 if metadata["metric"] == "l2" else np.uint8
        descriptor_array = (
            np.stack(descriptors).astype(dtype, copy=False)
            if descriptors
            else np.empty((0, int(metadata["length"])), dtype=dtype)
        )
        descriptor_type = str(metadata["type"])
        extractor_name = str(metadata["extractor_name"])
        if self.config.root_sift:
            descriptor_type = "root-sift-float32"
            extractor_name = "root-sift"
        return TangentDescriptorHypothesisSet(
            physical_keypoint_ids=np.asarray(physical_ids, dtype=np.int64),
            patch_indices=np.asarray(patch_indices, dtype=np.int64),
            scale_multipliers=np.asarray(scale_multipliers, dtype=np.float32),
            orientation_ranks=np.asarray(orientation_ranks, dtype=np.int32),
            orientation_confidences=np.asarray(
                orientation_confidences, dtype=np.float32
            ),
            descriptor_valid_fractions=np.asarray(
                descriptor_valid_fractions, dtype=np.float32
            ),
            descriptors=descriptor_array,
            angles_deg=np.asarray(angles, dtype=np.float32),
            descriptor_type=f"tangent-{descriptor_type}",
            descriptor_metric=str(metadata["metric"]),
            extractor_name=f"tangent-{extractor_name}",
            config=self.config,
        )


def _normalize_patch_photometry(
    image: np.ndarray, validity_mask: np.ndarray, policy: str
) -> np.ndarray:
    """Return deterministic uint8 luminance under one explicit policy."""

    values = np.asarray(image)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise TypeError("photometric normalization requires uint8 luminance")
    mask = np.asarray(validity_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("validity mask and luminance must have identical shape")
    if policy == "none":
        return np.ascontiguousarray(values)
    active = values[mask].astype(np.float32)
    output = np.zeros(values.shape, dtype=np.uint8)
    if active.size == 0:
        return output
    working = values.astype(np.float32)
    if policy == "local-standardization":
        mean = float(active.mean(dtype=np.float64))
        standard_deviation = float(active.std(dtype=np.float64))
        if standard_deviation <= 1e-6:
            output[mask] = np.uint8(128)
            return output
        normalized = np.clip((working - mean) / standard_deviation, -3.0, 3.0)
        scaled = (normalized + 3.0) * (255.0 / 6.0)
    elif policy == "robust-percentile-2-98":
        low, high = np.percentile(active, (2.0, 98.0))
        if float(high - low) <= 1e-6:
            output[mask] = np.uint8(128)
            return output
        scaled = (working - float(low)) * (255.0 / float(high - low))
    else:  # pragma: no cover - config validation owns this boundary
        raise ValueError(f"unsupported photometric policy: {policy}")
    output[mask] = np.rint(np.clip(scaled[mask], 0.0, 255.0)).astype(np.uint8)
    return output


def _descriptor_support_valid_fraction(
    validity_mask: np.ndarray, centre_xy: np.ndarray, diameter_px: float
) -> float:
    """Measure validity inside the declared circular descriptor support."""

    mask = np.asarray(validity_mask, dtype=bool)
    yy, xx = np.indices(mask.shape, dtype=np.float64)
    radius = max(0.5 * float(diameter_px), 0.5)
    support = (xx - float(centre_xy[0])) ** 2 + (
        yy - float(centre_xy[1])
    ) ** 2 <= radius * radius
    count = int(np.count_nonzero(support))
    return float(np.count_nonzero(mask & support)) / count if count else 0.0


def _descriptor_orientations(
    image: np.ndarray, config: OpenCVTangentDescriptorV2Config
) -> tuple[tuple[float, float], ...]:
    if config.orientation_policy == "fixed-zero":
        return ((0.0, 1.0),)
    peaks = _orientation_peaks(
        image,
        config.orientation_bins,
        peak_ratio=(
            1.0
            if config.orientation_policy == "dominant-gradient"
            else config.orientation_peak_ratio
        ),
        maximum=(
            1
            if config.orientation_policy == "dominant-gradient"
            else config.max_orientations
        ),
    )
    return peaks


def _dominant_orientation(image: np.ndarray, bins: int) -> float | None:
    import cv2

    values = image.astype(np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    height, width = values.shape
    yy, xx = np.indices(values.shape, dtype=np.float32)
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    radius = min(height, width) * 0.28
    squared_radius = (xx - center_x) ** 2 + (yy - center_y) ** 2
    window = squared_radius <= radius * radius
    weights = magnitude * np.exp(-squared_radius / (2.0 * (0.5 * radius) ** 2))
    if not np.any(weights[window] > 0.0):
        return None
    indices = np.floor(angle * bins / 360.0).astype(np.int64) % bins
    histogram = np.bincount(
        indices[window], weights=weights[window], minlength=bins
    ).astype(np.float64)
    for _ in range(4):
        histogram = (
            np.roll(histogram, 1) + 2.0 * histogram + np.roll(histogram, -1)
        ) / 4.0
    peak = int(np.argmax(histogram))
    left = histogram[(peak - 1) % bins]
    center = histogram[peak]
    right = histogram[(peak + 1) % bins]
    denominator = left - 2.0 * center + right
    offset = 0.0 if abs(denominator) < 1e-15 else 0.5 * (left - right) / denominator
    return float(((peak + offset + 0.5) * 360.0 / bins) % 360.0)


def _orientation_peaks(
    image: np.ndarray,
    bins: int,
    *,
    peak_ratio: float,
    maximum: int,
) -> tuple[tuple[float, float], ...]:
    """Return stable local orientation peaks ordered by strength then angle."""

    import cv2

    values = image.astype(np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    height, width = values.shape
    yy, xx = np.indices(values.shape, dtype=np.float32)
    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    radius = min(height, width) * 0.28
    squared_radius = (xx - center_x) ** 2 + (yy - center_y) ** 2
    window = squared_radius <= radius * radius
    weights = magnitude * np.exp(-squared_radius / (2.0 * (0.5 * radius) ** 2))
    if not np.any(weights[window] > 0.0):
        return ()
    indices = np.floor(angle * bins / 360.0).astype(np.int64) % bins
    histogram = np.bincount(
        indices[window], weights=weights[window], minlength=bins
    ).astype(np.float64)
    for _ in range(4):
        histogram = (
            np.roll(histogram, 1) + 2.0 * histogram + np.roll(histogram, -1)
        ) / 4.0
    strongest = float(histogram.max())
    if strongest <= 0.0:
        return ()
    candidates: list[tuple[float, float]] = []
    for peak in range(bins):
        left = histogram[(peak - 1) % bins]
        center = histogram[peak]
        right = histogram[(peak + 1) % bins]
        if center < peak_ratio * strongest or center < left or center < right:
            continue
        denominator = left - 2.0 * center + right
        offset = 0.0 if abs(denominator) < 1e-15 else 0.5 * (left - right) / denominator
        refined = float(((peak + offset + 0.5) * 360.0 / bins) % 360.0)
        candidates.append((refined, float(center / strongest)))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return tuple(candidates[:maximum])


def _root_sift(descriptor: np.ndarray) -> np.ndarray:
    values = descriptor.astype(np.float32, copy=True)
    total = float(np.sum(np.abs(values)))
    if total > 0.0:
        values = np.sqrt(values / total).astype(np.float32, copy=False)
    return values
