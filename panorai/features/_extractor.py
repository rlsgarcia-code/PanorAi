"""High-level spherical feature extraction over canonical gnomonic faces."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from panorai.geometry import (
    equirectangular_to_gnomonic,
    gnomonic_pixels_to_rays,
    rays_to_erp_pixels,
)

from ._config import FaceSetSpec, FeatureExtractorConfig
from ._geometry import deduplicate_spherical_keypoints, gnomonic_feature_mask
from ._models import (
    DeduplicationResult,
    FaceDetectionDiagnostics,
    FeatureProvenance,
    MultifaceDetectionDiagnostics,
    SphericalFeature,
    SphericalFeatureSet,
)
from .backends.opencv import OpenCVFeatureBackend


class FeatureExtractor:
    """Extract OpenCV features and express them in spherical PanorAi objects."""

    def __init__(
        self,
        config: FeatureExtractorConfig | None = None,
        *,
        backend: OpenCVFeatureBackend | None = None,
        minimum_opencv_version: str = "4.9.0",
    ) -> None:
        self.config = config or FeatureExtractorConfig()
        self.backend = backend or OpenCVFeatureBackend()
        self.minimum_opencv_version = minimum_opencv_version

    def extract(
        self,
        panorama: Any,
        *,
        face_set: Any | None = None,
        face_set_spec: FaceSetSpec | None = None,
        panorama_id: str | None = None,
        validity_mask: Any | None = None,
    ) -> SphericalFeatureSet:
        """Materialize views, run OpenCV, and return spherical features."""

        self.backend.require_version(self.minimum_opencv_version)
        panorama = _as_panorama(panorama)
        image = panorama.image
        source_shape = _spatial_shape(image)
        checksum = _array_checksum(image)
        panorama_id = panorama_id or f"panorama-{checksum[:16]}"
        if face_set is None:
            if face_set_spec is None:
                face_set = panorama.views()
                face_set_spec = _face_set_spec_from_views(face_set)
            else:
                face_set = panorama.views(
                    face_set_spec.sampler,
                    size=face_set_spec.shape_hw,
                    fov=face_set_spec.effective_fov_deg,
                    count=face_set_spec.count,
                    subdivisions=face_set_spec.subdivisions,
                )
        if face_set_spec is None:
            face_set_spec = _face_set_spec_from_views(face_set)
        if validity_mask is not None:
            _validate_erp_mask(validity_mask, image, source_shape)

        raw: list[dict[str, Any]] = []
        face_diagnostics: list[FaceDetectionDiagnostics] = []
        descriptor_blocks: list[np.ndarray] = []
        descriptor_metadata: dict[str, Any] | None = None
        descriptor_offset = 0
        for face in face_set:
            user_face_mask = None
            if validity_mask is not None:
                user_face_mask = equirectangular_to_gnomonic(
                    validity_mask, face.spec, interpolation="nearest"
                ).data
            image_mask = gnomonic_feature_mask(
                face,
                edge_margin_px=0,
                user_mask=user_face_mask,
            )
            detection_mask = gnomonic_feature_mask(
                face,
                edge_margin_px=self.config.edge_margin_px,
                user_mask=user_face_mask,
            )
            image_mask_np = _mask_numpy(image_mask)
            mask_np = _mask_numpy(detection_mask)
            validity_distance = None
            descriptor_distance = None
            if (
                self.config.validity_margin_px
                or self.config.validity_scale_margin > 0.0
            ):
                validity_distance = _mask_distance(image_mask_np)
            if self.config.validity_scale_margin > 0.0:
                assert validity_distance is not None
                descriptor_distance = np.minimum(
                    validity_distance, _image_border_distance(image_mask_np.shape)
                )
            if self.config.validity_margin_px and validity_distance is not None:
                mask_np &= validity_distance > self.config.validity_margin_px
            cv_image = _opencv_image(face.image, image_mask_np)
            detected = self.backend.detect_and_describe(
                cv_image, mask_np.astype(np.uint8) * 255, self.config
            )
            if descriptor_metadata is None:
                descriptor_metadata = detected["metadata"]
            elif detected["metadata"] != descriptor_metadata:
                raise RuntimeError("descriptor metadata changed between faces")
            pixels = detected["pixels_xy"]
            projection = gnomonic_pixels_to_rays(pixels, face.spec)
            keypoint_valid = np.asarray(projection.valid, dtype=bool)
            if len(pixels):
                ix = np.floor(pixels[:, 0] + 0.5).astype(np.int64)
                iy = np.floor(pixels[:, 1] + 0.5).astype(np.int64)
                inside = (
                    (ix >= 0)
                    & (ix < mask_np.shape[1])
                    & (iy >= 0)
                    & (iy < mask_np.shape[0])
                )
                sampled_mask = np.zeros(len(pixels), dtype=bool)
                sampled_mask[inside] = mask_np[iy[inside], ix[inside]]
                keypoint_valid &= sampled_mask
                if self.config.validity_scale_margin > 0.0:
                    assert descriptor_distance is not None
                    distance = np.zeros(len(pixels), dtype=np.float64)
                    distance[inside] = descriptor_distance[iy[inside], ix[inside]]
                    required = self.config.validity_scale_margin * np.asarray(
                        detected["scales"], dtype=np.float64
                    )
                    keypoint_valid &= distance > required
            keep = np.flatnonzero(keypoint_valid)
            face_diagnostics.append(
                FaceDetectionDiagnostics(
                    face_id=face.face_id,
                    detected_count=int(len(pixels)),
                    valid_count=int(len(keep)),
                )
            )
            descriptors = detected["descriptors"][keep]
            descriptor_blocks.append(descriptors)
            bearings = np.asarray(projection.rays_xyz)[keep]
            source_pixels = rays_to_erp_pixels(bearings, source_shape).pixels_xy
            for local_output, detected_index in enumerate(keep):
                raw.append(
                    {
                        "face_id": face.face_id,
                        "spec": face.spec,
                        "pixel_xy": pixels[detected_index].copy(),
                        "source_erp_xy": np.asarray(source_pixels[local_output]).copy(),
                        "bearing_xyz": bearings[local_output].copy(),
                        "response": float(detected["responses"][detected_index]),
                        "scale": float(detected["scales"][detected_index]),
                        "angle_deg": float(detected["angles_deg"][detected_index]),
                        "octave": int(detected["octaves"][detected_index]),
                        "raw_descriptor_index": descriptor_offset + local_output,
                    }
                )
            descriptor_offset += len(descriptors)

        metadata = descriptor_metadata or _empty_descriptor_metadata(self.config)
        all_descriptors = (
            np.concatenate(descriptor_blocks, axis=0)
            if descriptor_blocks
            else np.empty(
                (0, int(metadata["length"])), dtype=_descriptor_dtype(metadata)
            )
        )
        if len(raw):
            bearings = np.stack([item["bearing_xyz"] for item in raw])
            responses = np.asarray([item["response"] for item in raw], dtype=np.float64)
        else:
            bearings = np.empty((0, 3), dtype=np.float64)
            responses = np.empty(0, dtype=np.float64)
        if self.config.deduplicate_overlaps:
            dedup = deduplicate_spherical_keypoints(
                bearings,
                responses,
                angular_threshold_rad=math.radians(
                    self.config.angular_dedup_threshold_deg
                ),
            )
            selected = dedup.selected_indices
        else:
            selected = np.arange(len(raw), dtype=np.int64)
            dedup = DeduplicationResult(
                selected_indices=selected,
                group_index_for_input=selected.copy(),
                angular_distance_to_selected_rad=np.zeros(len(raw), dtype=np.float64),
                duplicate_groups=tuple((int(index),) for index in selected),
                selection_reasons=tuple("deduplication-disabled" for _ in selected),
            )
        unique_after_deduplication = selected.copy()
        if len(selected) > self.config.max_features:
            ranked = np.lexsort((selected, -responses[selected]))
            selected = np.sort(selected[ranked[: self.config.max_features]])

        features: list[SphericalFeature] = []
        descriptor_rows = []
        for output_index, raw_index in enumerate(selected):
            item = raw[int(raw_index)]
            group_index = int(dedup.group_index_for_input[int(raw_index)])
            group = dedup.duplicate_groups[group_index]
            alternatives = tuple(
                raw[index]["face_id"] for index in group if index != raw_index
            )
            duplicate_descriptor_indices = tuple(
                int(raw[index]["raw_descriptor_index"])
                for index in group
                if index != raw_index
            )
            angular_distances = tuple(
                math.degrees(float(dedup.angular_distance_to_selected_rad[index]))
                for index in group
                if index != raw_index
            )
            provenance = FeatureProvenance(
                interface="panorai-spherical-features/v1",
                source_panorama_checksum=checksum,
                projection_backend="panorai.geometry",
                projection_backend_version=_panorai_version(),
                face_id=item["face_id"],
                generating_commit=_panorai_commit(),
                alternative_face_ids=alternatives,
                duplicate_descriptor_indices=duplicate_descriptor_indices,
                duplicate_angular_distances_deg=angular_distances,
            )
            features.append(
                SphericalFeature(
                    feature_id=f"{panorama_id}:feature-{output_index:06d}",
                    panorama_id=panorama_id,
                    face_id=item["face_id"],
                    pixel_xy=item["pixel_xy"],
                    source_erp_xy=item["source_erp_xy"],
                    bearing_xyz=item["bearing_xyz"],
                    response=item["response"],
                    scale=item["scale"],
                    angle_deg=item["angle_deg"],
                    octave=item["octave"],
                    descriptor_index=output_index,
                    valid=True,
                    projection_spec=item["spec"],
                    provenance=provenance,
                )
            )
            descriptor_rows.append(all_descriptors[int(item["raw_descriptor_index"])])
        descriptors = (
            np.stack(descriptor_rows)
            if descriptor_rows
            else np.empty(
                (0, int(metadata["length"])), dtype=_descriptor_dtype(metadata)
            )
        )
        return SphericalFeatureSet(
            panorama_id=panorama_id,
            features=features,
            descriptors=descriptors,
            descriptor_type=str(metadata["type"]),
            descriptor_metric=str(metadata["metric"]),
            extractor_name=str(metadata["extractor_name"]),
            extractor_config=self.config.to_dict(),
            backend_name=self.backend.name,
            backend_version=self.backend.version,
            face_set_spec=face_set_spec,
            projection_backend="panorai.geometry",
            projection_backend_version=_panorai_version(),
            panorama_checksum=checksum,
            generating_commit=_panorai_commit(),
            detection_diagnostics=_multiface_detection_diagnostics(
                raw,
                all_descriptors,
                dedup,
                face_diagnostics,
                per_face_capacity=self.config.effective_max_features_per_face,
                unique_after_deduplication_count=len(unique_after_deduplication),
                output_count=len(features),
            ),
        )


def _multiface_detection_diagnostics(
    raw: list[dict[str, Any]],
    descriptors: np.ndarray,
    deduplication: DeduplicationResult,
    per_face: list[FaceDetectionDiagnostics],
    *,
    per_face_capacity: int,
    unique_after_deduplication_count: int,
    output_count: int,
) -> MultifaceDetectionDiagnostics:
    group_sizes = np.asarray(
        [len(group) for group in deduplication.duplicate_groups], dtype=np.int64
    )
    multiplicity_histogram = {
        str(value): int(np.sum(group_sizes == value))
        for value in np.unique(group_sizes)
    }
    normalized_l2: list[float] = []
    cosine_similarity: list[float] = []
    face_pairs: dict[str, int] = {}
    unique_face_multiplicities: list[int] = []
    cross_face_group_count = 0
    cross_face_candidate_count = 0
    same_face_candidate_count = 0
    for group in deduplication.duplicate_groups:
        group_faces = {str(raw[int(member)]["face_id"]) for member in group}
        unique_face_multiplicities.append(len(group_faces))
        if len(group_faces) > 1:
            cross_face_group_count += 1
        if len(group) <= 1:
            continue
        representative = int(group[0])
        representative_face = str(raw[representative]["face_id"])
        representative_row = int(raw[representative]["raw_descriptor_index"])
        representative_descriptor = descriptors[representative_row].astype(
            np.float64, copy=False
        )
        representative_norm = float(np.linalg.norm(representative_descriptor))
        for member_value in group[1:]:
            member = int(member_value)
            member_face = str(raw[member]["face_id"])
            if member_face == representative_face:
                same_face_candidate_count += 1
                continue
            cross_face_candidate_count += 1
            member_row = int(raw[member]["raw_descriptor_index"])
            member_descriptor = descriptors[member_row].astype(np.float64, copy=False)
            member_norm = float(np.linalg.norm(member_descriptor))
            if representative_norm > 0.0 and member_norm > 0.0:
                cosine = float(
                    np.clip(
                        representative_descriptor @ member_descriptor
                        / (representative_norm * member_norm),
                        -1.0,
                        1.0,
                    )
                )
                cosine_similarity.append(cosine)
                normalized_l2.append(math.sqrt(max(0.0, 2.0 - 2.0 * cosine)))
            pair = "|".join(
                sorted((representative_face, member_face))
            )
            face_pairs[pair] = face_pairs.get(pair, 0) + 1

    distances = np.asarray(normalized_l2, dtype=np.float64)
    cosine_values = np.asarray(cosine_similarity, dtype=np.float64)
    duplicate_groups = int(np.sum(group_sizes > 1))
    duplicate_candidates = int(np.sum(np.maximum(group_sizes - 1, 0)))
    face_multiplicities = np.asarray(unique_face_multiplicities, dtype=np.int64)
    face_multiplicity_histogram = {
        str(value): int(np.sum(face_multiplicities == value))
        for value in np.unique(face_multiplicities)
    }
    return MultifaceDetectionDiagnostics(
        face_count=len(per_face),
        per_face_capacity=int(per_face_capacity),
        per_face_capacity_hit_count=int(
            sum(item.detected_count >= per_face_capacity for item in per_face)
        ),
        detected_count=int(sum(item.detected_count for item in per_face)),
        valid_count=int(sum(item.valid_count for item in per_face)),
        unique_after_deduplication=int(unique_after_deduplication_count),
        output_after_budget=int(output_count),
        duplicate_group_count=duplicate_groups,
        duplicate_candidate_count=duplicate_candidates,
        cross_face_ambiguous_group_count=int(cross_face_group_count),
        cross_face_duplicate_candidate_count=int(cross_face_candidate_count),
        same_face_duplicate_candidate_count=int(same_face_candidate_count),
        maximum_group_multiplicity=int(group_sizes.max()) if len(group_sizes) else 0,
        multiplicity_histogram=multiplicity_histogram,
        maximum_unique_face_multiplicity=(
            int(face_multiplicities.max()) if len(face_multiplicities) else 0
        ),
        unique_face_multiplicity_histogram=face_multiplicity_histogram,
        descriptor_pair_count=int(len(distances)),
        normalized_descriptor_l2_median=(
            float(np.median(distances)) if len(distances) else None
        ),
        normalized_descriptor_l2_p90=(
            float(np.quantile(distances, 0.9)) if len(distances) else None
        ),
        normalized_descriptor_l2_maximum=(
            float(distances.max()) if len(distances) else None
        ),
        descriptor_cosine_similarity_median=(
            float(np.median(cosine_values)) if len(cosine_values) else None
        ),
        descriptor_cosine_similarity_p10=(
            float(np.quantile(cosine_values, 0.1)) if len(cosine_values) else None
        ),
        duplicate_face_pair_histogram=dict(sorted(face_pairs.items())),
        per_face=tuple(per_face),
    )


def extract_opencv_features(
    erp_image: Any,
    face_specs: Any,
    detector: Any,
    *,
    edge_margin_px: int = 0,
    validity_mask: Any | None = None,
) -> SphericalFeatureSet:
    """Advanced route using an injected OpenCV-compatible detector."""

    from panorai.data.gnomonic_image import GnomonicFace
    from panorai.data.gnomonic_imageset import GnomonicFaceSet
    from panorai.geometry import GnomonicProjector

    panorama = _as_panorama(erp_image)
    face_specs = list(face_specs)
    if not face_specs:
        raise ValueError("face_specs cannot be empty")
    faces = []
    for index, spec in enumerate(face_specs):
        projection_input = panorama.image
        if _is_torch(projection_input):
            if not projection_input.dtype.is_floating_point:
                projection_input = projection_input.to(dtype=_torch_float32())
        elif not np.issubdtype(np.asarray(projection_input).dtype, np.floating):
            projection_input = np.asarray(projection_input, dtype=np.float32)
        result = GnomonicProjector(spec).project(projection_input)
        face = GnomonicFace(
            result.data,
            spec.center_lat_deg,
            spec.center_lon_deg,
            spec.hfov_deg,
            face_id=f"face-{index:04d}",
        )
        face.spec = spec
        face.support_mask = result.support_mask
        face._erp_shape_hw = _spatial_shape(panorama.image)
        support_copy = (
            result.support_mask.clone()
            if _is_torch(result.support_mask)
            else result.support_mask.copy()
        )
        face._workflow_metadata = {
            "image": {
                "kind": "image",
                "units": None,
                "validity": support_copy,
                "interpolation": "bilinear",
            }
        }
        face._workflow_support = {
            "image": support_copy.clone()
            if _is_torch(support_copy)
            else support_copy.copy()
        }
        faces.append(face)
    face_set = GnomonicFaceSet(faces)
    config = FeatureExtractorConfig(edge_margin_px=edge_margin_px)
    return FeatureExtractor(
        config,
        backend=OpenCVFeatureBackend(extractor=detector),
    ).extract(
        panorama,
        face_set=face_set,
        face_set_spec=FaceSetSpec(
            sampler="custom",
            shape_hw=face_specs[0].output_shape_hw,
            fov_deg=(face_specs[0].hfov_deg, face_specs[0].vfov_deg),
        ),
        validity_mask=validity_mask,
    )


def _as_panorama(value: Any):
    from panorai.data.equirectangular_image import EquirectangularImage

    if isinstance(value, EquirectangularImage):
        if value.image is None:
            raise ValueError("panorama must contain an image modality")
        return value
    return EquirectangularImage(value)


def _is_torch(value: Any) -> bool:
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _torch_float32():
    import torch

    return torch.float32


def _spatial_shape(value: Any) -> tuple[int, int]:
    return tuple(
        int(item)
        for item in (value.shape[-2:] if _is_torch(value) else value.shape[:2])
    )


def _array_checksum(value: Any) -> str:
    array = (
        value.detach().cpu().contiguous().numpy()
        if _is_torch(value)
        else np.ascontiguousarray(value)
    )
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mask_numpy(mask: Any) -> np.ndarray:
    result = mask.detach().cpu().numpy() if _is_torch(mask) else np.asarray(mask)
    if result.ndim != 2:
        raise ValueError("OpenCV feature extraction requires one unbatched panorama")
    return result.astype(bool, copy=False)


def _mask_distance(mask: np.ndarray) -> np.ndarray:
    """Return Euclidean distance to invalid support in face pixels."""

    import cv2

    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise TypeError("feature mask must be a two-dimensional boolean array")
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def _image_border_distance(shape_hw: tuple[int, int]) -> np.ndarray:
    """Return pixel-center distance to the area immediately outside a raster."""

    height, width = shape_hw
    yy, xx = np.indices(shape_hw, dtype=np.float32)
    return np.minimum.reduce((xx + 1.0, yy + 1.0, width - xx, height - yy))


def _opencv_image(value: Any, mask: np.ndarray) -> np.ndarray:
    array = value.detach().cpu().numpy() if _is_torch(value) else np.asarray(value)
    if _is_torch(value) and array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim not in {2, 3}:
        raise ValueError("OpenCV feature extraction requires HW, HWC, or CHW input")
    if array.shape[:2] != mask.shape:
        raise ValueError("image and feature mask spatial shapes must match")
    working = np.array(array, copy=True)
    if working.ndim == 3:
        working[~mask, :] = 0
    else:
        working[~mask] = 0
    if working.dtype != np.uint8:
        if not np.issubdtype(working.dtype, np.floating):
            raise TypeError("OpenCV feature images must use uint8 or floating data")
        active = working[mask]
        if active.size and not np.isfinite(active).all():
            raise ValueError("valid image support contains non-finite values")
        minimum = float(active.min()) if active.size else 0.0
        maximum = float(active.max()) if active.size else 0.0
        if minimum >= 0.0 and maximum <= 1.0:
            working = np.rint(np.clip(working, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif minimum >= 0.0 and maximum <= 255.0:
            working = np.rint(np.clip(working, 0.0, 255.0)).astype(np.uint8)
        else:
            raise ValueError(
                "floating feature images must be explicitly scaled to [0, 1] or [0, 255]"
            )
    if working.ndim == 3:
        import cv2

        if working.shape[2] == 1:
            working = working[..., 0]
        elif working.shape[2] == 3:
            working = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
        elif working.shape[2] == 4:
            working = cv2.cvtColor(working, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError("feature images must have 1, 3, or 4 channels")
    return np.ascontiguousarray(working)


def _face_set_spec_from_views(face_set: Any) -> FaceSetSpec:
    description = face_set.describe()
    first = description["specs"][0]
    return FaceSetSpec(
        sampler=description["layout"],
        shape_hw=tuple(first["output_shape_hw"]),
        fov_deg=(float(first["hfov_deg"]), float(first["vfov_deg"])),
        count=description["view_count"]
        if description["layout"] in {"fibonacci", "spiral"}
        else None,
    )


def _validate_erp_mask(mask: Any, image: Any, shape: tuple[int, int]) -> None:
    if _is_torch(mask) != _is_torch(image):
        raise TypeError("validity_mask and panorama must use the same backend")
    if _is_torch(mask):
        import torch

        if mask.dtype != torch.bool or mask.device != image.device:
            raise TypeError(
                "Torch validity_mask must be boolean on the panorama device"
            )
    elif not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("NumPy validity_mask must have boolean dtype")
    if tuple(mask.shape) != shape:
        raise ValueError(f"validity_mask must have shape {shape}")


def _empty_descriptor_metadata(config: FeatureExtractorConfig) -> dict[str, Any]:
    if config.method == "sift":
        return {
            "type": "sift-float32",
            "metric": "l2",
            "length": 128,
            "extractor_name": "sift",
        }
    if config.method == "orb":
        return {
            "type": "orb-binary",
            "metric": "hamming",
            "length": 32,
            "extractor_name": "orb",
        }
    binary = config.akaze_descriptor_type == "binary"
    return {
        "type": "akaze-binary" if binary else "akaze-float32",
        "metric": "hamming" if binary else "l2",
        "length": 61 if binary else 64,
        "extractor_name": "akaze",
    }


def _descriptor_dtype(metadata: dict[str, Any]):
    return np.uint8 if metadata["metric"] in {"hamming", "hamming2"} else np.float32


def _panorai_version() -> str:
    from panorai._version import __version__

    return __version__


def _panorai_commit() -> str | None:
    try:
        from panorai._version import __commit_id__
    except ImportError:
        return None
    return None if __commit_id__ is None else str(__commit_id__)
