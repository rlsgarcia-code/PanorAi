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
    FeatureProvenance,
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
        minimum_opencv_version: str = "4.8.0",
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
        descriptor_blocks: list[np.ndarray] = []
        descriptor_metadata: dict[str, Any] | None = None
        descriptor_offset = 0
        for face in face_set:
            user_face_mask = None
            if validity_mask is not None:
                user_face_mask = equirectangular_to_gnomonic(
                    validity_mask, face.spec, interpolation="nearest"
                ).data
            mask = gnomonic_feature_mask(
                face,
                edge_margin_px=self.config.edge_margin_px,
                user_mask=user_face_mask,
            )
            mask_np = _mask_numpy(mask)
            cv_image = _opencv_image(face.image, mask_np)
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
            keep = np.flatnonzero(keypoint_valid)
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
            extractor_name=self.config.method,
            extractor_config=self.config.to_dict(),
            backend_name=self.backend.name,
            backend_version=self.backend.version,
            face_set_spec=face_set_spec,
            projection_backend="panorai.geometry",
            projection_backend_version=_panorai_version(),
            panorama_checksum=checksum,
            generating_commit=_panorai_commit(),
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
        return {"type": "sift-float32", "metric": "l2", "length": 128}
    if config.method == "orb":
        return {"type": "orb-binary", "metric": "hamming", "length": 32}
    binary = config.akaze_descriptor_type == "binary"
    return {
        "type": "akaze-binary" if binary else "akaze-float32",
        "metric": "hamming" if binary else "l2",
        "length": 61 if binary else 64,
    }


def _descriptor_dtype(metadata: dict[str, Any]):
    return np.uint8 if metadata["metric"] == "hamming" else np.float32


def _panorai_version() -> str:
    from panorai._version import __version__

    return __version__


def _panorai_commit() -> str | None:
    try:
        from panorai._version import __commit_id__
    except ImportError:
        return None
    return None if __commit_id__ is None else str(__commit_id__)
