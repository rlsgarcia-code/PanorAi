"""Typed contracts for dense spherical reconstruction.

All distances are radial ranges.  Geometric support, data validity, and
numeric values remain separate throughout the reconstruction route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from panorai.data import EquirectangularImage
from panorai.geometry import relative_camera_pose
from panorai.stereo.panoramic import (
    FullSphereStereoResult,
    PanoramicStereoPose,
    RegisteredPanoramaSet,
    StereoSourceView,
)

from ._models import SphericalCameraPose

_INTERFACE = "panorai-spherical-dense-reconstruction/v1-experimental"


def _readonly(
    value: Any,
    *,
    dtype: Any | None = None,
    shape: tuple[int, ...] | None = None,
    name: str,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}")
    result.setflags(write=False)
    return result


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisteredSphericalView:
    """One central ERP and its metric camera pose in a shared world frame."""

    image: EquirectangularImage
    pose: SphericalCameraPose
    validity: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, EquirectangularImage):
            raise TypeError("image must be an EquirectangularImage")
        if not isinstance(self.pose, SphericalCameraPose):
            raise TypeError("pose must be a SphericalCameraPose")
        pixels = np.asarray(self.image.image)
        if pixels.ndim not in {2, 3}:
            raise ValueError("registered ERP images must use HW or HWC layout")
        if pixels.shape[1] != 2 * pixels.shape[0]:
            raise ValueError("registered images must be canonical 2:1 ERPs")
        if self.validity is not None:
            validity = np.asarray(self.validity)
            if validity.dtype != np.bool_ or validity.shape != pixels.shape[:2]:
                raise ValueError("validity must be a boolean HW mask")
            object.__setattr__(
                self,
                "validity",
                _readonly(validity, dtype=np.bool_, name="validity"),
            )

    @property
    def panorama_id(self) -> str:
        return self.pose.panorama_id

    @property
    def shape_hw(self) -> tuple[int, int]:
        shape = np.asarray(self.image.image).shape
        return int(shape[0]), int(shape[1])

    def effective_validity(self) -> np.ndarray:
        """Return explicit image validity intersected with geometric support."""

        valid = np.asarray(self.image.validity("image"), dtype=bool).copy()
        valid &= np.asarray(self.image.support("image"), dtype=bool)
        if self.validity is not None:
            valid &= self.validity
        pixels = np.asarray(self.image.image)
        finite = (
            np.isfinite(pixels)
            if pixels.ndim == 2
            else np.all(np.isfinite(pixels), axis=-1)
        )
        if not np.all(finite[valid]):
            raise ValueError(
                f"valid image samples for {self.panorama_id!r} must be finite"
            )
        return valid


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisteredSphericalScene:
    """Metric central-camera ERPs registered in one named world frame."""

    views: tuple[RegisteredSphericalView, ...]
    units: str
    frame: str = "world"

    def __post_init__(self) -> None:
        views = tuple(self.views)
        if len(views) < 2:
            raise ValueError("a registered spherical scene requires at least two views")
        if any(not isinstance(item, RegisteredSphericalView) for item in views):
            raise TypeError("views must contain only RegisteredSphericalView objects")
        ids = tuple(item.panorama_id for item in views)
        if len(ids) != len(set(ids)):
            raise ValueError("registered panorama IDs must be unique")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty physically defined scale")
        if self.units.strip().lower() in {
            "arbitrary",
            "baseline-units",
            "scale-free",
            "unitless",
            "unknown",
        }:
            raise ValueError("scene units must identify a physical or defined scale")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("frame must be a non-empty string")
        object.__setattr__(self, "views", views)

    @property
    def panorama_ids(self) -> tuple[str, ...]:
        return tuple(item.panorama_id for item in self.views)

    def view(self, panorama_id: str) -> RegisteredSphericalView:
        for item in self.views:
            if item.panorama_id == panorama_id:
                return item
        raise KeyError(f"unknown registered panorama {panorama_id!r}")

    def registered_set(self, reference_id: str) -> RegisteredPanoramaSet:
        """Build the exact source-from-reference relations consumed by MVS."""

        reference = self.view(reference_id)
        sources: list[StereoSourceView] = []
        for source in sorted(self.views, key=lambda item: item.panorama_id):
            if source.panorama_id == reference_id:
                continue
            rotation, translation = relative_camera_pose(
                reference.pose.R,
                reference.pose.center,
                source.pose.R,
                source.pose.center,
            )
            baseline = float(np.linalg.norm(translation))
            if not math.isfinite(baseline) or baseline <= 0.0:
                raise ValueError(
                    f"camera centres for {reference_id!r} and "
                    f"{source.panorama_id!r} must be distinct"
                )
            sources.append(
                StereoSourceView(
                    source_id=source.panorama_id,
                    panorama=source.image,
                    pose=PanoramicStereoPose(
                        rotation_source_from_reference=rotation,
                        translation_source_from_reference=translation,
                        baseline=baseline,
                        units=self.units,
                        scale_status="metric",
                        provenance="registered-spherical-scene",
                    ),
                    validity=source.validity,
                )
            )
        return RegisteredPanoramaSet(
            reference_id=reference_id,
            reference=reference.image,
            sources=tuple(sources),
            reference_validity=reference.validity,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiViewRangeMap:
    """One reference ERP radial-range estimate with its complete evidence."""

    reference_id: str
    pose: SphericalCameraPose
    radial_range: np.ndarray
    validity: np.ndarray
    sampling_support: np.ndarray
    observable: np.ndarray
    confidence: np.ndarray
    observable_source_count: np.ndarray
    source_count: np.ndarray
    source_consensus_count: np.ndarray
    contribution_count: np.ndarray
    source_ids: tuple[str, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    interface: str = _INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("reference_id must be a non-empty string")
        if self.pose.panorama_id != self.reference_id:
            raise ValueError("range-map pose ID must equal reference_id")
        radial = np.asarray(self.radial_range)
        if radial.ndim != 2:
            raise ValueError("radial_range must have shape HW")
        shape = radial.shape
        radial = _readonly(radial, dtype=np.float32, shape=shape, name="radial_range")
        validity = _readonly(
            self.validity, dtype=np.bool_, shape=shape, name="validity"
        )
        support = _readonly(
            self.sampling_support,
            dtype=np.bool_,
            shape=shape,
            name="sampling_support",
        )
        observable = _readonly(
            self.observable, dtype=np.bool_, shape=shape, name="observable"
        )
        if np.any(validity & ~np.isfinite(radial)):
            raise ValueError("valid radial ranges must be finite")
        if np.any(~validity & ~np.isnan(radial)):
            raise ValueError("invalid radial ranges must be NaN")
        if np.any(validity & (~support | ~observable)):
            raise ValueError("valid ranges must be supported and observable")
        object.__setattr__(self, "radial_range", radial)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "sampling_support", support)
        object.__setattr__(self, "observable", observable)
        confidence = _readonly(
            self.confidence, dtype=np.float32, shape=shape, name="confidence"
        )
        if np.any(~np.isfinite(confidence)) or np.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        if np.any(confidence[~validity] != 0.0):
            raise ValueError("invalid range samples must have zero confidence")
        object.__setattr__(
            self,
            "confidence",
            confidence,
        )
        for name in (
            "observable_source_count",
            "source_count",
            "source_consensus_count",
            "contribution_count",
        ):
            object.__setattr__(
                self,
                name,
                _readonly(getattr(self, name), dtype=np.uint16, shape=shape, name=name),
            )
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source_ids must be unique")
        if self.reference_id in self.source_ids:
            raise ValueError("reference_id cannot also be a source_id")
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    @classmethod
    def from_stereo_result(
        cls,
        *,
        reference_id: str,
        pose: SphericalCameraPose,
        result: FullSphereStereoResult,
    ) -> "MultiViewRangeMap":
        return cls(
            reference_id=reference_id,
            pose=pose,
            radial_range=result.radial_range,
            validity=result.validity,
            sampling_support=result.sampling_support,
            observable=result.observable,
            confidence=result.confidence,
            observable_source_count=result.observable_source_count,
            source_count=result.source_count,
            source_consensus_count=result.source_consensus_count,
            contribution_count=result.contribution_count,
            source_ids=result.source_ids,
            diagnostics=result.diagnostics,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiViewRangeSet:
    """Per-camera radial ranges tied to their registered input scene."""

    scene: RegisteredSphericalScene
    maps: tuple[MultiViewRangeMap, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    interface: str = _INTERFACE

    def __post_init__(self) -> None:
        maps = tuple(self.maps)
        if not maps:
            raise ValueError("a range set must contain at least one reference map")
        ids = tuple(item.reference_id for item in maps)
        if len(ids) != len(set(ids)):
            raise ValueError("reference IDs in a range set must be unique")
        unknown = set(ids).difference(self.scene.panorama_ids)
        if unknown:
            raise ValueError(f"range maps reference unknown cameras: {sorted(unknown)}")
        for item in maps:
            expected_shape = self.scene.view(item.reference_id).shape_hw
            if item.radial_range.shape != expected_shape:
                raise ValueError(
                    f"range map {item.reference_id!r} has shape "
                    f"{item.radial_range.shape}; expected {expected_shape}"
                )
        object.__setattr__(self, "maps", maps)
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    def by_reference(self, reference_id: str) -> MultiViewRangeMap:
        for item in self.maps:
            if item.reference_id == reference_id:
                return item
        raise KeyError(f"unknown range-map reference {reference_id!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class TSDFSpec:
    """Deterministic bounded dense-TSDF configuration."""

    voxel_size: float
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    truncation_distance: float | None = None
    max_voxels: int = 8_000_000
    chunk_voxels: int = 262_144
    min_surface_views: int = 2
    min_valid_weight: float = 1.0
    weighting: str = "uniform"

    def __post_init__(self) -> None:
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive and finite")
        truncation = (
            4.0 * self.voxel_size
            if self.truncation_distance is None
            else float(self.truncation_distance)
        )
        if not math.isfinite(truncation) or truncation <= 0.0:
            raise ValueError("truncation_distance must be positive and finite")
        if self.bounds is not None:
            bounds = np.asarray(self.bounds, dtype=np.float64)
            if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
                raise ValueError("bounds must contain finite min/max XYZ triples")
            if np.any(bounds[1] <= bounds[0]):
                raise ValueError("bounds maxima must be greater than minima")
            object.__setattr__(
                self,
                "bounds",
                (tuple(bounds[0].tolist()), tuple(bounds[1].tolist())),
            )
        for name in ("max_voxels", "chunk_voxels", "min_surface_views"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.min_valid_weight) <= 1.0:
            raise ValueError("min_valid_weight must be in (0, 1]")
        if self.weighting not in {"uniform", "confidence"}:
            raise ValueError("weighting must be 'uniform' or 'confidence'")
        object.__setattr__(self, "truncation_distance", truncation)


@dataclass(frozen=True, slots=True, kw_only=True)
class TSDFVolume:
    """Dense ZYX TSDF grid in a named world frame."""

    tsdf_zyx: np.ndarray
    weight_zyx: np.ndarray
    view_count_zyx: np.ndarray
    origin_xyz: np.ndarray
    voxel_size: float
    truncation_distance: float
    units: str
    frame: str
    min_surface_views: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    interface: str = _INTERFACE

    def __post_init__(self) -> None:
        tsdf = np.asarray(self.tsdf_zyx)
        if tsdf.ndim != 3 or any(size < 2 for size in tsdf.shape):
            raise ValueError("tsdf_zyx must be a 3-D grid with each size at least 2")
        shape = tsdf.shape
        tsdf = _readonly(tsdf, dtype=np.float32, shape=shape, name="tsdf_zyx")
        weight = _readonly(
            self.weight_zyx, dtype=np.float32, shape=shape, name="weight_zyx"
        )
        view_count = _readonly(
            self.view_count_zyx,
            dtype=np.uint16,
            shape=shape,
            name="view_count_zyx",
        )
        if np.any(~np.isfinite(tsdf)) or np.any((tsdf < -1.0) | (tsdf > 1.0)):
            raise ValueError("TSDF values must be finite and in [-1, 1]")
        if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
            raise ValueError("TSDF weights must be finite and non-negative")
        if np.any((weight == 0.0) & (view_count != 0)):
            raise ValueError("zero-weight voxels must have zero view count")
        object.__setattr__(self, "tsdf_zyx", tsdf)
        object.__setattr__(self, "weight_zyx", weight)
        object.__setattr__(self, "view_count_zyx", view_count)
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive and finite")
        if (
            not math.isfinite(self.truncation_distance)
            or self.truncation_distance <= 0.0
        ):
            raise ValueError("truncation_distance must be positive and finite")
        if self.min_surface_views < 1:
            raise ValueError("min_surface_views must be positive")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty string")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("frame must be a non-empty string")
        object.__setattr__(
            self,
            "origin_xyz",
            _readonly(self.origin_xyz, dtype=np.float64, shape=(3,), name="origin_xyz"),
        )
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    @property
    def bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        extent_xyz = np.asarray(self.tsdf_zyx.shape[::-1]) * self.voxel_size
        return self.origin_xyz.copy(), self.origin_xyz + extent_xyz


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceMesh:
    """World-frame triangle mesh with geometric and color evidence."""

    vertices_xyz: np.ndarray
    faces: np.ndarray
    normals_xyz: np.ndarray
    vertex_weight: np.ndarray
    vertex_view_count: np.ndarray
    colors_rgb: np.ndarray | None = None
    color_validity: np.ndarray | None = None
    color_contribution_count: np.ndarray | None = None
    units: str = "m"
    frame: str = "world"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    interface: str = _INTERFACE

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_xyz)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("vertices_xyz must have shape (N, 3)")
        count = len(vertices)
        faces = np.asarray(self.faces)
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("faces must have shape (M, 3)")
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError("faces must use an integer dtype")
        if faces.size and (np.any(faces < 0) or np.any(faces >= count)):
            raise ValueError("faces contain out-of-range vertex indices")
        object.__setattr__(
            self,
            "vertices_xyz",
            _readonly(
                vertices, dtype=np.float64, shape=(count, 3), name="vertices_xyz"
            ),
        )
        object.__setattr__(
            self,
            "faces",
            _readonly(faces, dtype=np.int32, shape=faces.shape, name="faces"),
        )
        object.__setattr__(
            self,
            "normals_xyz",
            _readonly(
                self.normals_xyz,
                dtype=np.float64,
                shape=(count, 3),
                name="normals_xyz",
            ),
        )
        if not np.all(np.isfinite(self.vertices_xyz)):
            raise ValueError("vertices_xyz must be finite")
        if not np.all(np.isfinite(self.normals_xyz)):
            raise ValueError("normals_xyz must be finite")
        normal_lengths = np.linalg.norm(self.normals_xyz, axis=1)
        if count and not np.allclose(normal_lengths, 1.0, atol=1e-5):
            raise ValueError("normals_xyz must contain unit vectors")
        object.__setattr__(
            self,
            "vertex_weight",
            _readonly(
                self.vertex_weight,
                dtype=np.float32,
                shape=(count,),
                name="vertex_weight",
            ),
        )
        object.__setattr__(
            self,
            "vertex_view_count",
            _readonly(
                self.vertex_view_count,
                dtype=np.uint16,
                shape=(count,),
                name="vertex_view_count",
            ),
        )
        color_items = (
            self.colors_rgb,
            self.color_validity,
            self.color_contribution_count,
        )
        if any(item is not None for item in color_items) and not all(
            item is not None for item in color_items
        ):
            raise ValueError(
                "colors_rgb, color_validity, and color_contribution_count "
                "must be provided together"
            )
        if self.colors_rgb is not None:
            colors = _readonly(
                self.colors_rgb,
                dtype=np.float32,
                shape=(count, 3),
                name="colors_rgb",
            )
            if np.any(~np.isfinite(colors)) or np.any((colors < 0) | (colors > 1)):
                raise ValueError("colors_rgb must be finite and in [0, 1]")
            object.__setattr__(self, "colors_rgb", colors)
        if self.color_validity is not None:
            object.__setattr__(
                self,
                "color_validity",
                _readonly(
                    self.color_validity,
                    dtype=np.bool_,
                    shape=(count,),
                    name="color_validity",
                ),
            )
        if self.color_contribution_count is not None:
            object.__setattr__(
                self,
                "color_contribution_count",
                _readonly(
                    self.color_contribution_count,
                    dtype=np.uint16,
                    shape=(count,),
                    name="color_contribution_count",
                ),
            )
        object.__setattr__(self, "provenance", _mapping(self.provenance))
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty string")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("frame must be a non-empty string")


@dataclass(frozen=True, slots=True, kw_only=True)
class SphericalDenseReconstructionResult:
    """Auditable accepted or rejected dense-reconstruction outcome."""

    status: str
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    surface: SurfaceMesh | None = None
    ranges: MultiViewRangeSet | None = None
    volume: TSDFVolume | None = None
    interface: str = _INTERFACE

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "rejected"}:
            raise ValueError("status must be 'accepted' or 'rejected'")
        reasons = tuple(str(item) for item in self.reasons)
        if self.status == "accepted" and self.surface is None:
            raise ValueError("an accepted result requires a surface")
        if self.status == "rejected" and not reasons:
            raise ValueError("a rejected result requires at least one reason")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "diagnostics", _mapping(self.diagnostics))

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def require_success(self) -> "SphericalDenseReconstructionResult":
        if not self.accepted:
            raise RuntimeError(
                "dense reconstruction was rejected: " + "; ".join(self.reasons)
            )
        return self


__all__ = [
    "MultiViewRangeMap",
    "MultiViewRangeSet",
    "RegisteredSphericalScene",
    "RegisteredSphericalView",
    "SphericalDenseReconstructionResult",
    "SurfaceMesh",
    "TSDFSpec",
    "TSDFVolume",
]
