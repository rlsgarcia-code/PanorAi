"""Stable NumPy-canonical observations lifted from gnomonic views."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any, Generic, Mapping, Sequence, TypeVar

import numpy as np

from panorai.geometry import (
    GnomonicProjector,
    GnomonicSpec,
    erp_pixels_to_rays,
    gnomonic_pixels_to_rays,
    rays_to_erp_pixels,
)

from .spherical_mask import SphericalBinaryMask


T = TypeVar("T")


_REGION_PROJECTION_BATCH_LIMIT = 16
_REGION_PROJECTION_BYTE_BUDGET = 16 * 1024 * 1024
_MASK_SCAN_CHUNK_PIXELS = 65_536


def _reject_torch(value: object, name: str, *, _seen: set[int] | None = None) -> None:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        raise TypeError(
            f"{name} is NumPy-canonical; convert Torch explicitly with from_torch()"
        )
    seen = set() if _seen is None else _seen
    marker = id(value)
    if marker in seen:
        return
    seen.add(marker)
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_torch(item, name, _seen=seen)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            _reject_torch(item, name, _seen=seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_torch(item, name, _seen=seen)


def _readonly(
    value: object, *, dtype: np.dtype | type | None = None, ndim: int | None = None
) -> np.ndarray:
    _reject_torch(value, "observation arrays")
    result = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"array must have {ndim} dimensions")
    result.setflags(write=False)
    return result


def _boolean_array(value: object, name: str, *, ndim: int) -> np.ndarray:
    result = _readonly(value, ndim=ndim)
    if result.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    return result


def _aligned_optional(
    value: object | None,
    count: int,
    name: str,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    if dtype in {bool, np.bool_}:
        result = _boolean_array(value, name, ndim=1)
    else:
        result = _readonly(value, dtype=dtype)
    if result.ndim != 1 or len(result) != count:
        raise ValueError(f"{name} must have shape ({count},)")
    return result


def _attributes(values: Mapping[str, object], count: int) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        key = str(name).strip()
        if not key:
            raise ValueError("attribute names must be non-empty")
        item = _readonly(value)
        if item.ndim == 0 or item.shape[0] != count:
            raise ValueError(f"attribute {key!r} must have leading dimension {count}")
        result[key] = item
    return MappingProxyType(result)


def _scalar_attributes(values: Mapping[str, object]) -> Mapping[str, object]:
    result: dict[str, object] = {}
    for name, value in values.items():
        key = str(name).strip()
        if not key:
            raise ValueError("attribute names must be non-empty")
        _reject_torch(value, f"attribute {key!r}")
        result[key] = _readonly(value) if isinstance(value, np.ndarray) else value
    return MappingProxyType(result)


def _shape_hw(value: object, *, optional: bool = False) -> tuple[int, int] | None:
    if optional and value is None:
        return None
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("shape must contain exactly (height,width)") from exc
    if len(raw) != 2:
        raise ValueError("shape must contain exactly (height,width)")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in raw):
        raise TypeError("shape dimensions must be integers")
    shape = (int(raw[0]), int(raw[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("shape dimensions must be positive")
    return shape


def _torch_numpy(value: object | None) -> object | None:
    if value is None:
        return None
    module = type(value).__module__
    if not (module == "torch" or module.startswith("torch.")):
        return value
    return value.detach().cpu().numpy()  # type: ignore[attr-defined]


def _boolean_scalar(value: object, name: str) -> bool:
    _reject_torch(value, name)
    scalar = np.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar boolean")
    if scalar.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    return bool(scalar.item())


def _score_order(score: float | None) -> tuple[bool, float]:
    """Sort known scores descending and leave unscored items last."""

    return score is None, 0.0 if score is None else -score


def _optional_score(value: object | None) -> float | None:
    if value is None:
        return None
    _reject_torch(value, "score")
    scalar = np.asarray(value)
    if scalar.ndim != 0:
        raise ValueError("score must be scalar")
    score = float(scalar.item())
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return score


def _class_id(value: object | None) -> object | None:
    if value is None:
        return None
    _reject_torch(value, "class_id")
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("class_id must be scalar")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("class_id must be a hashable scalar") from exc
    return value


@dataclass(frozen=True, slots=True)
class FacePointBatch:
    """One detector's point outputs in canonical face pixel coordinates."""

    pixels_xy: np.ndarray
    valid: np.ndarray | None = None
    scores: np.ndarray | None = None
    class_ids: np.ndarray | None = None
    attributes: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pixels = _readonly(self.pixels_xy, dtype=np.float64, ndim=2)
        if pixels.shape[1:] != (2,):
            raise ValueError("pixels_xy must have shape (N,2)")
        count = len(pixels)
        valid = _aligned_optional(self.valid, count, "valid", dtype=bool)
        if valid is None:
            valid = _readonly(np.ones(count, dtype=bool))
        scores = _aligned_optional(self.scores, count, "scores", dtype=np.float64)
        if scores is not None and np.any(~np.isfinite(scores)):
            raise ValueError("scores must be finite")
        class_ids = _aligned_optional(self.class_ids, count, "class_ids")
        attrs = _attributes(self.attributes, count)
        object.__setattr__(self, "pixels_xy", pixels)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "attributes", attrs)

    @classmethod
    def from_torch(
        cls,
        pixels_xy: object,
        *,
        valid: object | None = None,
        scores: object | None = None,
        class_ids: object | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> "FacePointBatch":
        return cls(
            _torch_numpy(pixels_xy),
            valid=_torch_numpy(valid),
            scores=_torch_numpy(scores),
            class_ids=_torch_numpy(class_ids),
            attributes={
                key: _torch_numpy(value) for key, value in (attributes or {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class SphericalPointObservation:
    observation_id: str
    source_id: str
    face_id: str
    pixel_xy: np.ndarray
    source_erp_xy: np.ndarray | None
    bearing_xyz: np.ndarray
    valid: bool
    score: float | None = None
    class_id: object | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    spec: GnomonicSpec | None = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.source_id or not self.face_id:
            raise ValueError("observation_id, source_id, and face_id must be non-empty")
        pixel = _readonly(self.pixel_xy, dtype=np.float64, ndim=1)
        bearing = _readonly(self.bearing_xyz, dtype=np.float64, ndim=1)
        if pixel.shape != (2,) or bearing.shape != (3,):
            raise ValueError("pixel_xy and bearing_xyz must have shapes (2,) and (3,)")
        if np.any(~np.isfinite(pixel)):
            raise ValueError("pixel_xy must be finite")
        valid = _boolean_scalar(self.valid, "valid")
        if valid:
            if np.any(~np.isfinite(bearing)):
                raise ValueError("valid bearing_xyz must be finite")
            norm = float(np.linalg.norm(bearing))
            if not np.isclose(norm, 1.0, atol=1e-9):
                raise ValueError("valid bearing_xyz must be a unit vector")
        source_pixel = None
        if self.source_erp_xy is not None:
            source_pixel = _readonly(self.source_erp_xy, dtype=np.float64, ndim=1)
            if source_pixel.shape != (2,):
                raise ValueError("source_erp_xy must have shape (2,)")
        score = _optional_score(self.score)
        class_id = _class_id(self.class_id)
        object.__setattr__(self, "pixel_xy", pixel)
        object.__setattr__(self, "source_erp_xy", source_pixel)
        object.__setattr__(self, "bearing_xyz", bearing)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "attributes", _scalar_attributes(self.attributes))


@dataclass(frozen=True, slots=True)
class ObservationFusionResult(Generic[T]):
    observations: T
    kept_indices: tuple[int, ...]
    suppressed_by: tuple[tuple[int, int], ...]
    invalid_indices: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    policy: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class SphericalPointSet:
    source_id: str
    observations: tuple[SphericalPointObservation, ...]
    erp_shape_hw: tuple[int, int] | None

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise ValueError("source_id must be non-empty")
        if any(item.source_id != self.source_id for item in self.observations):
            raise ValueError("all observations must share source_id")
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(
            self, "erp_shape_hw", _shape_hw(self.erp_shape_hw, optional=True)
        )

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def bearings(self) -> np.ndarray:
        if not self.observations:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack([item.bearing_xyz for item in self.observations])

    def angular_nms(
        self, max_angle_deg: float, *, class_aware: bool = True
    ) -> ObservationFusionResult["SphericalPointSet"]:
        threshold = float(max_angle_deg)
        if not math.isfinite(threshold) or threshold < 0 or threshold > 180:
            raise ValueError("max_angle_deg must be finite and within [0,180]")
        valid_indices = [i for i, item in enumerate(self.observations) if item.valid]
        invalid = tuple(i for i, item in enumerate(self.observations) if not item.valid)
        order = sorted(
            valid_indices,
            key=lambda i: (
                *_score_order(self.observations[i].score),
                self.observations[i].face_id,
                i,
            ),
        )
        kept: list[int] = []
        suppressed: list[tuple[int, int]] = []
        groups: dict[int, list[int]] = {}
        cosine = math.cos(math.radians(threshold))
        bearings = self.bearings
        for index in order:
            item = self.observations[index]
            owner = None
            if kept:
                dot_tolerance = 32.0 * np.finfo(np.float64).eps
                matches = bearings[kept] @ item.bearing_xyz >= (cosine - dot_tolerance)
                if class_aware:
                    matches &= np.asarray(
                        [
                            item.class_id == self.observations[kept_index].class_id
                            for kept_index in kept
                        ],
                        dtype=bool,
                    )
                matched = np.flatnonzero(matches)
                if len(matched):
                    owner = kept[int(matched[0])]
            if owner is None:
                kept.append(index)
                groups[index] = [index]
            else:
                suppressed.append((index, owner))
                groups[owner].append(index)
        output = SphericalPointSet(
            self.source_id,
            tuple(self.observations[index] for index in kept),
            self.erp_shape_hw,
        )
        return ObservationFusionResult(
            output,
            tuple(kept),
            tuple(sorted(suppressed)),
            invalid,
            tuple(tuple(groups[index]) for index in kept),
            "angular-nms",
            {"max_angle_deg": threshold, "class_aware": class_aware},
        )


def lift_face_points(
    face: Any,
    batch: FacePointBatch,
    *,
    source_id: str,
    erp_shape_hw: tuple[int, int] | None = None,
) -> SphericalPointSet:
    if not str(source_id).strip():
        raise ValueError("source_id must be non-empty")
    spec = face.spec or face.get_geometry().spec
    inherited_shape = face._erp_shape_hw if erp_shape_hw is None else erp_shape_hw
    shape = _shape_hw(inherited_shape, optional=True)
    projected = gnomonic_pixels_to_rays(batch.pixels_xy, spec)
    support = np.asarray(face.get_geometry().support_mask, dtype=bool)
    height, width = support.shape
    nearest_x = np.clip(np.floor(batch.pixels_xy[:, 0] + 0.5).astype(int), 0, width - 1)
    nearest_y = np.clip(
        np.floor(batch.pixels_xy[:, 1] + 0.5).astype(int), 0, height - 1
    )
    supported = support[nearest_y, nearest_x]
    valid = np.asarray(batch.valid) & np.asarray(projected.valid) & supported
    erp_pixels = (
        None
        if shape is None
        else rays_to_erp_pixels(projected.rays_xyz, shape).pixels_xy
    )
    observations = []
    for index in range(len(batch.pixels_xy)):
        attributes = {
            name: np.array(value[index], copy=True)
            for name, value in batch.attributes.items()
        }
        for value in attributes.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        pixel = _readonly(batch.pixels_xy[index], dtype=np.float64)
        bearing = _readonly(projected.rays_xyz[index], dtype=np.float64)
        source_pixel = (
            None
            if erp_pixels is None
            else _readonly(erp_pixels[index], dtype=np.float64)
        )
        observations.append(
            SphericalPointObservation(
                observation_id=f"{face.face_id}:{index}",
                source_id=str(source_id),
                face_id=face.face_id,
                pixel_xy=pixel,
                source_erp_xy=source_pixel,
                bearing_xyz=bearing,
                valid=bool(valid[index]),
                score=None if batch.scores is None else float(batch.scores[index]),
                class_id=None
                if batch.class_ids is None
                else _class_id(batch.class_ids[index]),
                attributes=MappingProxyType(attributes),
                spec=spec,
            )
        )
    return SphericalPointSet(str(source_id), tuple(observations), shape)


@dataclass(frozen=True, slots=True)
class FaceRegionBatch:
    """Homogeneous face-local boxes, polygons, or masks."""

    kind: str
    geometries: tuple[np.ndarray, ...]
    valid: np.ndarray
    scores: np.ndarray | None = None
    class_ids: np.ndarray | None = None
    attributes: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"box", "polygon", "mask"}:
            raise ValueError("kind must be 'box', 'polygon', or 'mask'")
        geometries = tuple(_readonly(item) for item in self.geometries)
        count = len(geometries)
        valid = _aligned_optional(self.valid, count, "valid", dtype=bool)
        if valid is None:
            raise ValueError("valid is required")
        scores = _aligned_optional(self.scores, count, "scores", dtype=np.float64)
        if scores is not None and np.any(~np.isfinite(scores)):
            raise ValueError("scores must be finite")
        class_ids = _aligned_optional(self.class_ids, count, "class_ids")
        attrs = _attributes(self.attributes, count)
        object.__setattr__(self, "geometries", geometries)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "attributes", attrs)

    @classmethod
    def from_boxes(
        cls,
        boxes_xyxy: object,
        *,
        valid: object | None = None,
        scores: object | None = None,
        class_ids: object | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> "FaceRegionBatch":
        boxes = _readonly(boxes_xyxy, dtype=np.float64, ndim=2)
        if boxes.shape[1:] != (4,):
            raise ValueError("boxes_xyxy must have shape (N,4)")
        if np.any(~np.isfinite(boxes)):
            raise ValueError("boxes must be finite")
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise ValueError("boxes must have x1>x0 and y1>y0")
        count = len(boxes)
        mask = np.ones(count, dtype=bool) if valid is None else valid
        return cls("box", tuple(boxes), mask, scores, class_ids, attributes or {})

    @classmethod
    def from_polygons(
        cls,
        polygons_xy: Sequence[object],
        *,
        valid: object | None = None,
        scores: object | None = None,
        class_ids: object | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> "FaceRegionBatch":
        polygons = tuple(
            _readonly(item, dtype=np.float64, ndim=2) for item in polygons_xy
        )
        if any(item.shape[1:] != (2,) or len(item) < 3 for item in polygons):
            raise ValueError("each polygon must have shape (M,2) with M>=3")
        if any(np.any(~np.isfinite(item)) for item in polygons):
            raise ValueError("polygon coordinates must be finite")
        mask = np.ones(len(polygons), dtype=bool) if valid is None else valid
        return cls("polygon", polygons, mask, scores, class_ids, attributes or {})

    @classmethod
    def from_masks(
        cls,
        masks: object,
        *,
        valid: object | None = None,
        scores: object | None = None,
        class_ids: object | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> "FaceRegionBatch":
        values = _readonly(masks, ndim=3)
        if values.dtype != np.bool_:
            raise TypeError("masks must have boolean dtype")
        mask = np.ones(len(values), dtype=bool) if valid is None else valid
        return cls("mask", tuple(values), mask, scores, class_ids, attributes or {})

    @classmethod
    def from_torch(
        cls, kind: str, geometries: object, **kwargs: object
    ) -> "FaceRegionBatch":
        if kind == "polygon":
            converted_value = _torch_numpy(geometries)
            if isinstance(converted_value, np.ndarray):
                converted = (
                    (converted_value,)
                    if converted_value.ndim == 2
                    else tuple(converted_value)
                )
            else:
                converted = tuple(_torch_numpy(item) for item in geometries)  # type: ignore[union-attr]
        else:
            converted = _torch_numpy(geometries)
        converted_kwargs = {
            key: (
                {name: _torch_numpy(item) for name, item in value.items()}
                if key == "attributes" and isinstance(value, Mapping)
                else _torch_numpy(value)
            )
            for key, value in kwargs.items()
        }
        if kind == "box":
            return cls.from_boxes(converted, **converted_kwargs)
        if kind == "mask":
            return cls.from_masks(converted, **converted_kwargs)
        if kind == "polygon":
            return cls.from_polygons(converted, **converted_kwargs)
        raise ValueError("kind must be 'box', 'polygon', or 'mask'")


def _polygon_mask(shape_hw: tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    height, width = shape_hw
    y, x = np.indices((height, width), dtype=np.float64)
    inside = np.zeros((height, width), dtype=bool)
    previous = polygon_xy[-1]
    for current in polygon_xy:
        x0, y0 = previous
        x1, y1 = current
        crosses = ((y0 > y) != (y1 > y)) & (
            x < (x1 - x0) * (y - y0) / ((y1 - y0) + 1e-300) + x0
        )
        inside ^= crosses
        previous = current
    return inside


def _clip_polygon(polygon_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Clip a continuous pixel-centre polygon to the complete face footprint."""

    height, width = shape_hw
    points = [np.asarray(point, dtype=np.float64) for point in polygon_xy]

    def clip_edge(points, inside, intersection):
        if not points:
            return []
        result = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside != previous_inside:
                result.append(intersection(previous, current))
            if current_inside:
                result.append(current)
            previous, previous_inside = current, current_inside
        return result

    boundaries = (
        (
            lambda point: point[0] >= -0.5,
            lambda first, second: (
                first + (second - first) * ((-0.5 - first[0]) / (second[0] - first[0]))
            ),
        ),
        (
            lambda point: point[0] <= width - 0.5,
            lambda first, second: (
                first
                + (second - first) * ((width - 0.5 - first[0]) / (second[0] - first[0]))
            ),
        ),
        (
            lambda point: point[1] >= -0.5,
            lambda first, second: (
                first + (second - first) * ((-0.5 - first[1]) / (second[1] - first[1]))
            ),
        ),
        (
            lambda point: point[1] <= height - 0.5,
            lambda first, second: (
                first
                + (second - first)
                * ((height - 0.5 - first[1]) / (second[1] - first[1]))
            ),
        ),
    )
    for inside, intersection in boundaries:
        points = clip_edge(points, inside, intersection)
    return (
        np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if points
        else np.empty((0, 2), dtype=np.float64)
    )


def _backproject_mask(
    local_mask: np.ndarray,
    support: np.ndarray,
    spec: GnomonicSpec,
    output_shape_hw: tuple[int, int],
) -> SphericalBinaryMask:
    projector = GnomonicProjector(spec, interpolation="nearest")
    region = projector.back_project(local_mask, output_shape_hw)
    valid_support = projector.back_project(support, output_shape_hw)
    mask = (
        np.asarray(region.data, dtype=bool)
        & np.asarray(region.support_mask, dtype=bool)
        & np.asarray(valid_support.data, dtype=bool)
        & np.asarray(valid_support.support_mask, dtype=bool)
    )
    return SphericalBinaryMask.from_array(mask)


def _resample_erp_mask(
    mask: SphericalBinaryMask, output_shape_hw: tuple[int, int]
) -> SphericalBinaryMask:
    source = mask.to_array()
    height, width = output_shape_hw
    y, x = np.indices((height, width), dtype=np.float64)
    rays = erp_pixels_to_rays(np.stack((x, y), axis=-1), output_shape_hw)
    pixels = rays_to_erp_pixels(rays, mask.shape_hw).pixels_xy
    source_x = np.floor(pixels[..., 0] + 0.5).astype(np.int64) % mask.shape_hw[1]
    source_y = np.clip(
        np.floor(pixels[..., 1] + 0.5).astype(np.int64), 0, mask.shape_hw[0] - 1
    )
    return SphericalBinaryMask.from_array(source[source_y, source_x])


@dataclass(frozen=True, slots=True)
class SphericalRegionObservation:
    region_id: str
    source_id: str
    face_id: str
    source_kind: str
    center_bearing_xyz: np.ndarray
    boundary_bearings_xyz: np.ndarray
    spec: GnomonicSpec
    erp_shape_hw: tuple[int, int]
    source_support: np.ndarray
    polygon_xy: np.ndarray | None = None
    local_mask: SphericalBinaryMask | None = None
    global_mask: SphericalBinaryMask | None = None
    valid: bool = True
    clipped: bool = False
    score: float | None = None
    class_id: object | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.region_id or not self.source_id or not self.face_id:
            raise ValueError("region_id, source_id, and face_id must be non-empty")
        center = _readonly(self.center_bearing_xyz, dtype=np.float64, ndim=1)
        boundary = _readonly(self.boundary_bearings_xyz, dtype=np.float64, ndim=2)
        if center.shape != (3,) or boundary.shape[1:] != (3,):
            raise ValueError("region bearings must use XYZ unit vectors")
        if not np.isclose(np.linalg.norm(center), 1.0, atol=1e-9):
            raise ValueError("center_bearing_xyz must be a unit vector")
        if len(boundary) and not np.allclose(
            np.linalg.norm(boundary, axis=1), 1.0, atol=1e-9
        ):
            raise ValueError("boundary bearings must be unit vectors")
        support = _boolean_array(self.source_support, "source_support", ndim=2)
        if support.shape != self.spec.output_shape_hw:
            raise ValueError("source_support must match spec.output_shape_hw")
        polygon = None
        if self.polygon_xy is not None:
            polygon = _readonly(self.polygon_xy, dtype=np.float64, ndim=2)
            if polygon.shape[1:] != (2,) or len(polygon) < 3:
                raise ValueError("polygon_xy must have shape (N,2), N>=3")
        valid = _boolean_scalar(self.valid, "valid")
        clipped = _boolean_scalar(self.clipped, "clipped")
        score = _optional_score(self.score)
        class_id = _class_id(self.class_id)
        object.__setattr__(self, "center_bearing_xyz", center)
        object.__setattr__(self, "boundary_bearings_xyz", boundary)
        object.__setattr__(self, "erp_shape_hw", _shape_hw(self.erp_shape_hw))
        object.__setattr__(self, "source_support", support)
        object.__setattr__(self, "polygon_xy", polygon)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "clipped", clipped)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "attributes", _scalar_attributes(self.attributes))

    def to_mask(
        self, evaluation_shape_hw: tuple[int, int] | None = None
    ) -> SphericalBinaryMask:
        shape = (
            self.erp_shape_hw
            if evaluation_shape_hw is None
            else _shape_hw(evaluation_shape_hw)
        )
        if self.global_mask is not None:
            return (
                self.global_mask
                if shape == self.global_mask.shape_hw
                else _resample_erp_mask(self.global_mask, shape)
            )
        if self.local_mask is not None:
            local = self.local_mask.to_array()
        elif self.polygon_xy is not None:
            local = _polygon_mask(self.spec.output_shape_hw, self.polygon_xy)
        else:
            local = np.zeros(self.spec.output_shape_hw, dtype=bool)
        return _backproject_mask(local, self.source_support, self.spec, shape)


def _region_masks(
    observations: tuple[SphericalRegionObservation, ...],
    indices: Sequence[int],
    shape_hw: tuple[int, int],
) -> dict[int, SphericalBinaryMask]:
    """Rasterize regions while retaining only one face plan at a time."""

    masks: dict[int, SphericalBinaryMask] = {}
    grouped: dict[GnomonicSpec, list[int]] = {}
    for index in indices:
        observation = observations[index]
        if observation.global_mask is not None:
            masks[index] = (
                observation.global_mask
                if shape_hw == observation.global_mask.shape_hw
                else _resample_erp_mask(observation.global_mask, shape_hw)
            )
        else:
            grouped.setdefault(observation.spec, []).append(index)
    for spec, group in grouped.items():
        projector = GnomonicProjector(spec, interpolation="nearest")
        output_pixels = shape_hw[0] * shape_hw[1]
        pair_output_bytes = 2 * output_pixels * np.dtype(bool).itemsize
        batch_size = min(
            _REGION_PROJECTION_BATCH_LIMIT,
            max(1, _REGION_PROJECTION_BYTE_BUDGET // pair_output_bytes),
        )
        for start in range(0, len(group), batch_size):
            batch_indices = group[start : start + batch_size]
            local_masks_and_support = []
            for index in batch_indices:
                observation = observations[index]
                if observation.local_mask is not None:
                    local = observation.local_mask.to_array()
                elif observation.polygon_xy is not None:
                    local = _polygon_mask(
                        observation.spec.output_shape_hw, observation.polygon_xy
                    )
                else:
                    local = np.zeros(observation.spec.output_shape_hw, dtype=bool)
                local_masks_and_support.extend((local, observation.source_support))
            if pair_output_bytes > _REGION_PROJECTION_BYTE_BUDGET:
                local, source_support = local_masks_and_support
                projected_local = projector.back_project(local, shape_hw)
                projected_source = projector.back_project(source_support, shape_hw)
                masks[batch_indices[0]] = SphericalBinaryMask.from_array(
                    np.asarray(projected_local.data, dtype=bool)
                    & np.asarray(projected_local.support_mask, dtype=bool)
                    & np.asarray(projected_source.data, dtype=bool)
                    & np.asarray(projected_source.support_mask, dtype=bool)
                )
            else:
                projected = projector.back_project(
                    np.stack(local_masks_and_support, axis=-1), shape_hw
                )
                projected_data = np.asarray(projected.data, dtype=bool)
                projected_support = np.asarray(projected.support_mask, dtype=bool)
                for channel, index in enumerate(batch_indices):
                    masks[index] = SphericalBinaryMask.from_array(
                        projected_data[..., 2 * channel]
                        & projected_data[..., 2 * channel + 1]
                        & projected_support
                    )
    return masks


def _mask_metrics(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    height, width = first.shape
    latitude_north = math.pi / 2 - np.arange(height) * math.pi / height
    latitude_south = math.pi / 2 - (np.arange(height) + 1) * math.pi / height
    row_weights = np.sin(latitude_north) - np.sin(latitude_south)
    weights = np.broadcast_to(row_weights[:, None], (height, width))
    intersection = float(weights[first & second].sum())
    area_first = float(weights[first].sum())
    area_second = float(weights[second].sum())
    union = area_first + area_second - intersection
    iou = intersection / union if union > 0 else 1.0
    smaller = min(area_first, area_second)
    containment = intersection / smaller if smaller > 0 else 1.0
    return iou, containment


def _compact_mask_metrics(
    first: SphericalBinaryMask, second: SphericalBinaryMask
) -> tuple[float, float]:
    """Evaluate one pair while keeping the stored mask catalogue bit-packed."""

    return _mask_metrics(first.to_array(), second.to_array())


def _solid_angle_row_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    latitude_north = math.pi / 2 - np.arange(height) * math.pi / height
    latitude_south = math.pi / 2 - (np.arange(height) + 1) * math.pi / height
    return np.sin(latitude_north) - np.sin(latitude_south)


def _compact_mask_area(mask: SphericalBinaryMask, row_weights: np.ndarray) -> float:
    active = np.unpackbits(
        mask.packed_bits,
        bitorder="little",
        count=mask.shape_hw[0] * mask.shape_hw[1],
    ).reshape(mask.shape_hw)
    active_per_row = np.count_nonzero(active, axis=1)
    return float(np.sum(active_per_row * row_weights))


def _compact_mask_iou(
    first: SphericalBinaryMask,
    second: SphericalBinaryMask,
    row_weights: np.ndarray,
    area_first: float,
    area_second: float,
) -> float:
    """Evaluate exact solid-angle IoU without unpacking both stored masks."""

    intersection_bits = np.bitwise_and(first.packed_bits, second.packed_bits)
    intersection_active = np.unpackbits(
        intersection_bits,
        bitorder="little",
        count=first.shape_hw[0] * first.shape_hw[1],
    ).reshape(first.shape_hw)
    active_per_row = np.count_nonzero(intersection_active, axis=1)
    intersection = float(np.sum(active_per_row * row_weights))
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 1.0


def _compact_mask_meets_iou_threshold(
    first: SphericalBinaryMask,
    second: SphericalBinaryMask,
    row_weights: np.ndarray,
    area_first: float,
    area_second: float,
    threshold: float,
) -> bool:
    """Compare IoU while preserving the historical decision at roundoff ties."""

    approximate = _compact_mask_iou(first, second, row_weights, area_first, area_second)
    height = first.shape_hw[0]
    conservative_roundoff = 256.0 * (height + 2) * np.finfo(np.float64).eps
    if abs(approximate - threshold) <= conservative_roundoff:
        exact, _ = _compact_mask_metrics(first, second)
        return exact >= threshold
    return approximate >= threshold


def _mask_enclosing_cap_radius(
    mask: SphericalBinaryMask,
    center_bearing_xyz: np.ndarray,
) -> float:
    """Return a conservative cap radius for the mask's sampled ERP support.

    The NMS contract is evaluated on ERP pixel centres.  Enclosing every active
    pixel centre therefore gives a safe broad-phase test: two masks that share
    an active ERP pixel cannot have disjoint enclosing caps.  Empty masks use a
    full-sphere cap so that the exact empty-mask semantics remain authoritative.
    """

    height, width = mask.shape_hw
    flat = mask.to_array().reshape(-1)
    minimum_dot = 1.0
    has_active = False
    for start in range(0, flat.size, _MASK_SCAN_CHUNK_PIXELS):
        relative = np.flatnonzero(flat[start : start + _MASK_SCAN_CHUNK_PIXELS])
        if relative.size == 0:
            continue
        has_active = True
        active = start + relative
        rows, columns = np.divmod(active, width)
        pixels_xy = np.column_stack((columns, rows)).astype(np.float64, copy=False)
        rays_xyz = erp_pixels_to_rays(pixels_xy, (height, width))
        dots = np.clip(rays_xyz @ center_bearing_xyz, -1.0, 1.0)
        minimum_dot = min(minimum_dot, float(np.min(dots)))
    if not has_active:
        return math.pi
    return math.acos(minimum_dot)


def _mask_mean_bearing(
    mask: SphericalBinaryMask, fallback_xyz: np.ndarray
) -> np.ndarray:
    """Return a bounded-memory mean of active ERP pixel-centre bearings."""

    height, width = mask.shape_hw
    flat = mask.to_array().reshape(-1)
    vector_sum = np.zeros(3, dtype=np.float64)
    active_count = 0
    for start in range(0, flat.size, _MASK_SCAN_CHUNK_PIXELS):
        relative = np.flatnonzero(flat[start : start + _MASK_SCAN_CHUNK_PIXELS])
        if relative.size == 0:
            continue
        active = start + relative
        rows, columns = np.divmod(active, width)
        pixels_xy = np.column_stack((columns, rows)).astype(np.float64, copy=False)
        vector_sum += erp_pixels_to_rays(pixels_xy, (height, width)).sum(axis=0)
        active_count += relative.size
    norm = float(np.linalg.norm(vector_sum))
    mean_norm = norm / active_count if active_count else 0.0
    # Symmetric ERP lattices accumulate resolution- and chunk-order-dependent
    # residuals of hundreds of epsilons; normalizing them invents a direction.
    if active_count == 0 or mean_norm <= 4096.0 * np.finfo(np.float64).eps:
        return np.asarray(fallback_xyz, dtype=np.float64)
    return vector_sum / norm


def _spherical_cap_candidates(
    center_bearing_xyz: np.ndarray,
    radius_rad: float,
    kept_centers_xyz: np.ndarray,
    kept_radii_rad: np.ndarray,
) -> np.ndarray:
    """Select enclosing caps that may overlap the query cap."""

    if kept_centers_xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    numerical_slack_rad = 1e-12
    combined = np.minimum(math.pi, kept_radii_rad + radius_rad + numerical_slack_rad)
    center_dots = kept_centers_xyz @ center_bearing_xyz
    tolerance = 64.0 * np.finfo(np.float64).eps
    return center_dots + tolerance >= np.cos(combined)


@dataclass(frozen=True, slots=True)
class SphericalRegionSet:
    source_id: str
    observations: tuple[SphericalRegionObservation, ...]
    erp_shape_hw: tuple[int, int]

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise ValueError("source_id must be non-empty")
        if any(item.source_id != self.source_id for item in self.observations):
            raise ValueError("all observations must share source_id")
        shape = _shape_hw(self.erp_shape_hw)
        if any(item.erp_shape_hw != shape for item in self.observations):
            raise ValueError("all region observations must share erp_shape_hw")
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "erp_shape_hw", shape)

    def __len__(self) -> int:
        return len(self.observations)

    def nms(
        self,
        iou_threshold: float,
        *,
        class_aware: bool = True,
        evaluation_shape_hw: tuple[int, int] | None = None,
    ) -> ObservationFusionResult["SphericalRegionSet"]:
        threshold = float(iou_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("iou_threshold must lie within [0,1]")
        shape = (
            self.erp_shape_hw
            if evaluation_shape_hw is None
            else _shape_hw(evaluation_shape_hw)
        )
        valid_indices = [i for i, item in enumerate(self.observations) if item.valid]
        invalid = tuple(i for i, item in enumerate(self.observations) if not item.valid)
        order = sorted(
            valid_indices,
            key=lambda i: (
                *_score_order(self.observations[i].score),
                self.observations[i].face_id,
                i,
            ),
        )
        masks: dict[int, SphericalBinaryMask] = {}
        areas: dict[int, float] = {}
        kept: list[int] = []
        kept_centers = np.empty((len(order), 3), dtype=np.float64)
        kept_radii = np.empty(len(order), dtype=np.float64)
        row_weights = _solid_angle_row_weights(shape)
        suppressed: list[tuple[int, int]] = []
        groups: dict[int, list[int]] = {}
        for index in order:
            item = self.observations[index]
            if threshold > 0:
                masks[index] = item.to_mask(shape)
            area = (
                _compact_mask_area(masks[index], row_weights) if threshold > 0 else 0.0
            )
            radius = (
                _mask_enclosing_cap_radius(masks[index], item.center_bearing_xyz)
                if threshold > 0
                else math.pi
            )
            candidates = _spherical_cap_candidates(
                item.center_bearing_xyz,
                radius,
                kept_centers[: len(kept)],
                kept_radii[: len(kept)],
            )
            if threshold == 0:
                candidates[:] = True
            owner = None
            for kept_position, kept_index in enumerate(kept):
                kept_item = self.observations[kept_index]
                if class_aware and item.class_id != kept_item.class_id:
                    continue
                if not candidates[kept_position]:
                    continue
                if threshold == 0:
                    owner = kept_index
                    break
                meets_threshold = _compact_mask_meets_iou_threshold(
                    masks[index],
                    masks[kept_index],
                    row_weights,
                    area,
                    areas[kept_index],
                    threshold,
                )
                if meets_threshold:
                    owner = kept_index
                    break
            if owner is None:
                kept.append(index)
                kept_centers[len(kept) - 1] = item.center_bearing_xyz
                kept_radii[len(kept) - 1] = radius
                areas[index] = area
                groups[index] = [index]
            else:
                suppressed.append((index, owner))
                groups[owner].append(index)
                masks.pop(index, None)
        output = SphericalRegionSet(
            self.source_id,
            tuple(self.observations[index] for index in kept),
            self.erp_shape_hw,
        )
        return ObservationFusionResult(
            output,
            tuple(kept),
            tuple(sorted(suppressed)),
            invalid,
            tuple(tuple(groups[index]) for index in kept),
            "solid-angle-nms",
            {
                "iou_threshold": threshold,
                "class_aware": class_aware,
                "evaluation_shape_hw": tuple(shape),
            },
        )

    def merge(
        self,
        *,
        iou_threshold: float | None = None,
        containment_threshold: float | None = None,
        class_aware: bool = True,
        evaluation_shape_hw: tuple[int, int] | None = None,
    ) -> ObservationFusionResult["SphericalRegionSet"]:
        if iou_threshold is None and containment_threshold is None:
            raise ValueError("at least one merge threshold is required")
        for name, value in (
            ("iou_threshold", iou_threshold),
            ("containment_threshold", containment_threshold),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must lie within [0,1]")
        shape = (
            self.erp_shape_hw
            if evaluation_shape_hw is None
            else _shape_hw(evaluation_shape_hw)
        )
        valid_indices = [i for i, item in enumerate(self.observations) if item.valid]
        invalid = tuple(i for i, item in enumerate(self.observations) if not item.valid)
        masks = _region_masks(self.observations, valid_indices, shape)
        remaining = set(valid_indices)
        groups: list[tuple[int, ...]] = []
        while remaining:
            seed = min(remaining)
            group = {seed}
            frontier = [seed]
            remaining.remove(seed)
            while frontier:
                left = frontier.pop()
                for right in sorted(tuple(remaining)):
                    if (
                        class_aware
                        and self.observations[left].class_id
                        != self.observations[right].class_id
                    ):
                        continue
                    iou, containment = _compact_mask_metrics(masks[left], masks[right])
                    accepted = (iou_threshold is not None and iou >= iou_threshold) or (
                        containment_threshold is not None
                        and containment >= containment_threshold
                    )
                    if accepted:
                        remaining.remove(right)
                        group.add(right)
                        frontier.append(right)
            groups.append(tuple(sorted(group)))
        merged: list[SphericalRegionObservation] = []
        kept: list[int] = []
        suppressed: list[tuple[int, int]] = []
        for group in groups:
            keeper = sorted(
                group,
                key=lambda i: (
                    *_score_order(self.observations[i].score),
                    self.observations[i].face_id,
                    i,
                ),
            )[0]
            kept.append(keeper)
            suppressed.extend((item, keeper) for item in group if item != keeper)
            packed_union = np.bitwise_or.reduce(
                [masks[item].packed_bits for item in group]
            )
            union_mask = SphericalBinaryMask(shape, packed_union)
            center = _mask_mean_bearing(
                union_mask, self.observations[keeper].center_bearing_xyz
            )
            original = self.observations[keeper]
            observed_scores = [
                self.observations[item].score
                for item in group
                if self.observations[item].score is not None
            ]
            merged.append(
                SphericalRegionObservation(
                    region_id=f"merge:{original.region_id}",
                    source_id=self.source_id,
                    face_id=original.face_id,
                    source_kind="merged",
                    center_bearing_xyz=_readonly(center, dtype=np.float64),
                    boundary_bearings_xyz=np.empty((0, 3), dtype=np.float64),
                    spec=original.spec,
                    erp_shape_hw=shape,
                    source_support=np.ones(original.spec.output_shape_hw, dtype=bool),
                    global_mask=union_mask,
                    valid=True,
                    clipped=any(self.observations[item].clipped for item in group),
                    score=max(observed_scores) if observed_scores else None,
                    class_id=original.class_id,
                    attributes=MappingProxyType(
                        {
                            "source_region_ids": tuple(
                                self.observations[item].region_id for item in group
                            )
                        }
                    ),
                )
            )
        output = SphericalRegionSet(self.source_id, tuple(merged), shape)
        return ObservationFusionResult(
            output,
            tuple(kept),
            tuple(sorted(suppressed)),
            invalid,
            tuple(groups),
            "solid-angle-merge",
            {
                "iou_threshold": iou_threshold,
                "containment_threshold": containment_threshold,
                "class_aware": class_aware,
                "evaluation_shape_hw": tuple(shape),
            },
        )


def lift_face_regions(
    face: Any,
    batch: FaceRegionBatch,
    *,
    source_id: str,
    erp_shape_hw: tuple[int, int] | None = None,
) -> SphericalRegionSet:
    if not str(source_id).strip():
        raise ValueError("source_id must be non-empty")
    spec = face.spec or face.get_geometry().spec
    inherited_shape = face._erp_shape_hw if erp_shape_hw is None else erp_shape_hw
    shape = _shape_hw(inherited_shape, optional=True)
    if shape is None:
        raise ValueError("erp_shape_hw is required for region lifting")
    support = np.asarray(face.get_geometry().support_mask, dtype=bool).copy()
    support.setflags(write=False)
    height, width = spec.output_shape_hw
    observations = []
    for index, geometry in enumerate(batch.geometries):
        clipped = False
        local_mask = None
        global_mask = None
        if batch.kind == "box":
            x0, y0, x1, y1 = geometry.astype(np.float64)
            clipped_values = np.array(
                [
                    np.clip(x0, -0.5, width - 0.5),
                    np.clip(y0, -0.5, height - 0.5),
                    np.clip(x1, -0.5, width - 0.5),
                    np.clip(y1, -0.5, height - 0.5),
                ]
            )
            clipped = not np.array_equal(clipped_values, geometry)
            x0, y0, x1, y1 = clipped_values
            polygon = (
                np.array(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
                if x1 > x0 and y1 > y0
                else None
            )
        elif batch.kind == "polygon":
            original_polygon = geometry.astype(np.float64, copy=True)
            clipped_polygon = _clip_polygon(original_polygon, (height, width))
            clipped = not (
                clipped_polygon.shape == original_polygon.shape
                and np.array_equal(clipped_polygon, original_polygon)
            )
            polygon = clipped_polygon if len(clipped_polygon) >= 3 else None
        else:
            if geometry.shape != (height, width):
                raise ValueError(f"mask region must have face shape {(height, width)}")
            mask = np.asarray(geometry, dtype=bool) & support
            local_mask = SphericalBinaryMask.from_array(mask)
            global_mask = _backproject_mask(mask, support, spec, shape)
            polygon = None
        if polygon is not None:
            boundary = gnomonic_pixels_to_rays(polygon, spec).rays_xyz
            center_pixel = polygon.mean(axis=0, keepdims=True)
            center = gnomonic_pixels_to_rays(center_pixel, spec).rays_xyz[0]
            polygon_readonly = _readonly(polygon, dtype=np.float64)
        elif local_mask is not None:
            rows, columns = np.nonzero(local_mask.to_array())
            if len(rows):
                sample_pixels = np.column_stack((columns, rows)).astype(np.float64)
                rays = gnomonic_pixels_to_rays(sample_pixels, spec).rays_xyz
                center = rays.mean(axis=0)
                center /= np.linalg.norm(center)
            else:
                center = gnomonic_pixels_to_rays(
                    np.array([[(width - 1) / 2, (height - 1) / 2]]), spec
                ).rays_xyz[0]
            boundary = np.empty((0, 3), dtype=np.float64)
            polygon_readonly = None
        else:
            center = gnomonic_pixels_to_rays(
                np.array([[(width - 1) / 2, (height - 1) / 2]]), spec
            ).rays_xyz[0]
            boundary = np.empty((0, 3), dtype=np.float64)
            polygon_readonly = None
        supported_area = (
            bool(global_mask.pixel_count)
            if global_mask is not None
            else bool(
                polygon is not None
                and np.any(_polygon_mask((height, width), polygon) & support)
            )
        )
        attributes = {
            name: np.array(value[index], copy=True)
            for name, value in batch.attributes.items()
        }
        observations.append(
            SphericalRegionObservation(
                region_id=f"{face.face_id}:{index}",
                source_id=str(source_id),
                face_id=face.face_id,
                source_kind=batch.kind,
                center_bearing_xyz=_readonly(center, dtype=np.float64),
                boundary_bearings_xyz=_readonly(boundary, dtype=np.float64),
                spec=spec,
                erp_shape_hw=tuple(shape),
                source_support=support,
                polygon_xy=polygon_readonly,
                local_mask=local_mask,
                global_mask=global_mask,
                valid=bool(batch.valid[index]) and supported_area,
                clipped=clipped,
                score=None if batch.scores is None else float(batch.scores[index]),
                class_id=None
                if batch.class_ids is None
                else _class_id(batch.class_ids[index]),
                attributes=MappingProxyType(attributes),
            )
        )
    return SphericalRegionSet(str(source_id), tuple(observations), tuple(shape))


@dataclass(frozen=True, slots=True)
class FaceViewOutput:
    scores: np.ndarray | None = None
    class_ids: np.ndarray | None = None
    embedding: np.ndarray | None = None
    valid: bool = True
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = _boolean_scalar(self.valid, "valid")
        scores = (
            None
            if self.scores is None
            else _readonly(self.scores, dtype=np.float64, ndim=1)
        )
        class_ids = (
            None if self.class_ids is None else _readonly(self.class_ids, ndim=1)
        )
        if scores is not None:
            if np.any(~np.isfinite(scores)):
                raise ValueError("scores must be finite")
            if class_ids is None or len(class_ids) != len(scores):
                raise ValueError("class_ids must align with scores")
        elif class_ids is not None:
            raise ValueError("class_ids require scores")
        embedding = (
            None
            if self.embedding is None
            else _readonly(self.embedding, dtype=np.float64, ndim=1)
        )
        if embedding is not None and (
            not len(embedding) or np.any(~np.isfinite(embedding))
        ):
            raise ValueError("embedding must be finite and non-empty")
        if scores is None and embedding is None:
            raise ValueError("at least one of scores or embedding is required")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "attributes", _scalar_attributes(self.attributes))

    @classmethod
    def from_torch(
        cls,
        *,
        scores: object | None = None,
        class_ids: object | None = None,
        embedding: object | None = None,
        valid: bool = True,
        attributes: Mapping[str, object] | None = None,
    ) -> "FaceViewOutput":
        return cls(
            scores=_torch_numpy(scores),
            class_ids=_torch_numpy(class_ids),
            embedding=_torch_numpy(embedding),
            valid=_torch_numpy(valid),
            attributes={
                name: _torch_numpy(value) for name, value in (attributes or {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class SphericalViewObservation:
    observation_id: str
    source_id: str
    face_id: str
    spec: GnomonicSpec
    support: np.ndarray
    scores: np.ndarray | None
    class_ids: np.ndarray | None
    embedding: np.ndarray | None
    valid: bool
    attributes: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.observation_id or not self.source_id or not self.face_id:
            raise ValueError("observation_id, source_id, and face_id must be non-empty")
        support = _boolean_array(self.support, "support", ndim=2)
        if support.shape != self.spec.output_shape_hw:
            raise ValueError("support must match spec.output_shape_hw")
        scores = (
            None
            if self.scores is None
            else _readonly(self.scores, dtype=np.float64, ndim=1)
        )
        class_ids = (
            None if self.class_ids is None else _readonly(self.class_ids, ndim=1)
        )
        embedding = (
            None
            if self.embedding is None
            else _readonly(self.embedding, dtype=np.float64, ndim=1)
        )
        if scores is not None and (class_ids is None or len(scores) != len(class_ids)):
            raise ValueError("class_ids must align with scores")
        if scores is None and class_ids is not None:
            raise ValueError("class_ids require scores")
        if scores is not None and np.any(~np.isfinite(scores)):
            raise ValueError("scores must be finite")
        if embedding is not None and (
            not len(embedding) or np.any(~np.isfinite(embedding))
        ):
            raise ValueError("embedding must be finite and non-empty")
        if scores is None and embedding is None:
            raise ValueError("at least one of scores or embedding is required")
        valid = _boolean_scalar(self.valid, "valid")
        if valid and not np.any(support):
            raise ValueError("valid view evidence requires non-empty support")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "attributes", _scalar_attributes(self.attributes))

    def footprint(self, shape_hw: tuple[int, int]) -> SphericalBinaryMask:
        return _backproject_mask(
            self.support, self.support, self.spec, _shape_hw(shape_hw)
        )


@dataclass(frozen=True, slots=True)
class AggregatedViewEvidence:
    source_id: str
    policy: str
    scores: np.ndarray | None
    class_ids: np.ndarray | None
    embedding: np.ndarray | None
    supported_solid_angle_fraction: float
    contributing_view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if self.policy not in {"coverage_mean", "max"}:
            raise ValueError("policy must be 'coverage_mean' or 'max'")
        for name in ("scores", "class_ids", "embedding"):
            value = getattr(self, name)
            if value is not None:
                dtype = np.float64 if name in {"scores", "embedding"} else None
                object.__setattr__(self, name, _readonly(value, dtype=dtype, ndim=1))
        if self.scores is not None:
            if self.class_ids is None or len(self.scores) != len(self.class_ids):
                raise ValueError("class_ids must align with scores")
            if np.any(~np.isfinite(self.scores)):
                raise ValueError("scores must be finite")
        elif self.class_ids is not None:
            raise ValueError("class_ids require scores")
        if self.embedding is not None and (
            not len(self.embedding) or np.any(~np.isfinite(self.embedding))
        ):
            raise ValueError("embedding must be finite and non-empty")
        if self.scores is None and self.embedding is None:
            raise ValueError("at least one of scores or embedding is required")
        if self.policy == "max" and self.embedding is not None:
            raise ValueError("max aggregation is not defined for embeddings")
        _reject_torch(
            self.supported_solid_angle_fraction,
            "supported_solid_angle_fraction",
        )
        fraction = float(self.supported_solid_angle_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("supported_solid_angle_fraction must lie within [0,1]")
        object.__setattr__(self, "supported_solid_angle_fraction", fraction)
        view_ids = tuple(self.contributing_view_ids)
        if not view_ids or any(
            not isinstance(item, str) or not item.strip() for item in view_ids
        ):
            raise ValueError("contributing_view_ids must be non-empty strings")
        object.__setattr__(self, "contributing_view_ids", view_ids)


@dataclass(frozen=True, slots=True)
class SphericalViewSet:
    source_id: str
    observations: tuple[SphericalViewObservation, ...]
    erp_shape_hw: tuple[int, int]

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise ValueError("source_id must be non-empty")
        if any(item.source_id != self.source_id for item in self.observations):
            raise ValueError("all observations must share source_id")
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "erp_shape_hw", _shape_hw(self.erp_shape_hw))

    def aggregate(
        self,
        policy: str,
        *,
        evaluation_shape_hw: tuple[int, int] | None = None,
        normalize_embeddings: bool = True,
    ) -> AggregatedViewEvidence:
        if policy not in {"coverage_mean", "max"}:
            raise ValueError("policy must be 'coverage_mean' or 'max'")
        valid = [item for item in self.observations if item.valid]
        if not valid:
            raise ValueError("no valid view observations to aggregate")
        shape = (
            self.erp_shape_hw
            if evaluation_shape_hw is None
            else _shape_hw(evaluation_shape_hw)
        )
        class_ids = valid[0].class_ids
        if any(
            (item.class_ids is None) != (class_ids is None)
            or (class_ids is not None and not np.array_equal(item.class_ids, class_ids))
            for item in valid
        ):
            raise ValueError("all score vectors must use identical class_ids")
        embeddings = [item.embedding for item in valid if item.embedding is not None]
        score_vectors = [item.scores for item in valid if item.scores is not None]
        if score_vectors and len(score_vectors) != len(valid):
            raise ValueError("scores must be present on every contributing view")
        if embeddings and len(embeddings) != len(valid):
            raise ValueError("embeddings must be present on every contributing view")
        if embeddings and any(item.shape != embeddings[0].shape for item in embeddings):
            raise ValueError("all embeddings must have one dimension")
        height, width = shape
        coverage = np.zeros(shape, dtype=np.uint32)
        for item in valid:
            coverage += item.footprint(shape).to_array()
        latitude_north = math.pi / 2 - np.arange(height) * math.pi / height
        latitude_south = math.pi / 2 - (np.arange(height) + 1) * math.pi / height
        row_area = np.sin(latitude_north) - np.sin(latitude_south)
        supported_area = float(row_area @ np.count_nonzero(coverage, axis=1))
        support_fraction = supported_area / float(width * row_area.sum())
        if policy == "max":
            if embeddings:
                raise ValueError("max aggregation is not defined for embeddings")
            if not score_vectors:
                raise ValueError("max aggregation requires scores")
            scores = np.max(np.stack(score_vectors), axis=0)
        else:
            scores = (
                np.zeros_like(score_vectors[0], dtype=np.float64)
                if score_vectors
                else None
            )
            embedding = (
                np.zeros_like(embeddings[0], dtype=np.float64) if embeddings else None
            )
            chunk_rows = max(1, (8 * 1024 * 1024) // max(8 * width, 1))
            for item in valid:
                footprint = item.footprint(shape).to_array()
                weighted_coverage = 0.0
                for start in range(0, height, chunk_rows):
                    stop = min(height, start + chunk_rows)
                    mask_chunk = footprint[start:stop]
                    fraction = np.zeros(mask_chunk.shape, dtype=np.float64)
                    np.divide(
                        mask_chunk,
                        coverage[start:stop],
                        out=fraction,
                        where=mask_chunk,
                    )
                    weighted_coverage += float(
                        row_area[start:stop] @ fraction.sum(axis=1)
                    )
                view_weight = weighted_coverage / supported_area
                if scores is not None:
                    scores += view_weight * item.scores
                if embedding is not None:
                    vector = item.embedding
                    if normalize_embeddings:
                        norm = np.linalg.norm(vector)
                        if norm <= 1e-12:
                            raise ValueError("cannot normalize a zero embedding")
                        vector = vector / norm
                    embedding += view_weight * vector
            if embedding is not None and normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 1e-12:
                    embedding /= norm
            return AggregatedViewEvidence(
                self.source_id,
                policy,
                None if scores is None else _readonly(scores, dtype=np.float64),
                class_ids,
                None if embedding is None else _readonly(embedding, dtype=np.float64),
                support_fraction,
                tuple(item.face_id for item in valid),
            )
        return AggregatedViewEvidence(
            self.source_id,
            policy,
            _readonly(scores, dtype=np.float64),
            class_ids,
            None,
            support_fraction,
            tuple(item.face_id for item in valid),
        )


def lift_face_view(
    face: Any, output: FaceViewOutput, *, source_id: str
) -> SphericalViewObservation:
    if not str(source_id).strip():
        raise ValueError("source_id must be non-empty")
    spec = face.spec or face.get_geometry().spec
    support = np.asarray(face.get_geometry().support_mask, dtype=bool).copy()
    support.setflags(write=False)
    return SphericalViewObservation(
        observation_id=str(face.face_id),
        source_id=str(source_id),
        face_id=face.face_id,
        spec=spec,
        support=support,
        scores=output.scores,
        class_ids=output.class_ids,
        embedding=output.embedding,
        valid=bool(output.valid) and bool(np.any(support)),
        attributes=output.attributes,
    )
