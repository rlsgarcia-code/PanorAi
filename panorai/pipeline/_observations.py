"""Non-raster results and panorama-frame spherical observations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


NON_RASTER_KINDS = (
    "classification",
    "embedding",
    "detections",
    "keypoints",
)


def _numpy(value: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    if result.dtype == object and dtype is not object:
        raise TypeError(f"{name} must be a regular numeric array")
    return result


def _one_dimensional(
    value: Any | None,
    *,
    name: str,
    count: int,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    result = _numpy(value, name=name, dtype=dtype)
    if result.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    return result.copy()


@dataclass(frozen=True, slots=True)
class ViewResult:
    """One non-raster result with the geometry of its source view."""

    face_id: str
    spec: Any
    value: Any
    support: Any
    validity: Any


class PerViewResults(Sequence[ViewResult]):
    """Ordered non-raster results produced from one gnomonic face set."""

    interface = "panorai-per-view-results/v1-experimental"
    stability = "experimental"

    def __init__(
        self,
        kind: str,
        items: Sequence[ViewResult],
        *,
        source_field: str,
        erp_shape_hw: tuple[int, int],
    ) -> None:
        if kind not in NON_RASTER_KINDS:
            raise ValueError(f"non-raster kind must be one of {NON_RASTER_KINDS}")
        resolved = tuple(items)
        if not resolved:
            raise ValueError("PerViewResults cannot be empty")
        face_ids = [item.face_id for item in resolved]
        if len(set(face_ids)) != len(face_ids):
            raise ValueError("PerViewResults face identifiers must be unique")
        self.kind = kind
        self.items = resolved
        self.source_field = source_field
        self.erp_shape_hw = tuple(int(item) for item in erp_shape_hw)

    def __getitem__(self, index: int) -> ViewResult:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[ViewResult]:
        return iter(self.items)

    @property
    def values(self) -> tuple[Any, ...]:
        return tuple(item.value for item in self.items)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "kind": self.kind,
            "count": len(self),
            "source_field": self.source_field,
            "erp_shape_hw": self.erp_shape_hw,
            "face_ids": [item.face_id for item in self.items],
        }


class _TypedGlobalResult(Sequence[ViewResult]):
    """Classification or embedding with explicit execution scope."""

    kind: str
    interface = "panorai-typed-model-result/v1-experimental"
    stability = "experimental"

    def __init__(
        self,
        value: Any | None = None,
        *,
        items: Sequence[ViewResult] = (),
        source_field: str | None = None,
        erp_shape_hw: tuple[int, int] | None = None,
    ) -> None:
        resolved = tuple(items)
        if (value is None) == (not resolved):
            raise ValueError(
                f"{type(self).__name__} requires exactly one of value or items"
            )
        if resolved:
            face_ids = [item.face_id for item in resolved]
            if len(set(face_ids)) != len(face_ids):
                raise ValueError("per-view result face identifiers must be unique")
            if source_field is None or erp_shape_hw is None:
                raise ValueError(
                    "per-view typed results require source_field and erp_shape_hw"
                )
        self.value = value
        self.items = resolved
        self.source_field = source_field
        self.erp_shape_hw = (
            None if erp_shape_hw is None else tuple(int(item) for item in erp_shape_hw)
        )

    @property
    def is_per_view(self) -> bool:
        return bool(self.items)

    @property
    def values(self) -> tuple[Any, ...]:
        if not self.items:
            raise AttributeError("an aggregated result has one value, not view values")
        return tuple(item.value for item in self.items)

    def __getitem__(self, index: int) -> ViewResult:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "kind": self.kind,
            "scope": "views" if self.is_per_view else "global",
            "count": len(self.items) if self.is_per_view else 1,
            "source_field": self.source_field,
            "erp_shape_hw": self.erp_shape_hw,
            "face_ids": [item.face_id for item in self.items],
            "value_type": (
                type(self.items[0].value).__name__
                if self.items
                else type(self.value).__name__
            ),
        }


class ClassificationResult(_TypedGlobalResult):
    """Typed classification logits, scores, probabilities, or class id."""

    kind = "classification"


class EmbeddingResult(_TypedGlobalResult):
    """Typed global or per-view embedding that preserves autograd in memory."""

    kind = "embedding"


@dataclass(frozen=True, slots=True)
class ViewObservations:
    """Detector-neutral point, box, or polygon observations in one view."""

    geometry: str
    pixels_xy: tuple[np.ndarray, ...]
    scores: np.ndarray
    labels: np.ndarray | None = None
    features: np.ndarray | None = None
    valid: np.ndarray | None = None
    metadata: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.geometry not in {"points", "boxes", "polygons"}:
            raise ValueError("geometry must be 'points', 'boxes', or 'polygons'")
        pixels = tuple(
            _numpy(item, name="pixels_xy", dtype=np.float64).copy()
            for item in self.pixels_xy
        )
        expected_vertices = {"points": 1, "boxes": 4}.get(self.geometry)
        for item in pixels:
            if item.ndim != 2 or item.shape[1] != 2:
                raise ValueError("each observation geometry must have shape (M, 2)")
            if expected_vertices is not None and item.shape[0] != expected_vertices:
                raise ValueError(
                    f"{self.geometry} observations require {expected_vertices} vertices"
                )
            if self.geometry == "polygons" and item.shape[0] < 3:
                raise ValueError("polygon observations require at least three vertices")
        count = len(pixels)
        scores = _one_dimensional(
            self.scores, name="scores", count=count, dtype=np.float64
        )
        assert scores is not None
        if not np.isfinite(scores).all():
            raise ValueError("observation scores must be finite")
        labels = _one_dimensional(self.labels, name="labels", count=count)
        valid = _one_dimensional(self.valid, name="valid", count=count, dtype=bool)
        if valid is None:
            valid = np.ones(count, dtype=bool)
        features = None
        if self.features is not None:
            features = _numpy(self.features, name="features")
            if features.ndim < 2 or features.shape[0] != count:
                raise ValueError("features must have shape (N, ...)")
            features = features.copy()
        metadata = self.metadata or tuple(MappingProxyType({}) for _ in range(count))
        if len(metadata) != count:
            raise ValueError("metadata must contain one mapping per observation")
        object.__setattr__(self, "pixels_xy", pixels)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(
            self,
            "metadata",
            tuple(MappingProxyType(dict(item)) for item in metadata),
        )

    def __len__(self) -> int:
        return len(self.pixels_xy)


def coerce_view_observations(value: Any) -> ViewObservations:
    """Normalize a small detector-neutral mapping into view observations."""

    if isinstance(value, ViewObservations):
        return value
    value_module = type(value).__module__
    if (
        isinstance(value, np.ndarray)
        or value_module == "torch"
        or value_module.startswith("torch.")
    ):
        points = _numpy(value, name="points_xy", dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("point arrays must have shape (N, 2)")
        count = points.shape[0]
        return ViewObservations(
            "points",
            tuple(row.reshape(1, 2) for row in points),
            np.ones(count, dtype=np.float64),
        )
    if not isinstance(value, Mapping):
        raise TypeError(
            "detections/keypoints must return ViewObservations, an (N, 2) point "
            "array, or a mapping with points_xy, boxes_xyxy, or polygons_xy"
        )
    geometry_keys = [
        name for name in ("points_xy", "boxes_xyxy", "polygons_xy") if name in value
    ]
    if len(geometry_keys) != 1:
        raise ValueError(
            "observation mappings require exactly one of points_xy, boxes_xyxy, "
            "or polygons_xy"
        )
    geometry_key = geometry_keys[0]
    if geometry_key == "points_xy":
        array = _numpy(value[geometry_key], name=geometry_key, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("points_xy must have shape (N, 2)")
        geometry = "points"
        pixels = tuple(row.reshape(1, 2) for row in array)
    elif geometry_key == "boxes_xyxy":
        array = _numpy(value[geometry_key], name=geometry_key, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("boxes_xyxy must have shape (N, 4)")
        if np.any(array[:, 2:] < array[:, :2]):
            raise ValueError("boxes_xyxy must satisfy xmax>=xmin and ymax>=ymin")
        geometry = "boxes"
        pixels = tuple(
            np.asarray(
                ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)),
                dtype=np.float64,
            )
            for xmin, ymin, xmax, ymax in array
        )
    else:
        geometry = "polygons"
        pixels = tuple(
            _numpy(item, name="polygons_xy", dtype=np.float64)
            for item in value[geometry_key]
        )
    count = len(pixels)
    scores = value.get("scores", np.ones(count, dtype=np.float64))
    return ViewObservations(
        geometry,
        pixels,
        scores,
        labels=value.get("labels"),
        features=value.get("features", value.get("descriptors")),
        valid=value.get("valid"),
        metadata=tuple(value.get("metadata", ())),
    )


class SphericalObservations(Sequence[np.ndarray]):
    """Observations expressed as unit bearings in the panorama frame.

    ``bearings_xyz`` contains one representative bearing per observation.
    ``footprints_xyz`` retains every point, box corner, or polygon vertex.
    Geometric support, source-data validity, and final observation validity
    remain separate arrays.
    """

    interface = "panorai-spherical-observations/v1-experimental"
    stability = "experimental"

    def __init__(
        self,
        bearings_xyz: Any,
        *,
        footprints_xyz: Sequence[Any],
        scores: Any,
        support: Any,
        data_validity: Any,
        valid: Any,
        source_face_ids: Any,
        source_indices: Any,
        source_pixels_xy: Sequence[Any],
        geometry: str,
        labels: Any | None = None,
        features: Any | None = None,
        metadata: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if geometry not in {"points", "boxes", "polygons"}:
            raise ValueError("geometry must be 'points', 'boxes', or 'polygons'")
        bearings = _numpy(bearings_xyz, name="bearings_xyz", dtype=np.float64)
        if bearings.ndim != 2 or bearings.shape[1] != 3:
            raise ValueError("bearings_xyz must have shape (N, 3)")
        count = bearings.shape[0]
        footprints = tuple(
            _numpy(item, name="footprints_xyz", dtype=np.float64).copy()
            for item in footprints_xyz
        )
        pixels = tuple(
            _numpy(item, name="source_pixels_xy", dtype=np.float64).copy()
            for item in source_pixels_xy
        )
        if len(footprints) != count or len(pixels) != count:
            raise ValueError("footprints and source pixels must match bearing count")
        for footprint, source_pixels in zip(footprints, pixels, strict=True):
            if footprint.ndim != 2 or footprint.shape[1] != 3:
                raise ValueError("each spherical footprint must have shape (M, 3)")
            if source_pixels.shape != (footprint.shape[0], 2):
                raise ValueError("source pixels and footprint vertices must agree")
        scores_array = _one_dimensional(
            scores, name="scores", count=count, dtype=np.float64
        )
        support_array = _one_dimensional(
            support, name="support", count=count, dtype=bool
        )
        data_validity_array = _one_dimensional(
            data_validity, name="data_validity", count=count, dtype=bool
        )
        valid_array = _one_dimensional(valid, name="valid", count=count, dtype=bool)
        face_ids = _one_dimensional(
            source_face_ids, name="source_face_ids", count=count, dtype=object
        )
        indices = _one_dimensional(
            source_indices, name="source_indices", count=count, dtype=np.int64
        )
        assert scores_array is not None
        assert support_array is not None
        assert data_validity_array is not None
        assert valid_array is not None
        assert face_ids is not None
        assert indices is not None
        if not np.isfinite(scores_array).all():
            raise ValueError("scores must be finite")
        if np.any(indices < 0):
            raise ValueError("source_indices must be non-negative")
        if np.any(valid_array & ~(support_array & data_validity_array)):
            raise ValueError("valid observations must have support and data validity")
        labels_array = _one_dimensional(labels, name="labels", count=count)
        features_array = None
        if features is not None:
            features_array = _numpy(features, name="features")
            if features_array.ndim < 2 or features_array.shape[0] != count:
                raise ValueError("features must have shape (N, ...)")
            features_array = features_array.copy()
        if bearings.size:
            norms = np.linalg.norm(bearings, axis=1)
            finite_valid = np.isfinite(bearings).all(axis=1) & (norms > 0)
            if np.any(valid_array & ~finite_valid):
                raise ValueError("valid observations require finite non-zero bearings")
            bearings[finite_valid] /= norms[finite_valid, None]
        resolved_metadata = metadata or tuple(
            MappingProxyType({}) for _ in range(count)
        )
        if len(resolved_metadata) != count:
            raise ValueError("metadata must contain one mapping per observation")
        self.bearings_xyz = bearings.copy()
        self.footprints_xyz = footprints
        self.scores = scores_array
        self.support = support_array
        self.data_validity = data_validity_array
        self.valid = valid_array
        self.source_face_ids = face_ids
        self.source_indices = indices
        self.source_pixels_xy = pixels
        self.geometry = geometry
        self.labels = labels_array
        self.features = features_array
        self.metadata = tuple(
            MappingProxyType(dict(item)) for item in resolved_metadata
        )

    def __getitem__(self, index: int) -> np.ndarray:
        return self.bearings_xyz[index]

    def __len__(self) -> int:
        return int(self.bearings_xyz.shape[0])

    @property
    def bearings(self) -> np.ndarray:
        return self.bearings_xyz

    def take(self, indices: Any) -> "SphericalObservations":
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("selected observation indices must have shape (N,)")
        return SphericalObservations(
            self.bearings_xyz[selected],
            footprints_xyz=tuple(self.footprints_xyz[int(index)] for index in selected),
            scores=self.scores[selected],
            support=self.support[selected],
            data_validity=self.data_validity[selected],
            valid=self.valid[selected],
            source_face_ids=self.source_face_ids[selected],
            source_indices=self.source_indices[selected],
            source_pixels_xy=tuple(
                self.source_pixels_xy[int(index)] for index in selected
            ),
            geometry=self.geometry,
            labels=None if self.labels is None else self.labels[selected],
            features=None if self.features is None else self.features[selected],
            metadata=tuple(self.metadata[int(index)] for index in selected),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "stability": self.stability,
            "geometry": self.geometry,
            "count": len(self),
            "support_count": int(self.support.sum()),
            "data_valid_count": int(self.data_validity.sum()),
            "valid_count": int(self.valid.sum()),
            "face_ids": sorted({str(item) for item in self.source_face_ids}),
            "has_labels": self.labels is not None,
            "feature_shape": (
                None if self.features is None else tuple(self.features.shape)
            ),
        }


__all__ = [
    "ClassificationResult",
    "EmbeddingResult",
    "NON_RASTER_KINDS",
    "PerViewResults",
    "SphericalObservations",
    "ViewObservations",
    "ViewResult",
]
