"""Typed results for sphere-native region analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from types import MappingProxyType
from typing import ClassVar, Mapping

import numpy as np


INTERFACE = "panorai-spherical-region-analysis/v1"


def _validate_interface(value: object) -> None:
    if value != INTERFACE:
        raise ValueError(f"interface must be {INTERFACE!r}")


def _shape(value: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("shape_hw must be a two-integer tuple")
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
        for item in value
    ):
        raise TypeError("shape_hw must be a two-integer tuple")
    result = tuple(int(item) for item in value)
    if any(item < 2 for item in result):
        raise ValueError("shape_hw dimensions must be at least 2")
    return result


def _bearing(value: object, *, optional: bool = False) -> np.ndarray | None:
    if optional and value is None:
        return None
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        raise TypeError("bearing arrays are NumPy-canonical; convert Torch explicitly")
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("bearing must be one finite XYZ vector")
    norm = float(np.linalg.norm(result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("bearing must be a unit vector")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EnclosingSphericalCap:
    """A conservative spherical cap enclosing sampled region centres."""

    center_bearing_xyz: np.ndarray
    radius_deg: float
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        center = _bearing(self.center_bearing_xyz)
        radius = float(self.radius_deg)
        if not math.isfinite(radius) or not 0.0 <= radius <= 180.0:
            raise ValueError("radius_deg must lie within [0, 180]")
        object.__setattr__(self, "center_bearing_xyz", center)
        object.__setattr__(self, "radius_deg", radius)


@dataclass(frozen=True, slots=True)
class RegionMeasurement:
    """Geometry-aware measurements for one positive integer label."""

    label: int
    pixel_count: int
    solid_angle_sr: float
    support_fraction: float
    angular_perimeter_rad: float
    unsupported_boundary_rad: float
    centroid_bearing_xyz: np.ndarray | None
    enclosing_cap: EnclosingSphericalCap
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        if isinstance(self.label, (bool, np.bool_)) or not isinstance(
            self.label, Integral
        ):
            raise TypeError("label must be a positive integer")
        if int(self.label) < 1:
            raise ValueError("label must be a positive integer")
        if isinstance(self.pixel_count, (bool, np.bool_)) or not isinstance(
            self.pixel_count, Integral
        ):
            raise TypeError("pixel_count must be a positive integer")
        if int(self.pixel_count) < 1:
            raise ValueError("pixel_count must be a positive integer")
        for name in (
            "solid_angle_sr",
            "angular_perimeter_rad",
            "unsupported_boundary_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        fraction = float(self.support_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("support_fraction must lie within [0, 1]")
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "pixel_count", int(self.pixel_count))
        object.__setattr__(self, "support_fraction", fraction)
        object.__setattr__(
            self,
            "centroid_bearing_xyz",
            _bearing(self.centroid_bearing_xyz, optional=True),
        )


@dataclass(frozen=True, slots=True)
class RegionMeasurements:
    """Measurements for every positive label in one canonical ERP."""

    shape_hw: tuple[int, int]
    support_solid_angle_sr: float
    regions: tuple[RegionMeasurement, ...]
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        shape = _shape(self.shape_hw)
        support = float(self.support_solid_angle_sr)
        if not math.isfinite(support) or support <= 0.0:
            raise ValueError("support_solid_angle_sr must be finite and positive")
        regions = tuple(self.regions)
        labels = tuple(item.label for item in regions)
        if labels != tuple(sorted(labels)) or len(labels) != len(set(labels)):
            raise ValueError("regions must have unique labels in ascending order")
        object.__setattr__(self, "shape_hw", shape)
        object.__setattr__(self, "support_solid_angle_sr", support)
        object.__setattr__(self, "regions", regions)

    @property
    def by_label(self) -> Mapping[int, RegionMeasurement]:
        return MappingProxyType({item.label: item for item in self.regions})


@dataclass(frozen=True, slots=True)
class RegionAdjacencyEdge:
    """One undirected contact between two positive labels."""

    labels: tuple[int, int]
    shared_boundary_rad: float
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        if (
            not isinstance(self.labels, tuple)
            or len(self.labels) != 2
            or any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
                for item in self.labels
            )
        ):
            raise TypeError("labels must be a two-integer tuple")
        labels = tuple(int(item) for item in self.labels)
        if len(labels) != 2 or labels[0] < 1 or labels[0] >= labels[1]:
            raise ValueError("labels must contain two ascending positive labels")
        length = float(self.shared_boundary_rad)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("shared_boundary_rad must be finite and positive")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "shared_boundary_rad", length)


@dataclass(frozen=True, slots=True)
class RegionAdjacencyGraph:
    """Deterministic region adjacency graph with angular contact lengths."""

    labels: tuple[int, ...]
    edges: tuple[RegionAdjacencyEdge, ...]
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        if not isinstance(self.labels, tuple) or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
            for item in self.labels
        ):
            raise TypeError("labels must be an integer tuple")
        labels = tuple(int(item) for item in self.labels)
        if labels != tuple(sorted(set(labels))) or any(item < 1 for item in labels):
            raise ValueError(
                "labels must be unique positive integers in ascending order"
            )
        edges = tuple(self.edges)
        if any(set(edge.labels) - set(labels) for edge in edges):
            raise ValueError("every edge label must occur in labels")
        if tuple(edge.labels for edge in edges) != tuple(
            sorted(edge.labels for edge in edges)
        ):
            raise ValueError("edges must be sorted by label pair")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True, slots=True)
class SphericalBoundary:
    """Unordered pixel-centre samples of one label's spherical boundary."""

    label: int
    bearings_xyz: np.ndarray
    angular_length_rad: float
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        if isinstance(self.label, (bool, np.bool_)) or not isinstance(
            self.label, Integral
        ):
            raise TypeError("label must be a positive integer")
        if int(self.label) < 1:
            raise ValueError("label must be a positive integer")
        module = type(self.bearings_xyz).__module__
        if module == "torch" or module.startswith("torch."):
            raise TypeError("bearings_xyz is NumPy-canonical; convert Torch explicitly")
        bearings = np.array(self.bearings_xyz, dtype=np.float64, copy=True)
        if bearings.ndim != 2 or bearings.shape[1:] != (3,):
            raise ValueError("bearings_xyz must have shape (N, 3)")
        if not np.isfinite(bearings).all() or (
            len(bearings)
            and not np.allclose(np.linalg.norm(bearings, axis=1), 1.0, atol=1e-9)
        ):
            raise ValueError("bearings_xyz must contain finite unit vectors")
        length = float(self.angular_length_rad)
        if not math.isfinite(length) or length < 0.0:
            raise ValueError("angular_length_rad must be finite and non-negative")
        bearings.setflags(write=False)
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "bearings_xyz", bearings)
        object.__setattr__(self, "angular_length_rad", length)


@dataclass(frozen=True, slots=True)
class SphericalBoundarySet:
    """Boundary samples for each positive label in one ERP."""

    shape_hw: tuple[int, int]
    boundaries: tuple[SphericalBoundary, ...]
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        shape = _shape(self.shape_hw)
        boundaries = tuple(self.boundaries)
        labels = tuple(item.label for item in boundaries)
        if labels != tuple(sorted(labels)) or len(labels) != len(set(labels)):
            raise ValueError("boundaries must have unique labels in ascending order")
        object.__setattr__(self, "shape_hw", shape)
        object.__setattr__(self, "boundaries", boundaries)


@dataclass(frozen=True, slots=True)
class SphericalContour:
    """One ordered contour along ERP cell interfaces on the sphere.

    Vertices are sub-pixel cell-corner bearings in traversal order.  A closed
    contour does not repeat its first vertex at the end; consumers close the
    final geodesic segment explicitly.  Open contours occur only when a label
    boundary meets omitted support or invalid data and
    ``include_support_boundary=False`` was selected by the tracer.
    """

    label: int
    bearings_xyz: np.ndarray
    closed: bool
    angular_length_rad: float
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        if isinstance(self.label, (bool, np.bool_)) or not isinstance(
            self.label, Integral
        ):
            raise TypeError("label must be a positive integer")
        if int(self.label) < 1:
            raise ValueError("label must be a positive integer")
        if not isinstance(self.closed, (bool, np.bool_)):
            raise TypeError("closed must be boolean")
        module = type(self.bearings_xyz).__module__
        if module == "torch" or module.startswith("torch."):
            raise TypeError("bearings_xyz is NumPy-canonical; convert Torch explicitly")
        bearings = np.array(self.bearings_xyz, dtype=np.float64, copy=True)
        if bearings.ndim != 2 or bearings.shape[1:] != (3,):
            raise ValueError("bearings_xyz must have shape (N, 3)")
        minimum = 3 if bool(self.closed) else 2
        if len(bearings) < minimum:
            raise ValueError(
                f"{'closed' if self.closed else 'open'} contours require at least "
                f"{minimum} vertices"
            )
        if not np.isfinite(bearings).all() or not np.allclose(
            np.linalg.norm(bearings, axis=1), 1.0, atol=1e-9
        ):
            raise ValueError("bearings_xyz must contain finite unit vectors")
        length = float(self.angular_length_rad)
        if not math.isfinite(length) or length < 0.0:
            raise ValueError("angular_length_rad must be finite and non-negative")
        bearings.setflags(write=False)
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "closed", bool(self.closed))
        object.__setattr__(self, "bearings_xyz", bearings)
        object.__setattr__(self, "angular_length_rad", length)


@dataclass(frozen=True, slots=True)
class SphericalContourSet:
    """Deterministically ordered contours for positive labels in one ERP."""

    shape_hw: tuple[int, int]
    contours: tuple[SphericalContour, ...]
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        shape = _shape(self.shape_hw)
        contours = tuple(self.contours)
        labels = tuple(item.label for item in contours)
        if labels != tuple(sorted(labels)):
            raise ValueError("contours must be ordered by ascending label")
        object.__setattr__(self, "shape_hw", shape)
        object.__setattr__(self, "contours", contours)


@dataclass(frozen=True, slots=True)
class RegionAnalysisResult:
    """Measurements, adjacency and optional boundaries for one label field."""

    measurements: RegionMeasurements
    adjacency: RegionAdjacencyGraph
    boundaries: SphericalBoundarySet | None = None
    contours: SphericalContourSet | None = None
    interface: str = INTERFACE
    stability: ClassVar[str] = "stable"

    def __post_init__(self) -> None:
        _validate_interface(self.interface)
        labels = tuple(item.label for item in self.measurements.regions)
        if labels != self.adjacency.labels:
            raise ValueError("measurements and adjacency must contain the same labels")
        if self.boundaries is not None:
            boundary_labels = tuple(item.label for item in self.boundaries.boundaries)
            if self.boundaries.shape_hw != self.measurements.shape_hw:
                raise ValueError("boundaries and measurements must share shape_hw")
            if boundary_labels != labels:
                raise ValueError(
                    "boundaries and measurements must contain the same labels"
                )
        if self.contours is not None:
            contour_labels = {item.label for item in self.contours.contours}
            if self.contours.shape_hw != self.measurements.shape_hw:
                raise ValueError("contours and measurements must share shape_hw")
            if contour_labels - set(labels):
                raise ValueError("every contour label must occur in measurements")


__all__ = [
    "EnclosingSphericalCap",
    "RegionAdjacencyEdge",
    "RegionAdjacencyGraph",
    "RegionAnalysisResult",
    "RegionMeasurement",
    "RegionMeasurements",
    "SphericalBoundary",
    "SphericalBoundarySet",
    "SphericalContour",
    "SphericalContourSet",
]
