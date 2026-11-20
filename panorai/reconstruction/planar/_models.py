"""Immutable contracts for human-guided planar reconstruction (NumPy only).

Coordinates follow geometry-v1; all metric quantities are metres. Camera
poses satisfy x_camera = R @ (X_world - C). Masks never imply validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from panorai.reconstruction import SphericalCameraPose

INTERFACE = "panorai-human-guided-planar/v1-experimental"


def readonly(value: Any, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def boolean(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return readonly(array)


@dataclass(frozen=True)
class RegisteredPlanarView:
    view_id: str
    image: np.ndarray
    pose: SphericalCameraPose
    support: np.ndarray
    baseline_range: np.ndarray | None = None
    baseline_validity: np.ndarray | None = None

    def __post_init__(self):
        image = np.asarray(self.image)
        if not self.view_id or self.pose.panorama_id != self.view_id:
            raise ValueError("view_id must equal pose.panorama_id")
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("image must be native HWC RGB uint8")
        object.__setattr__(self, "image", readonly(image))
        object.__setattr__(
            self, "support", boolean(self.support, image.shape[:2], "support")
        )
        if (self.baseline_range is None) != (self.baseline_validity is None):
            raise ValueError("baseline range requires explicit baseline_validity")
        if self.baseline_range is not None:
            ranges = np.asarray(self.baseline_range)
            valid = boolean(
                self.baseline_validity, image.shape[:2], "baseline_validity"
            )
            if ranges.shape != valid.shape or np.any(
                valid & (~np.isfinite(ranges) | (ranges <= 0))
            ):
                raise ValueError(
                    "baseline must contain positive finite ranges where valid"
                )
            if np.any(valid & ~self.support):
                raise ValueError("baseline validity must lie within support")
            object.__setattr__(self, "baseline_range", readonly(ranges))
            object.__setattr__(self, "baseline_validity", valid)


@dataclass(frozen=True)
class PlanarRegion:
    plane_id: str
    view_id: str
    mask: np.ndarray
    state: str = "draft"
    label: str = "plane"
    positive: np.ndarray | None = None
    negative: np.ndarray | None = None
    roi: np.ndarray | None = None
    interpretation: str = "final"
    history: tuple[Mapping[str, Any], ...] = ()
    source: str = "human"

    def __post_init__(self):
        values = np.asarray(self.mask)
        if not self.plane_id or not self.view_id or values.ndim != 2:
            raise ValueError("plane/view identifiers and HW mask required")
        object.__setattr__(self, "mask", boolean(values, values.shape, "mask"))
        if self.state not in {"draft", "accepted", "rejected", "needs-review"}:
            raise ValueError("invalid review state")
        if self.interpretation not in {"seed", "final"}:
            raise ValueError("interpretation must be seed or final")
        if self.state == "accepted" and self.interpretation != "final":
            raise ValueError("accepted regions must be final masks, not seeds")
        for name in ("positive", "negative", "roi"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, boolean(getattr(self, name), values.shape, name)
                )
        if (
            self.positive is not None
            and self.negative is not None
            and np.any(self.positive & self.negative)
        ):
            raise ValueError("positive and negative strokes must not overlap")
        object.__setattr__(
            self,
            "history",
            tuple(MappingProxyType(dict(item)) for item in self.history),
        )


@dataclass(frozen=True)
class PlanarPointObservation:
    view_id: str
    pixel_xy: tuple[float, float]
    weight: float = 1.0

    def __post_init__(self):
        pixel = tuple(float(v) for v in self.pixel_xy)
        if len(pixel) != 2 or not np.isfinite(pixel).all() or not self.view_id:
            raise ValueError("observation requires view_id and finite pixel_xy")
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("observation weight must be positive")
        object.__setattr__(self, "pixel_xy", pixel)


@dataclass(frozen=True)
class PlaneCorrespondence:
    plane_id: str
    track_id: str
    observations: tuple[PlanarPointObservation, ...]
    source: str = "human"

    def __post_init__(self):
        observations = tuple(self.observations)
        if not self.plane_id or not self.track_id or len(observations) < 2:
            raise ValueError(
                "correspondence requires identifiers and at least two views"
            )
        if len({o.view_id for o in observations}) != len(observations):
            raise ValueError("one observation per view per track required")
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True)
class RegisteredPlanarScene:
    views: tuple[RegisteredPlanarView, ...]
    regions: tuple[PlanarRegion, ...] = ()
    correspondences: tuple[PlaneCorrespondence, ...] = ()
    scene_id: str = "scene"
    revision: str = "unversioned"
    units: str = "m"

    def __post_init__(self):
        for name in ("views", "regions", "correspondences"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.units != "m" or len(self.views) < 2:
            raise ValueError(
                "at least two metrically registered views in metres required"
            )
        views = {v.view_id: v for v in self.views}
        if len(views) != len(self.views):
            raise ValueError("view IDs must be unique")
        keys = [(r.plane_id, r.view_id) for r in self.regions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate plane/view observation")
        if len({(t.plane_id, t.track_id) for t in self.correspondences}) != len(
            self.correspondences
        ):
            raise ValueError("duplicate plane/track identifier")
        for region in self.regions:
            if (
                region.view_id not in views
                or region.mask.shape != views[region.view_id].support.shape
            ):
                raise ValueError("region must align with registered native ERP")
            if np.any(region.mask & ~views[region.view_id].support):
                raise ValueError("region must lie within explicit support")
        for track in self.correspondences:
            for observation in track.observations:
                if observation.view_id not in views:
                    raise ValueError("unknown correspondence view")
                h, w = views[observation.view_id].support.shape
                x, y = observation.pixel_xy
                if not (-0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5):
                    raise ValueError(
                        "observation outside native ERP pixel-centre domain"
                    )


@dataclass(frozen=True)
class PlanarReconstructionOptions:
    automatic_matching: bool = True
    feature_face_size: int = 1024
    max_features: int = 8000
    minimum_parallax_deg: float = 0.5
    maximum_reprojection_deg: float = 0.5
    plane_threshold_m: float = 0.03
    minimum_fit_tracks: int = 6
    ransac_trials: int = 500
    random_seed: int = 7
    holdout_fraction: float = 0.2
    minimum_incidence: float = 0.05
    minimum_range_m: float = 0.1
    maximum_range_m: float = 200.0
    maximum_range_sigma_m: float = 0.25
    block_rows: int = 128

    def __post_init__(self):
        if not isinstance(self.automatic_matching, bool):
            raise TypeError("automatic_matching must be boolean")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        for name in (
            "minimum_parallax_deg",
            "maximum_reprojection_deg",
            "plane_threshold_m",
            "minimum_incidence",
            "minimum_range_m",
            "maximum_range_m",
            "maximum_range_sigma_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "feature_face_size",
            "max_features",
            "minimum_fit_tracks",
            "ransac_trials",
            "block_rows",
        ):
            if (
                not isinstance(getattr(self, name), int)
                or isinstance(getattr(self, name), bool)
                or getattr(self, name) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_fit_tracks < 3 or not 0 <= self.holdout_fraction < 0.5:
            raise ValueError(
                "at least three fit tracks and holdout_fraction in [0,.5) required"
            )
        if (
            not self.minimum_range_m < self.maximum_range_m
            or self.minimum_incidence >= 1
        ):
            raise ValueError("invalid range bounds or minimum_incidence")


@dataclass(frozen=True)
class TriangulatedPlaneTrack:
    correspondence: PlaneCorrespondence
    point_world: np.ndarray | None
    sigma_m: float
    parallax_deg: float
    residual_deg: float
    reason: str | None = None


@dataclass(frozen=True)
class PlaneEstimate:
    plane_id: str
    normal_world: np.ndarray | None
    offset_m: float | None
    covariance: np.ndarray | None
    tangent_basis: np.ndarray | None
    state: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    inlier_track_ids: tuple[str, ...] = ()
    heldout_track_ids: tuple[str, ...] = ()
    covariance_origin_world: np.ndarray | None = None


@dataclass(frozen=True)
class PlanarRangeResult:
    view_id: str
    range: np.ndarray
    validity: np.ndarray
    uncertainty_m: np.ndarray
    plane_index: np.ndarray
    conflict: np.ndarray
    composed_range: np.ndarray
    composed_validity: np.ndarray
    source: np.ndarray  # 0 invalid, 1 baseline, 2 fitted plane
    plane_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlanarWorkspace:
    scene: RegisteredPlanarScene
    correspondences: tuple[PlaneCorrespondence, ...] = ()
    tracks: tuple[TriangulatedPlaneTrack, ...] = ()
    planes: tuple[PlaneEstimate, ...] = ()
    proposals: tuple[PlanarRegion, ...] = ()
    ranges: tuple[PlanarRangeResult, ...] = ()
    segmentation_alternatives: tuple[tuple[str, str, tuple[np.ndarray, ...]], ...] = ()

    @property
    def interface(self):
        return INTERFACE
