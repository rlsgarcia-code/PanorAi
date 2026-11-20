"""Typed contracts for known-height planar mapping from one ERP.

The local metric frame is +Y up with the camera centre at ``[0, h, 0]``.
All masks use the native ERP grid and all distances are metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from panorai.data import EquirectangularImage
from panorai.reconstruction.planar import PlaneEstimate, PlanarRangeResult, PlanarRegion


INTERFACE = "panorai-known-height-planar-mapping/v1-experimental"
VIEW_ID = "single-view"


def _readonly(value: Any, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _points(value: Any, *, name: str, minimum: int = 0) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum:
        raise ValueError(f"{name} must be an Nx2 ERP point array")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return _readonly(points)


@dataclass(frozen=True)
class PlanarConstraint:
    """One human-reviewed relation used to position a planar region."""

    constraint_id: str
    plane_id: str
    kind: str
    reference_plane_id: str | None = None
    contact_erp_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    state: str = "proposed"
    source: str = "human"
    orientation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.constraint_id or not self.plane_id:
            raise ValueError("constraint and plane identifiers are required")
        kinds = {
            "floor-anchor",
            "vertical-to-floor",
            "horizontal-to-plane",
            "coplanar-with",
            "orthogonal-to",
            "parallel-to",
        }
        if self.kind not in kinds:
            raise ValueError(f"constraint kind must be one of {sorted(kinds)}")
        if self.state not in {"proposed", "accepted", "rejected"}:
            raise ValueError("constraint state must be proposed, accepted or rejected")
        needs_reference = self.kind != "floor-anchor"
        if needs_reference != (self.reference_plane_id is not None):
            raise ValueError("only non-floor constraints require reference_plane_id")
        if self.reference_plane_id == self.plane_id:
            raise ValueError("a plane cannot reference itself")
        points = _points(self.contact_erp_xy, name="contact_erp_xy")
        if self.orientation_id is not None and not self.orientation_id:
            raise ValueError("orientation_id must be non-empty when supplied")
        if self.kind in {"floor-anchor", "coplanar-with"} and len(points):
            raise ValueError(f"{self.kind} does not consume contact points")
        if (
            self.state == "accepted"
            and self.kind in {"vertical-to-floor", "horizontal-to-plane"}
            and len(points) < 2
        ):
            raise ValueError(
                "accepted connected constraints require two contact points"
            )
        if self.state == "accepted" and self.kind == "parallel-to" and len(points) < 1:
            raise ValueError("accepted parallel constraints require a contact point")
        if self.state == "accepted" and self.kind == "orthogonal-to":
            minimum = 1 if self.orientation_id is not None else 2
            if len(points) < minimum:
                raise ValueError(
                    "accepted orthogonal constraint lacks contact evidence"
                )
        object.__setattr__(self, "contact_erp_xy", points)


@dataclass(frozen=True)
class PlanarOrientationEvidence:
    """Reviewed orientation evidence for one physical plane instance."""

    plane_id: str
    proposal_id: str
    normal_camera: np.ndarray
    line_ids: tuple[str, ...] = ()
    bundle_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    residual_deg: float = 0.0
    state: str = "draft"
    source: str = "automatic"

    def __post_init__(self) -> None:
        if not self.plane_id or not self.proposal_id:
            raise ValueError("orientation evidence requires plane and proposal IDs")
        normal = np.asarray(self.normal_camera, dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if normal.shape != (3,) or not np.isfinite(normal).all() or length < 1e-12:
            raise ValueError("normal_camera must be a finite non-degenerate 3-vector")
        if self.state not in {"draft", "accepted", "rejected", "needs-review"}:
            raise ValueError("invalid orientation evidence state")
        if self.source not in {"automatic", "human"}:
            raise ValueError("orientation source must be automatic or human")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("orientation confidence must lie in [0, 1]")
        if not np.isfinite(self.residual_deg) or self.residual_deg < 0:
            raise ValueError("orientation residual must be nonnegative")
        object.__setattr__(self, "normal_camera", _readonly(normal / length))
        object.__setattr__(self, "line_ids", tuple(str(item) for item in self.line_ids))
        object.__setattr__(
            self, "bundle_ids", tuple(str(item) for item in self.bundle_ids)
        )


@dataclass(frozen=True)
class PlanarPreview:
    """Non-authoritative geometry for incremental inspection."""

    plane_id: str
    state: str
    normal_world: np.ndarray | None = None
    offset_m: float | None = None
    contact_world_xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    footprint_world_xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    source_line_ids: tuple[str, ...] = ()
    source_region_id: str | None = None
    publishable: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plane_id or self.state not in {
            "orientation-only",
            "positioned-draft",
            "resolved",
        }:
            raise ValueError("invalid planar preview identity or state")
        normal = (
            None
            if self.normal_world is None
            else np.asarray(self.normal_world, dtype=np.float64)
        )
        if normal is not None:
            length = float(np.linalg.norm(normal))
            if normal.shape != (3,) or not np.isfinite(normal).all() or length < 1e-12:
                raise ValueError(
                    "preview normal must be a finite non-degenerate 3-vector"
                )
            normal = _readonly(normal / length)
        if self.offset_m is not None and not np.isfinite(self.offset_m):
            raise ValueError("preview offset must be finite when supplied")
        contact = np.asarray(self.contact_world_xyz, dtype=np.float64).reshape(-1, 3)
        footprint = np.asarray(self.footprint_world_xyz, dtype=np.float64).reshape(
            -1, 3
        )
        if not np.isfinite(contact).all() or not np.isfinite(footprint).all():
            raise ValueError("preview geometry must be finite")
        if self.publishable != (self.state == "resolved"):
            raise ValueError("only resolved previews may be publishable")
        object.__setattr__(self, "normal_world", normal)
        object.__setattr__(self, "contact_world_xyz", _readonly(contact))
        object.__setattr__(self, "footprint_world_xyz", _readonly(footprint))
        object.__setattr__(
            self, "source_line_ids", tuple(str(item) for item in self.source_line_ids)
        )
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


@dataclass(frozen=True)
class KnownHeightPlanarOptions:
    """Inspectable numerical and segmentation policy for the application."""

    segmentation: str = "simple"
    face_size: int = 512
    maximum_orientation_disagreement_deg: float = 3.0
    minimum_contact_span_m: float = 0.1
    maximum_contact_residual_m: float = 0.03
    minimum_incidence: float = 0.05
    minimum_range_m: float = 0.1
    maximum_range_m: float = 200.0

    def __post_init__(self) -> None:
        if self.segmentation not in {
            "manual",
            "simple",
            "watershed",
            "classifier",
            "sam",
        }:
            raise ValueError("unsupported planar segmentation method")
        if type(self.face_size) is not int or not 64 <= self.face_size <= 2048:
            raise ValueError("face_size must be an integer in [64, 2048]")
        for name in (
            "maximum_orientation_disagreement_deg",
            "minimum_contact_span_m",
            "maximum_contact_residual_m",
            "minimum_incidence",
            "minimum_range_m",
            "maximum_range_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_orientation_disagreement_deg >= 90:
            raise ValueError("orientation disagreement must be below 90 degrees")
        if self.minimum_incidence >= 1:
            raise ValueError("minimum_incidence must be below one")
        if self.minimum_range_m >= self.maximum_range_m:
            raise ValueError("minimum_range_m must be below maximum_range_m")


@dataclass(frozen=True)
class KnownHeightPlanarScene:
    """One immutable annotation revision for a single central panorama."""

    panorama: EquirectangularImage
    camera_height_m: float
    down_erp_xy: tuple[float, float]
    horizon_erp_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    regions: tuple[PlanarRegion, ...] = ()
    constraints: tuple[PlanarConstraint, ...] = ()
    camera_height_sigma_m: float | None = None
    scene_id: str = "known-height-scene"
    revision: str = "unversioned"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    horizontal_vanishing_erp_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    floor_line_families_erp_xy: tuple[np.ndarray, ...] = ()
    orientations: tuple[PlanarOrientationEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.panorama, EquirectangularImage):
            raise TypeError("panorama must be an EquirectangularImage")
        image = self.panorama.image
        if image is None:
            raise TypeError("panorama must expose an image modality")
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("panorama image must be native HWC RGB uint8")
        support = (
            np.ones(image.shape[:2], dtype=bool)
            if self.panorama.support_mask is None
            else np.asarray(self.panorama.support_mask)
        )
        validity = np.asarray(self.panorama.validity("image"))
        if support.dtype != np.bool_ or support.shape != image.shape[:2]:
            raise ValueError(
                "panorama support must be an explicit native HW boolean mask"
            )
        if validity.dtype != np.bool_ or validity.shape != image.shape[:2]:
            raise ValueError("image validity must be a native HW boolean mask")
        frozen_panorama = EquirectangularImage(
            _readonly(image),
            support_mask=_readonly(support),
            valid=_readonly(validity),
        )
        object.__setattr__(self, "panorama", frozen_panorama)
        if not np.isfinite(self.camera_height_m) or self.camera_height_m <= 0:
            raise ValueError("camera_height_m must be positive and finite")
        if self.camera_height_sigma_m is not None and (
            not np.isfinite(self.camera_height_sigma_m)
            or self.camera_height_sigma_m <= 0
        ):
            raise ValueError("camera_height_sigma_m must be positive when provided")
        down = tuple(float(value) for value in self.down_erp_xy)
        if len(down) != 2 or not np.isfinite(down).all():
            raise ValueError("down_erp_xy must contain two finite coordinates")
        object.__setattr__(self, "down_erp_xy", down)
        horizon = _points(self.horizon_erp_xy, name="horizon_erp_xy")
        if len(horizon) == 1:
            raise ValueError("horizon requires zero or at least two ERP points")
        object.__setattr__(self, "horizon_erp_xy", horizon)
        vanishing = _points(
            self.horizontal_vanishing_erp_xy,
            name="horizontal_vanishing_erp_xy",
        )
        if len(vanishing) == 1:
            raise ValueError(
                "horizontal vanishing evidence requires zero or at least two points"
            )
        line_families: list[np.ndarray] = []
        for index, family in enumerate(self.floor_line_families_erp_xy):
            lines = np.asarray(family, dtype=np.float64)
            if lines.size == 0:
                lines = np.empty((0, 2, 2), dtype=np.float64)
            if lines.ndim != 3 or lines.shape[1:] != (2, 2):
                raise ValueError(
                    f"floor line family {index} must be an Nx2x2 ERP array"
                )
            if not np.isfinite(lines).all():
                raise ValueError("floor line observations must be finite")
            line_families.append(_readonly(lines))
        object.__setattr__(self, "horizontal_vanishing_erp_xy", vanishing)
        object.__setattr__(self, "floor_line_families_erp_xy", tuple(line_families))
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "orientations", tuple(self.orientations))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        if not self.scene_id or not self.revision:
            raise ValueError("scene_id and revision are required")
        h, w = image.shape[:2]

        # Longitude is periodic.  Normalize observations on the antimeridian
        # before validating them so equivalent seam coordinates remain valid.
        def normalize_erp(points: np.ndarray) -> np.ndarray:
            normalized = np.array(points, dtype=np.float64, copy=True)
            if len(normalized):
                normalized[:, 0] = np.mod(normalized[:, 0] + 0.5, w) - 0.5
            return _readonly(normalized)

        down_points = normalize_erp(np.asarray(down, dtype=np.float64)[None])
        down = tuple(float(value) for value in down_points[0])
        horizon = normalize_erp(horizon)
        vanishing = normalize_erp(vanishing)
        line_families = tuple(
            _readonly(normalize_erp(lines.reshape(-1, 2)).reshape(lines.shape))
            for lines in line_families
        )
        constraints = tuple(
            replace(
                constraint,
                contact_erp_xy=normalize_erp(constraint.contact_erp_xy),
            )
            for constraint in self.constraints
        )
        object.__setattr__(self, "down_erp_xy", down)
        object.__setattr__(self, "horizon_erp_xy", horizon)
        object.__setattr__(self, "horizontal_vanishing_erp_xy", vanishing)
        object.__setattr__(self, "floor_line_families_erp_xy", line_families)
        object.__setattr__(self, "constraints", constraints)
        all_points = [np.asarray(down)[None], horizon, vanishing]
        all_points.extend(lines.reshape(-1, 2) for lines in line_families)
        all_points.extend(c.contact_erp_xy for c in constraints)
        for points in all_points:
            if len(points) and (
                np.any(points[:, 0] < -0.5)
                or np.any(points[:, 0] >= w - 0.5)
                or np.any(points[:, 1] < -0.5)
                or np.any(points[:, 1] >= h - 0.5)
            ):
                raise ValueError("ERP observations lie outside the pixel-centre domain")
        region_keys = [(region.plane_id, region.view_id) for region in self.regions]
        if len(region_keys) != len(set(region_keys)):
            raise ValueError("duplicate plane/view region")
        for region in self.regions:
            if region.view_id != VIEW_ID or region.mask.shape != (h, w):
                raise ValueError(
                    f"regions must use view_id={VIEW_ID!r} and the native ERP grid"
                )
            if np.any(region.mask & ~support):
                raise ValueError("region masks must lie within geometric support")
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint IDs must be unique")
        accepted_targets = [
            constraint.plane_id
            for constraint in self.constraints
            if constraint.state == "accepted"
        ]
        if len(accepted_targets) != len(set(accepted_targets)):
            raise ValueError("each plane can have only one accepted constraint")
        orientation_ids = [item.proposal_id for item in self.orientations]
        if len(orientation_ids) != len(set(orientation_ids)):
            raise ValueError("orientation proposal IDs must be unique")
        accepted_orientations = [
            item.plane_id for item in self.orientations if item.state == "accepted"
        ]
        if len(accepted_orientations) != len(set(accepted_orientations)):
            raise ValueError("each plane can have only one accepted orientation")
        known_planes = {region.plane_id for region in self.regions}
        if any(item.plane_id not in known_planes for item in self.orientations):
            raise ValueError("orientation evidence must reference a scene region")
        known_orientation_ids = set(orientation_ids)
        if any(
            c.orientation_id is not None
            and c.orientation_id not in known_orientation_ids
            for c in self.constraints
        ):
            raise ValueError("constraint references unknown orientation evidence")


@dataclass(frozen=True)
class KnownHeightPlanarMap:
    """Application result and immutable intermediate pipeline value."""

    scene: KnownHeightPlanarScene
    regions: tuple[PlanarRegion, ...]
    constraints: tuple[PlanarConstraint, ...]
    support: np.ndarray
    image_validity: np.ndarray
    gravity_camera: np.ndarray | None = None
    rotation_world_from_camera: np.ndarray | None = None
    camera_center_world: np.ndarray | None = None
    orientation_state: str = "unresolved"
    orientation_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    planes: tuple[PlaneEstimate, ...] = ()
    previews: tuple[PlanarPreview, ...] = ()
    ranges: tuple[PlanarRangeResult, ...] = ()
    segmentation_alternatives: tuple[tuple[str, tuple[np.ndarray, ...]], ...] = ()
    unresolved: Mapping[str, str] = field(default_factory=dict)
    state: str = "draft"

    @property
    def interface(self) -> str:
        return INTERFACE
