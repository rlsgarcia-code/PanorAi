"""Pipeline stages for deterministic known-height planar mapping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from panorai.geometry import erp_pixels_to_rays
from panorai.reconstruction import SphericalCameraPose
from panorai.reconstruction.planar import (
    PlaneEstimate,
    PlanarRangeResult,
    PlanarRegion,
    RegisteredPlanarView,
    segment_planar_region,
)

from ._models import (
    KnownHeightPlanarMap,
    KnownHeightPlanarOptions,
    KnownHeightPlanarScene,
    PlanarOrientationEvidence,
    PlanarPreview,
    VIEW_ID,
    _readonly,
)


def _out(value, **details):
    from panorai.pipeline import StageResult

    return StageResult(value, details)


class _KnownHeightStage:
    """Bridge known-height stages to the public ``Stage.execute`` contract."""

    def execute(self, value):
        return self._run(value)


def _normalise(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("cannot normalise a degenerate direction")
    return np.asarray(value, dtype=np.float64) / norm


def _plane(
    plane_id: str,
    normal: np.ndarray,
    offset: float,
    *,
    kind: str,
    diagnostics: dict,
) -> PlaneEstimate:
    normal = _normalise(normal)
    return PlaneEstimate(
        plane_id=plane_id,
        normal_world=_readonly(normal),
        offset_m=float(offset),
        covariance=None,
        tangent_basis=None,
        state="accepted",
        diagnostics=MappingProxyType({"kind": kind, **diagnostics}),
    )


def _accepted_regions(value: KnownHeightPlanarMap) -> dict[str, PlanarRegion]:
    return {
        region.plane_id: region
        for region in value.regions
        if region.state == "accepted" and region.interpretation == "final"
    }


def _accepted_orientation(
    value: KnownHeightPlanarMap, plane_id: str, orientation_id: str | None = None
) -> PlanarOrientationEvidence | None:
    candidates = [
        item
        for item in value.scene.orientations
        if item.plane_id == plane_id
        and item.state == "accepted"
        and (orientation_id is None or item.proposal_id == orientation_id)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _orientation_world(
    value: KnownHeightPlanarMap, evidence: PlanarOrientationEvidence
) -> np.ndarray:
    if value.rotation_world_from_camera is None:
        raise ValueError("reference frame is unavailable")
    return _normalise(
        np.asarray(evidence.normal_camera) @ value.rotation_world_from_camera.T
    )


def _orient_towards_camera(
    normal: np.ndarray, point: np.ndarray, camera_center: np.ndarray
) -> np.ndarray:
    normal = _normalise(normal)
    return -normal if normal @ (camera_center - point) < 0 else normal


def _fit_floor_line_vanishing_direction(
    lines_erp_xy: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[float]]:
    """Intersect interpretation planes from one parallel floor-line family."""

    if len(lines_erp_xy) < 2:
        raise ValueError("each floor line family requires at least two lines")
    rays = erp_pixels_to_rays(lines_erp_xy.reshape(-1, 2), shape).reshape(-1, 2, 3)
    normals = np.cross(rays[:, 0], rays[:, 1])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError("floor line endpoints must define non-degenerate lines")
    normals /= lengths[:, None]
    _, singular, vh = np.linalg.svd(normals, full_matrices=True)
    if len(singular) < 2 or singular[1] < 1e-3:
        raise ValueError("floor lines in one family do not define a vanishing point")
    return _normalise(vh[-1]), singular.tolist()


def _horizontal_normal_from_vanishing(
    vanishing: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[float]]:
    if len(vanishing) < 2:
        raise ValueError("horizontal alignment requires two vanishing directions")
    rays = erp_pixels_to_rays(vanishing, shape)
    _, singular, vh = np.linalg.svd(rays, full_matrices=True)
    if len(singular) < 2 or singular[1] < 1e-3:
        raise ValueError("horizontal vanishing directions are degenerate")
    return _normalise(vh[-1]), singular.tolist()


@dataclass(frozen=True)
class ValidateKnownHeightPlanarScene(_KnownHeightStage):
    input_kind = "known-height-planar-scene"
    output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarScene):
            raise TypeError("expected KnownHeightPlanarScene")
        image = np.asarray(value.panorama.image)
        support = (
            np.ones(image.shape[:2], dtype=bool)
            if value.panorama.support_mask is None
            else np.asarray(value.panorama.support_mask, dtype=bool)
        )
        validity = np.asarray(value.panorama.validity("image"), dtype=bool)
        result = KnownHeightPlanarMap(
            scene=value,
            regions=value.regions,
            constraints=value.constraints,
            support=_readonly(support),
            image_validity=_readonly(validity),
        )
        return _out(
            result,
            scene_id=value.scene_id,
            revision=value.revision,
            units="m",
            frame="local-gravity-aligned",
            regions=len(value.regions),
            constraints=len(value.constraints),
        )


@dataclass(frozen=True)
class ResolvePlanarReferenceFrame(_KnownHeightStage):
    options: KnownHeightPlanarOptions = KnownHeightPlanarOptions()
    input_kind = output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarMap):
            raise TypeError("expected KnownHeightPlanarMap")
        shape = value.support.shape
        down = erp_pixels_to_rays(
            np.asarray(value.scene.down_erp_xy, dtype=np.float64)[None], shape
        )[0]
        up_from_down = -_normalise(down)
        horizon = value.scene.horizon_erp_xy
        vanishing = value.scene.horizontal_vanishing_erp_xy
        line_families = value.scene.floor_line_families_erp_xy
        diagnostics: dict[str, object] = {
            "source": "down-point",
            "horizon_points": int(len(horizon)),
            "horizontal_vanishing_points": int(len(vanishing)),
            "floor_line_family_sizes": [int(len(lines)) for lines in line_families],
        }
        state = "accepted"
        up = up_from_down
        if any(len(lines) for lines in line_families):
            try:
                complete = [lines for lines in line_families if len(lines) >= 2]
                if len(complete) < 2:
                    raise ValueError(
                        "horizontal alignment requires two floor line families"
                    )
                fitted = [
                    _fit_floor_line_vanishing_direction(lines, shape)
                    for lines in complete[:2]
                ]
                directions = np.stack([item[0] for item in fitted])
                candidate = _normalise(np.cross(directions[0], directions[1]))
                if candidate @ up_from_down < 0:
                    candidate = -candidate
                up = candidate
                diagnostics.update(
                    source="floor-line-families",
                    fitted_vanishing_directions_camera=directions.tolist(),
                    line_family_singular_values=[item[1] for item in fitted],
                )
            except ValueError as error:
                state = "needs-review"
                diagnostics["reason"] = "invalid-floor-line-alignment"
                diagnostics["detail"] = str(error)
        elif len(vanishing):
            try:
                candidate, singular = _horizontal_normal_from_vanishing(
                    vanishing, shape
                )
                if candidate @ up_from_down < 0:
                    candidate = -candidate
                up = candidate
                diagnostics.update(
                    source="horizontal-vanishing-points",
                    vanishing_singular_values=singular,
                )
            except ValueError as error:
                state = "needs-review"
                diagnostics["reason"] = "invalid-horizontal-vanishing-points"
                diagnostics["detail"] = str(error)
        elif len(horizon):
            rays = erp_pixels_to_rays(horizon, shape)
            _, singular, vh = np.linalg.svd(rays, full_matrices=True)
            if len(horizon) == 2 and np.linalg.norm(np.cross(rays[0], rays[1])) < 1e-6:
                state = "needs-review"
                diagnostics["reason"] = "degenerate-horizon-observations"
            else:
                candidate = _normalise(vh[-1])
                if candidate @ up_from_down < 0:
                    candidate = -candidate
                disagreement = float(
                    np.degrees(
                        np.arccos(np.clip(candidate @ up_from_down, -1.0, 1.0))
                    )
                )
                up = candidate
                diagnostics.update(
                    source="manual-horizon",
                    horizon_singular_values=singular.tolist(),
                    orientation_disagreement_deg=disagreement,
                )
                if disagreement > self.options.maximum_orientation_disagreement_deg:
                    state = "needs-review"
                    diagnostics["reason"] = "horizon-down-disagreement"

        forward = np.array([0.0, 0.0, 1.0])
        forward -= up * float(up @ forward)
        if np.linalg.norm(forward) < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
            right -= up * float(up @ right)
            right = _normalise(right)
            forward = _normalise(np.cross(right, up))
        else:
            forward = _normalise(forward)
            right = _normalise(np.cross(up, forward))
        rotation = np.stack((right, up, forward))
        center = np.array([0.0, value.scene.camera_height_m, 0.0])
        result = replace(
            value,
            gravity_camera=_readonly(up),
            rotation_world_from_camera=_readonly(rotation),
            camera_center_world=_readonly(center),
            orientation_state=state,
            orientation_diagnostics=MappingProxyType(diagnostics),
        )
        return _out(result, state=state, **diagnostics)


def _seed_roi(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask & support)
    if not len(xs):
        raise ValueError("segmentation requires a supported positive seed")
    shape = mask.shape
    seed_rays = erp_pixels_to_rays(
        np.stack((xs, ys), axis=-1).astype(np.float64), shape
    )
    centre = seed_rays.sum(axis=0)
    if np.linalg.norm(centre) < 1e-8:
        return support.copy()
    centre = _normalise(centre)
    radius = float(
        np.max(np.arccos(np.clip(seed_rays @ centre, -1.0, 1.0)))
        + np.deg2rad(8.0)
    )
    radius = min(np.pi, max(radius, np.deg2rad(8.0)))
    h, w = shape
    all_pixels = np.stack(
        np.meshgrid(np.arange(w), np.arange(h), indexing="xy"), axis=-1
    ).astype(np.float64)
    rays = erp_pixels_to_rays(all_pixels, shape)
    return (rays @ centre >= np.cos(radius)) & support


@dataclass(frozen=True)
class SegmentKnownHeightRegions(_KnownHeightStage):
    options: KnownHeightPlanarOptions = KnownHeightPlanarOptions()
    sam_backend: object | None = None
    input_kind = output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarMap):
            raise TypeError("expected KnownHeightPlanarMap")
        if self.options.segmentation == "sam" and self.sam_backend is None:
            raise ValueError("SAM segmentation requires an explicitly injected backend")
        domain = value.support & value.image_validity
        view = RegisteredPlanarView(
            VIEW_ID,
            np.asarray(value.scene.panorama.image),
            SphericalCameraPose(VIEW_ID, np.eye(3), np.zeros(3)),
            domain,
        )
        floor_ids = {
            constraint.plane_id
            for constraint in value.constraints
            if constraint.kind == "floor-anchor" and constraint.state != "rejected"
        }
        rays = None
        regions: list[PlanarRegion] = []
        alternatives: list[tuple[str, tuple[np.ndarray, ...]]] = []
        for region in value.regions:
            current = region
            if (
                current.interpretation == "seed"
                and current.state != "rejected"
                and self.options.segmentation != "manual"
            ):
                positive = current.positive if current.positive is not None else current.mask
                if current.roi is None:
                    if current.plane_id in floor_ids and value.gravity_camera is not None:
                        if rays is None:
                            h, w = domain.shape
                            pixels = np.stack(
                                np.meshgrid(np.arange(w), np.arange(h), indexing="xy"),
                                axis=-1,
                            ).astype(np.float64)
                            rays = erp_pixels_to_rays(pixels, domain.shape)
                        roi = (rays @ value.gravity_camera < 0) & domain
                        roi_source = "supported-below-horizon"
                    else:
                        roi = _seed_roi(positive, domain)
                        roi_source = "spherical-seed-cap-plus-8deg"
                    current = replace(
                        current,
                        roi=_readonly(roi),
                        history=current.history
                        + (
                            {
                                "action": "prepare-segmentation",
                                "roi": roi_source,
                            },
                        ),
                    )
                segmented, candidates = segment_planar_region(
                    view,
                    current,
                    method=self.options.segmentation,
                    face_size=self.options.face_size,
                    sam_backend=self.sam_backend,
                )
                regions.append(segmented)
                alternatives.append((current.plane_id, tuple(candidates)))
            else:
                regions.append(current)
        result = replace(
            value,
            regions=tuple(regions),
            segmentation_alternatives=tuple(alternatives),
        )
        return _out(
            result,
            method=self.options.segmentation,
            proposals=len(alternatives),
            review_required=bool(alternatives),
        )


@dataclass(frozen=True)
class AnchorFloorPlane(_KnownHeightStage):
    options: KnownHeightPlanarOptions = KnownHeightPlanarOptions()
    input_kind = output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarMap):
            raise TypeError("expected KnownHeightPlanarMap")
        unresolved = dict(value.unresolved)
        planes = list(value.planes)
        accepted = _accepted_regions(value)
        anchors = [
            constraint
            for constraint in value.constraints
            if constraint.kind == "floor-anchor" and constraint.state == "accepted"
        ]
        if len(anchors) != 1:
            unresolved["floor"] = "exactly-one-accepted-floor-anchor-required"
        elif value.orientation_state != "accepted":
            unresolved[anchors[0].plane_id] = "orientation-needs-review"
        elif anchors[0].plane_id not in accepted:
            unresolved[anchors[0].plane_id] = "accepted-floor-mask-required"
        else:
            planes.append(
                _plane(
                    anchors[0].plane_id,
                    np.array([0.0, 1.0, 0.0]),
                    0.0,
                    kind="floor-anchor",
                    diagnostics={
                        "camera_height_m": value.scene.camera_height_m,
                        "camera_height_sigma_m": value.scene.camera_height_sigma_m,
                        "frame": "+Y-up-camera-at-[0,h,0]",
                    },
                )
            )
            unresolved.pop(anchors[0].plane_id, None)
        result = replace(value, planes=tuple(planes), unresolved=MappingProxyType(unresolved))
        return _out(result, resolved=[plane.plane_id for plane in planes])


def _intersect_contact(
    value: KnownHeightPlanarMap,
    points_xy: np.ndarray,
    plane: PlaneEstimate,
    options: KnownHeightPlanarOptions,
) -> np.ndarray:
    rays_camera = erp_pixels_to_rays(points_xy, value.support.shape)
    rays_world = rays_camera @ value.rotation_world_from_camera.T
    normal = np.asarray(plane.normal_world)
    denominator = rays_world @ normal
    numerator = -(value.camera_center_world @ normal + float(plane.offset_m))
    distance = np.full(len(points_xy), np.nan, dtype=np.float64)
    usable = np.abs(denominator) >= options.minimum_incidence
    distance[usable] = numerator / denominator[usable]
    valid = usable & np.isfinite(distance) & (distance > options.minimum_range_m)
    valid &= distance < options.maximum_range_m
    return value.camera_center_world + distance[valid, None] * rays_world[valid]


def _contact_span(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
    direction = vh[0]
    coordinates = (points - centre) @ direction
    return float(np.ptp(coordinates)), centre, direction


@dataclass(frozen=True)
class ResolveConnectedPlanes(_KnownHeightStage):
    options: KnownHeightPlanarOptions = KnownHeightPlanarOptions()
    input_kind = output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarMap):
            raise TypeError("expected KnownHeightPlanarMap")
        planes = {plane.plane_id: plane for plane in value.planes}
        accepted_regions = _accepted_regions(value)
        unresolved = dict(value.unresolved)
        pending = [
            constraint
            for constraint in value.constraints
            if constraint.state == "accepted" and constraint.kind != "floor-anchor"
        ]
        while pending:
            progressed = False
            remaining = []
            for constraint in pending:
                if constraint.plane_id not in accepted_regions:
                    unresolved[constraint.plane_id] = "accepted-mask-required"
                    continue
                reference = planes.get(constraint.reference_plane_id)
                if reference is None:
                    remaining.append(constraint)
                    continue
                if constraint.kind == "coplanar-with":
                    planes[constraint.plane_id] = _plane(
                        constraint.plane_id,
                        reference.normal_world,
                        reference.offset_m,
                        kind="coplanar-with",
                        diagnostics={"reference_plane_id": reference.plane_id},
                    )
                    unresolved.pop(constraint.plane_id, None)
                    progressed = True
                    continue
                points = _intersect_contact(
                    value, constraint.contact_erp_xy, reference, self.options
                )
                orientation = _accepted_orientation(
                    value, constraint.plane_id, constraint.orientation_id
                )
                minimum_points = 1 if constraint.kind in {"parallel-to"} or (
                    constraint.kind == "orthogonal-to" and orientation is not None
                ) else 2
                if len(points) < minimum_points:
                    unresolved[constraint.plane_id] = "insufficient-valid-contact-points"
                    continue
                centre = np.median(points, axis=0)
                direction = None
                span = 0.0
                if len(points) >= 2:
                    span, centre, direction = _contact_span(points)
                    if minimum_points >= 2 and span < self.options.minimum_contact_span_m:
                        unresolved[constraint.plane_id] = "contact-span-too-short"
                        continue
                if constraint.kind == "vertical-to-floor":
                    if reference.diagnostics.get("kind") != "floor-anchor":
                        unresolved[constraint.plane_id] = "vertical-plane-requires-floor-reference"
                        continue
                    horizontal = np.array([direction[0], 0.0, direction[2]])
                    if np.linalg.norm(horizontal) < 1e-8:
                        unresolved[constraint.plane_id] = "degenerate-floor-contact-line"
                        continue
                    horizontal = _normalise(horizontal)
                    normal = _normalise(
                        np.array([horizontal[2], 0.0, -horizontal[0]])
                    )
                    if normal @ (value.camera_center_world - centre) < 0:
                        normal = -normal
                    offset = -float(normal @ centre)
                    residual = np.abs(points @ normal + offset)
                    kind = "vertical-to-floor"
                    orientation_adjustment = None
                elif constraint.kind == "horizontal-to-plane":
                    normal = np.array([0.0, 1.0, 0.0])
                    level = float(np.median(points[:, 1]))
                    offset = -level
                    residual = np.abs(points[:, 1] - level)
                    kind = "horizontal-to-plane"
                    orientation_adjustment = None
                elif constraint.kind == "parallel-to":
                    observed = (
                        _orientation_world(value, orientation)
                        if orientation is not None
                        else np.asarray(reference.normal_world)
                    )
                    if abs(float(observed @ reference.normal_world)) < np.cos(
                        np.deg2rad(self.options.maximum_orientation_disagreement_deg)
                    ):
                        unresolved[constraint.plane_id] = "parallel-orientation-needs-review"
                        continue
                    normal = np.asarray(reference.normal_world)
                    if normal @ observed < 0:
                        normal = -normal
                    normal = _orient_towards_camera(
                        normal, centre, value.camera_center_world
                    )
                    offset = -float(np.median(points @ normal))
                    residual = np.abs(points @ normal + offset)
                    kind = "parallel-to"
                    orientation_adjustment = float(
                        np.degrees(
                            np.arccos(np.clip(abs(observed @ normal), -1.0, 1.0))
                        )
                    )
                elif constraint.kind == "orthogonal-to":
                    reference_normal = np.asarray(reference.normal_world)
                    if orientation is not None:
                        observed = _orientation_world(value, orientation)
                        normal = observed - reference_normal * float(
                            observed @ reference_normal
                        )
                        if np.linalg.norm(normal) < 1e-8:
                            unresolved[constraint.plane_id] = "orthogonal-orientation-degenerate"
                            continue
                        normal = _normalise(normal)
                        if normal @ observed < 0:
                            normal = -normal
                        orientation_adjustment = float(
                            np.degrees(
                                np.arccos(np.clip(abs(observed @ normal), -1.0, 1.0))
                            )
                        )
                        if (
                            orientation_adjustment
                            > self.options.maximum_orientation_disagreement_deg
                        ):
                            unresolved[constraint.plane_id] = "orthogonal-orientation-needs-review"
                            continue
                    else:
                        assert direction is not None
                        normal = np.cross(reference_normal, direction)
                        if np.linalg.norm(normal) < 1e-8:
                            unresolved[constraint.plane_id] = "orthogonal-contact-degenerate"
                            continue
                        normal = _normalise(normal)
                        orientation_adjustment = None
                    normal = _orient_towards_camera(
                        normal, centre, value.camera_center_world
                    )
                    offset = -float(np.median(points @ normal))
                    residual = np.abs(points @ normal + offset)
                    kind = "orthogonal-to"
                else:  # pragma: no cover - PlanarConstraint validates the vocabulary.
                    unresolved[constraint.plane_id] = "unsupported-planar-relation"
                    continue
                maximum_residual = float(np.max(residual))
                if maximum_residual > self.options.maximum_contact_residual_m:
                    unresolved[constraint.plane_id] = "contact-residual-too-large"
                    continue
                planes[constraint.plane_id] = _plane(
                    constraint.plane_id,
                    normal,
                    offset,
                    kind=kind,
                    diagnostics={
                        "reference_plane_id": reference.plane_id,
                        "contact_points": int(len(points)),
                        "contact_span_m": span,
                        "maximum_contact_residual_m": maximum_residual,
                        "orientation_id": constraint.orientation_id,
                        "orientation_adjustment_deg": orientation_adjustment,
                    },
                )
                unresolved.pop(constraint.plane_id, None)
                progressed = True
            if not progressed:
                for constraint in remaining:
                    unresolved.setdefault(
                        constraint.plane_id, "reference-plane-unresolved-or-cycle"
                    )
                break
            pending = remaining
        result = replace(
            value,
            planes=tuple(planes.values()),
            unresolved=MappingProxyType(unresolved),
        )
        return _out(
            result,
            resolved=[plane.plane_id for plane in result.planes],
            unresolved=dict(result.unresolved),
        )


def _preview_footprint(
    value: KnownHeightPlanarMap,
    region: PlanarRegion,
    normal: np.ndarray,
    offset: float,
    options: KnownHeightPlanarOptions,
) -> np.ndarray:
    ys, xs = np.where(np.asarray(region.mask, dtype=bool) & value.support & value.image_validity)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    limit = 5000
    if len(xs) > limit:
        selection = np.linspace(0, len(xs) - 1, limit).astype(int)
        xs, ys = xs[selection], ys[selection]
    pixels = np.stack((xs, ys), axis=-1).astype(np.float64)
    rays_camera = erp_pixels_to_rays(pixels, value.support.shape)
    rays_world = rays_camera @ value.rotation_world_from_camera.T
    denominator = rays_world @ normal
    numerator = -(value.camera_center_world @ normal + offset)
    distance = np.full(len(pixels), np.nan, dtype=np.float64)
    usable = np.abs(denominator) >= options.minimum_incidence
    distance[usable] = numerator / denominator[usable]
    valid = usable & np.isfinite(distance)
    valid &= (distance > options.minimum_range_m) & (distance < options.maximum_range_m)
    return value.camera_center_world + distance[valid, None] * rays_world[valid]


@dataclass(frozen=True)
class ResolvePlanarPreviews(_KnownHeightStage):
    """Assemble explicitly non-authoritative geometry for interactive inspection."""

    options: KnownHeightPlanarOptions = KnownHeightPlanarOptions()
    input_kind = output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarMap):
            raise TypeError("expected KnownHeightPlanarMap")
        if value.rotation_world_from_camera is None or value.camera_center_world is None:
            return _out(value, previews=0, reason="reference-frame-unresolved")
        plane_by_id = {plane.plane_id: plane for plane in value.planes}
        region_by_id = {region.plane_id: region for region in value.regions}
        constraint_by_id = {
            constraint.plane_id: constraint
            for constraint in value.constraints
            if constraint.state != "rejected"
        }
        floor_anchors = [
            constraint
            for constraint in constraint_by_id.values()
            if constraint.kind == "floor-anchor"
        ]
        preview_floor_id = floor_anchors[0].plane_id if len(floor_anchors) == 1 else None
        evidence_by_id = {
            item.plane_id: item
            for item in value.scene.orientations
            if item.state in {"draft", "accepted", "needs-review"}
        }
        plane_ids = sorted(set(region_by_id) | set(evidence_by_id) | set(plane_by_id))
        previews: list[PlanarPreview] = []
        for plane_id in plane_ids:
            resolved = plane_by_id.get(plane_id)
            evidence = evidence_by_id.get(plane_id)
            constraint = constraint_by_id.get(plane_id)
            normal = None
            offset = None
            contacts = np.empty((0, 3), dtype=np.float64)
            state = "orientation-only"
            diagnostics: dict[str, object] = {}
            if resolved is not None:
                normal = np.asarray(resolved.normal_world)
                offset = float(resolved.offset_m)
                state = "resolved"
                diagnostics["source"] = "published-plane"
            elif evidence is not None:
                normal = _orientation_world(value, evidence)
                diagnostics["orientation_state"] = evidence.state
            if (
                resolved is None
                and plane_id == preview_floor_id
                and value.orientation_state == "accepted"
            ):
                # The accepted known-height frame fixes the floor at Y=0 even
                # before its mask and anchor are accepted for publication.
                # This is deliberately a non-authoritative preview: AnchorFloorPlane
                # remains the only stage that may publish the accepted floor.
                normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                offset = 0.0
                state = "positioned-draft"
                diagnostics.update(
                    source="known-height-floor-anchor",
                    constraint_state=constraint.state if constraint is not None else None,
                    camera_height_m=value.scene.camera_height_m,
                )
            if constraint is not None and constraint.reference_plane_id in plane_by_id:
                reference = plane_by_id[constraint.reference_plane_id]
                contacts = _intersect_contact(
                    value, constraint.contact_erp_xy, reference, self.options
                )
                if offset is None and len(contacts):
                    if constraint.kind == "coplanar-with":
                        normal = np.asarray(reference.normal_world)
                        offset = float(reference.offset_m)
                    elif constraint.kind == "parallel-to":
                        normal = np.asarray(reference.normal_world)
                        offset = -float(np.median(contacts @ normal))
                    elif constraint.kind == "orthogonal-to" and normal is not None:
                        reference_normal = np.asarray(reference.normal_world)
                        projected = normal - reference_normal * float(normal @ reference_normal)
                        if np.linalg.norm(projected) >= 1e-8:
                            normal = _normalise(projected)
                            offset = -float(np.median(contacts @ normal))
                    elif constraint.kind in {"vertical-to-floor", "horizontal-to-plane"}:
                        diagnostics["legacy_relation"] = constraint.kind
                    if normal is not None and offset is not None:
                        normal = _orient_towards_camera(
                            normal, np.median(contacts, axis=0), value.camera_center_world
                        )
                        offset = -float(np.median(contacts @ normal))
                        state = "positioned-draft"
            footprint = np.empty((0, 3), dtype=np.float64)
            region = region_by_id.get(plane_id)
            if normal is not None and offset is not None and region is not None:
                footprint = _preview_footprint(
                    value, region, normal, offset, self.options
                )
            previews.append(
                PlanarPreview(
                    plane_id=plane_id,
                    state=state,
                    normal_world=normal,
                    offset_m=offset,
                    contact_world_xyz=contacts,
                    footprint_world_xyz=footprint,
                    source_line_ids=() if evidence is None else evidence.line_ids,
                    source_region_id=None if region is None else region.plane_id,
                    publishable=state == "resolved",
                    diagnostics=diagnostics,
                )
            )
        result = replace(value, previews=tuple(previews))
        return _out(
            result,
            previews=len(previews),
            orientation_only=sum(item.state == "orientation-only" for item in previews),
            positioned_draft=sum(item.state == "positioned-draft" for item in previews),
            resolved=sum(item.state == "resolved" for item in previews),
        )


@dataclass(frozen=True)
class RasterizeKnownHeightRanges(_KnownHeightStage):
    options: KnownHeightPlanarOptions = KnownHeightPlanarOptions()
    input_kind = output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarMap):
            raise TypeError("expected KnownHeightPlanarMap")
        if value.rotation_world_from_camera is None or value.camera_center_world is None:
            return _out(value, valid_pixels=0, reason="reference-frame-unresolved")
        regions = _accepted_regions(value)
        planes = [plane for plane in value.planes if plane.plane_id in regions]
        shape = value.support.shape
        h, w = shape
        pixels = np.stack(
            np.meshgrid(np.arange(w), np.arange(h), indexing="xy"), axis=-1
        ).astype(np.float64)
        rays_camera = erp_pixels_to_rays(pixels, shape)
        rays_world = rays_camera @ value.rotation_world_from_camera.T
        candidate_ranges: list[np.ndarray] = []
        candidate_validity: list[np.ndarray] = []
        for plane in planes:
            normal = np.asarray(plane.normal_world)
            denominator = rays_world @ normal
            numerator = -(
                float(value.camera_center_world @ normal) + float(plane.offset_m)
            )
            distance = np.full(shape, np.nan, dtype=np.float64)
            incidence = np.abs(denominator) >= self.options.minimum_incidence
            distance[incidence] = numerator / denominator[incidence]
            valid = (
                regions[plane.plane_id].mask
                & value.support
                & value.image_validity
                & incidence
                & np.isfinite(distance)
                & (distance > self.options.minimum_range_m)
                & (distance < self.options.maximum_range_m)
            )
            candidate_ranges.append(distance)
            candidate_validity.append(valid)
        count = (
            np.sum(np.stack(candidate_validity), axis=0)
            if candidate_validity
            else np.zeros(shape, dtype=np.int16)
        )
        conflict = count > 1
        ranges = np.full(shape, np.nan, dtype=np.float64)
        validity = np.zeros(shape, dtype=bool)
        uncertainty = np.full(shape, np.nan, dtype=np.float64)
        plane_index = np.full(shape, -1, dtype=np.int32)
        for index, (distance, valid) in enumerate(
            zip(candidate_ranges, candidate_validity, strict=True)
        ):
            publish = valid & ~conflict
            ranges[publish] = distance[publish]
            validity[publish] = True
            plane_index[publish] = index
        sigma = value.scene.camera_height_sigma_m
        if sigma is not None:
            uncertainty[validity] = (
                np.abs(ranges[validity] / value.scene.camera_height_m) * sigma
            )
        source = np.zeros(shape, dtype=np.uint8)
        source[validity] = 2
        output = PlanarRangeResult(
            view_id=VIEW_ID,
            range=_readonly(ranges),
            validity=_readonly(validity),
            uncertainty_m=_readonly(uncertainty),
            plane_index=_readonly(plane_index),
            conflict=_readonly(conflict),
            composed_range=_readonly(ranges),
            composed_validity=_readonly(validity),
            source=_readonly(source),
            plane_ids=tuple(plane.plane_id for plane in planes),
        )
        result = replace(value, ranges=(output,))
        return _out(
            result,
            quantity="radial_range",
            units="m",
            valid_pixels=int(validity.sum()),
            conflict_pixels=int(conflict.sum()),
            uncertainty=(
                "height-linearized"
                if sigma is not None
                else "unavailable-no-height-sigma"
            ),
        )


@dataclass(frozen=True)
class AssembleKnownHeightPlanarMap(_KnownHeightStage):
    input_kind = output_kind = "known-height-planar-map"

    def _run(self, value):
        if not isinstance(value, KnownHeightPlanarMap):
            raise TypeError("expected KnownHeightPlanarMap")
        if value.orientation_state != "accepted":
            state = "needs-review"
        elif value.unresolved:
            state = "partial"
        elif value.ranges and value.ranges[0].validity.any():
            state = "accepted"
        else:
            state = "draft"
        result = replace(value, state=state)
        return _out(
            result,
            state=state,
            planes=len(result.planes),
            unresolved=dict(result.unresolved),
            review_required=state in {"draft", "partial", "needs-review"},
        )


KNOWN_HEIGHT_PLANAR_STAGE_TYPES = (
    ValidateKnownHeightPlanarScene,
    ResolvePlanarReferenceFrame,
    SegmentKnownHeightRegions,
    AnchorFloorPlane,
    ResolveConnectedPlanes,
    ResolvePlanarPreviews,
    RasterizeKnownHeightRanges,
    AssembleKnownHeightPlanarMap,
)
