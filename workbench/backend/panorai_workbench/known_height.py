"""Workbench I/O for the public known-height planar application.

This module only translates persisted user revisions to public PanorAi values
and translates the public result back to presentation artifacts.  All metric
geometry is executed by :class:`panorai.reconstruction.known_height_planar.KnownHeightPlanarMapping`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from panorai.data import EquirectangularImage
from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.reconstruction.known_height_planar import (
    KnownHeightPlanarMapping,
    KnownHeightPlanarOptions,
    KnownHeightPlanarScene,
    PlanarConstraint,
    PlanarOrientationEvidence,
    preview_planar_relation,
)
from panorai.reconstruction.planar import (
    PlaneEstimate,
    PlanarRegion,
    decode_planar_mask,
    region_document,
)


def preview_known_height_relation(document: dict[str, Any]) -> dict[str, Any]:
    """Translate one hover request to the public scientific preview operation."""

    reference = document["reference_plane"]
    plane = PlaneEstimate(
        plane_id=str(reference["plane_id"]),
        normal_world=np.asarray(reference["normal_world"], dtype=np.float64),
        offset_m=float(reference["offset_m"]),
        covariance=None,
        tangent_basis=None,
        state="accepted",
    )
    result = preview_planar_relation(
        plane_id=str(document["plane_id"]),
        relation=str(document["relation"]),
        reference_plane=plane,
        cursor_erp_xy=tuple(document["cursor_erp_xy"]),
        erp_shape_hw=tuple(document["erp_shape_hw"]),
        rotation_world_from_camera=np.asarray(
            document["rotation_world_from_camera"], dtype=np.float64
        ),
        camera_center_world=np.asarray(document["camera_center_world"], dtype=np.float64),
        candidate_normal_world=None
        if document.get("candidate_normal_world") is None
        else np.asarray(document["candidate_normal_world"], dtype=np.float64),
        candidate_normal_camera=None
        if document.get("candidate_normal_camera") is None
        else np.asarray(document["candidate_normal_camera"], dtype=np.float64),
    )
    return {
        "schema": "panorai-planar-relation-preview/v1",
        "plane_id": result.plane_id,
        "state": result.state,
        "normal_world": result.normal_world.tolist(),
        "offset_m": result.offset_m,
        "contact_world_xyz": result.contact_world_xyz.tolist(),
        "publishable": result.publishable,
        "diagnostics": dict(result.diagnostics),
    }


def _region(document: dict[str, Any], shape: tuple[int, int]) -> PlanarRegion:
    values: dict[str, Any] = {
        "plane_id": document["plane_id"],
        "view_id": "single-view",
        "label": document.get("label", document["plane_id"]),
        "state": document.get("state", "draft"),
        "mask": decode_planar_mask(document["mask"], shape),
        "interpretation": document.get("interpretation", "final"),
        "history": tuple(document.get("history", ())),
        "source": document.get("source", "workbench-human"),
    }
    for name in ("positive", "negative", "roi"):
        if document.get(name) is not None:
            values[name] = decode_planar_mask(document[name], shape)
    return PlanarRegion(**values)


def _constraint(document: dict[str, Any]) -> PlanarConstraint:
    return PlanarConstraint(
        constraint_id=document["constraint_id"],
        plane_id=document["plane_id"],
        kind=document["kind"],
        reference_plane_id=document.get("reference_plane_id"),
        contact_erp_xy=np.asarray(document.get("contact_erp_xy", ())),
        state=document.get("state", "proposed"),
        source=document.get("source", "workbench-human"),
        orientation_id=document.get("orientation_id"),
    )


def _orientation(document: dict[str, Any]) -> PlanarOrientationEvidence:
    return PlanarOrientationEvidence(
        plane_id=document["plane_id"],
        proposal_id=document["proposal_id"],
        normal_camera=np.asarray(document["normal_camera"], dtype=np.float64),
        line_ids=tuple(document.get("line_ids", ())),
        bundle_ids=tuple(document.get("bundle_ids", ())),
        confidence=float(document.get("confidence", 0.0)),
        residual_deg=float(document.get("residual_deg", 0.0)),
        state=document.get("state", "draft"),
        source=document.get("source", "automatic"),
    )


def _scene_document(result) -> dict[str, Any]:
    return {
        "schema": "panorai-known-height-planar-scene/v2",
        "scene_id": result.scene.scene_id,
        "revision": result.scene.revision,
        "camera_height_m": result.scene.camera_height_m,
        "camera_height_sigma_m": result.scene.camera_height_sigma_m,
        "down_erp_xy": list(result.scene.down_erp_xy),
        "horizon_erp_xy": result.scene.horizon_erp_xy.tolist(),
        "horizontal_vanishing_erp_xy": (
            result.scene.horizontal_vanishing_erp_xy.tolist()
        ),
        "floor_line_families_erp_xy": [
            family.tolist() for family in result.scene.floor_line_families_erp_xy
        ],
        "regions": [region_document(region) for region in result.regions],
        "constraints": [
            {
                "constraint_id": item.constraint_id,
                "plane_id": item.plane_id,
                "kind": item.kind,
                "reference_plane_id": item.reference_plane_id,
                "contact_erp_xy": item.contact_erp_xy.tolist(),
                "state": item.state,
                "source": item.source,
                "orientation_id": item.orientation_id,
            }
            for item in result.constraints
        ],
        "orientations": [
            {
                "plane_id": item.plane_id,
                "proposal_id": item.proposal_id,
                "normal_camera": item.normal_camera.tolist(),
                "line_ids": list(item.line_ids),
                "bundle_ids": list(item.bundle_ids),
                "confidence": item.confidence,
                "residual_deg": item.residual_deg,
                "state": item.state,
                "source": item.source,
            }
            for item in result.scene.orientations
        ],
        "provenance": dict(result.scene.provenance),
    }


def _summary(result, manifest: dict[str, Any]) -> dict[str, Any]:
    ranges = result.ranges[0] if result.ranges else None
    return {
        "schema": "panorai-known-height-planar-result/v1",
        "interface": result.interface,
        "state": result.state,
        "orientation": {
            "state": result.orientation_state,
            "diagnostics": dict(result.orientation_diagnostics),
            "gravity_camera": None
            if result.gravity_camera is None
            else result.gravity_camera.tolist(),
            "rotation_world_from_camera": None
            if result.rotation_world_from_camera is None
            else result.rotation_world_from_camera.tolist(),
            "camera_center_world": None
            if result.camera_center_world is None
            else result.camera_center_world.tolist(),
        },
        "planes": [
            {
                "plane_id": plane.plane_id,
                "normal_world": plane.normal_world.tolist(),
                "offset_m": plane.offset_m,
                "state": plane.state,
                "diagnostics": dict(plane.diagnostics),
            }
            for plane in result.planes
        ],
        "regions": [region_document(region) for region in result.regions],
        "constraints": _scene_document(result)["constraints"],
        "previews": [
            {
                "plane_id": item.plane_id,
                "state": item.state,
                "normal_world": None if item.normal_world is None else item.normal_world.tolist(),
                "offset_m": item.offset_m,
                "contact_world_xyz": item.contact_world_xyz.tolist(),
                "footprint_world_xyz": item.footprint_world_xyz.tolist(),
                "source_line_ids": list(item.source_line_ids),
                "source_region_id": item.source_region_id,
                "publishable": item.publishable,
                "diagnostics": dict(item.diagnostics),
            }
            for item in result.previews
        ],
        "unresolved": dict(result.unresolved),
        "range": None
        if ranges is None
        else {
            "quantity": "radial_range",
            "units": "m",
            "shape_hw": list(ranges.range.shape),
            "valid_pixels": int(ranges.validity.sum()),
            "conflict_pixels": int(ranges.conflict.sum()),
            "plane_ids": list(ranges.plane_ids),
            "uncertainty": "height-linearized"
            if result.scene.camera_height_sigma_m is not None
            else "unavailable-no-height-sigma",
        },
        "pipeline": manifest,
    }


def _orientation_presentation(result) -> dict[str, Any] | None:
    """Derive display-only leveling numbers from the public orientation result."""

    if result.gravity_camera is None or result.rotation_world_from_camera is None:
        return None
    up = np.asarray(result.gravity_camera, dtype=np.float64)
    canonical_up = np.asarray([0.0, 1.0, 0.0])
    tilt_deg = float(
        np.degrees(np.arccos(np.clip(float(up @ canonical_up), -1.0, 1.0)))
    )
    axis = np.cross(up, canonical_up)
    norm = float(np.linalg.norm(axis))
    if norm > 1e-12:
        axis /= norm
    else:
        axis = np.zeros(3, dtype=np.float64)
    zenith = np.asarray(
        rays_to_erp_pixels(up[None], result.support.shape).pixels_xy[0],
        dtype=np.float64,
    ).copy()
    # Longitude is undefined at the pole.  Keep the display coordinate inside
    # the canonical half-open pixel-centre interval even when round-off returns
    # the equivalent right seam value ``width - 0.5``.
    zenith[0] = (zenith[0] + 0.5) % result.support.shape[1] - 0.5
    return {
        "state": result.orientation_state,
        "tilt_deg": tilt_deg,
        "correction_axis_camera": axis.tolist(),
        "zenith_erp_xy": zenith.tolist(),
        "gravity_camera": up.tolist(),
        "rotation_world_from_camera": (
            np.asarray(result.rotation_world_from_camera).tolist()
        ),
        "camera_center_world": np.asarray(result.camera_center_world).tolist(),
        "rotation_convention": (
            "world_row = camera_row @ rotation_world_from_camera.T"
        ),
    }


def _plane_presentations(result) -> list[dict[str, Any]]:
    """Summarise resolved public planes without adding scientific inference.

    Extents describe only the currently published visible samples.  They are
    deliberately not labelled as physical plane dimensions or surface area.
    """

    ranges = result.ranges[0] if result.ranges else None
    camera = result.camera_center_world
    rotation = result.rotation_world_from_camera
    presentations: list[dict[str, Any]] = []
    for plane in result.planes:
        if plane.normal_world is None or plane.offset_m is None:
            continue
        normal = np.asarray(plane.normal_world, dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            continue
        unit_normal = normal / normal_norm
        item: dict[str, Any] = {
            "plane_id": plane.plane_id,
            "state": plane.state,
            "kind": plane.diagnostics.get("kind"),
            "reference_plane_id": plane.diagnostics.get("reference_plane_id"),
            "normal_world": unit_normal.tolist(),
            "offset_m": float(plane.offset_m) / normal_norm,
            "camera_distance_m": None,
            "valid_pixel_count": 0,
            "radial_range_m": None,
            "visible_extent_m": None,
            "diagnostics": dict(plane.diagnostics),
        }
        if camera is not None:
            item["camera_distance_m"] = abs(
                float(np.asarray(camera, dtype=np.float64) @ unit_normal)
                + item["offset_m"]
            )
        if (
            ranges is not None
            and rotation is not None
            and camera is not None
            and plane.plane_id in ranges.plane_ids
        ):
            plane_index = ranges.plane_ids.index(plane.plane_id)
            mask = ranges.validity & (ranges.plane_index == plane_index)
            values = np.asarray(ranges.range[mask], dtype=np.float64)
            item["valid_pixel_count"] = int(mask.sum())
            if values.size:
                item["radial_range_m"] = {
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "maximum": float(np.max(values)),
                }
                rows, columns = np.nonzero(mask)
                pixels = np.stack((columns, rows), axis=1).astype(np.float64)
                rays_camera = erp_pixels_to_rays(pixels, mask.shape)
                rays_world = rays_camera @ np.asarray(rotation).T
                points_world = (
                    np.asarray(camera)[None] + values[:, None] * rays_world
                )
                reference_axis = np.eye(3)[int(np.argmin(np.abs(unit_normal)))]
                tangent_u = np.cross(unit_normal, reference_axis)
                tangent_u /= np.linalg.norm(tangent_u)
                tangent_v = np.cross(unit_normal, tangent_u)
                projected_u = points_world @ tangent_u
                projected_v = points_world @ tangent_v
                item["visible_extent_m"] = {
                    "u": float(np.ptp(projected_u)),
                    "v": float(np.ptp(projected_v)),
                    "meaning": "extent-of-published-visible-samples",
                }
        presentations.append(item)
    return presentations


def execute_known_height_run(run, directory, image, support):
    """Execute one immutable scene revision through the public application."""

    from .runs import _artifact, _npy_bytes, _png_bytes, _point_cloud_payload

    params = run["parameters"]
    document = params.get("_scene")
    if not isinstance(document, dict):
        raise ValueError("known-height mapping requires a persisted _scene revision")
    shape = tuple(int(value) for value in image.shape[:2])
    validity = np.ones(shape, dtype=bool)
    panorama = EquirectangularImage(image, support_mask=support, valid=validity)
    scene = KnownHeightPlanarScene(
        panorama=panorama,
        camera_height_m=float(document.get("camera_height_m", params["camera_height_m"])),
        camera_height_sigma_m=document.get(
            "camera_height_sigma_m", params.get("camera_height_sigma_m")
        ),
        down_erp_xy=tuple(document["down_erp_xy"]),
        horizon_erp_xy=np.asarray(document.get("horizon_erp_xy", ())),
        horizontal_vanishing_erp_xy=np.asarray(
            document.get("horizontal_vanishing_erp_xy", ())
        ),
        floor_line_families_erp_xy=tuple(
            np.asarray(family)
            for family in document.get("floor_line_families_erp_xy", ())
        ),
        regions=tuple(_region(item, shape) for item in document.get("regions", ())),
        constraints=tuple(
            _constraint(item) for item in document.get("constraints", ())
        ),
        orientations=tuple(
            _orientation(item) for item in document.get("orientations", ())
        ),
        scene_id=document.get("scene_id", "known-height-scene"),
        revision=document.get("revision", "workbench-revision"),
        provenance=document.get("provenance", {}),
    )
    options = KnownHeightPlanarOptions(
        segmentation=str(params.get("segmentation", "simple")),
        face_size=int(params.get("face_size", 512)),
    )
    sam = None
    if options.segmentation == "sam":
        if params.get("accept_sam_terms") != "yes":
            raise ValueError("SAM requires explicit acceptance of its upstream terms")
        from .resources import require_sam21_runtime

        require_sam21_runtime()
        from panorai.experimental.deep_learning.segmentation import (
            load_sam21_hiera_large,
        )

        sam = load_sam21_hiera_large(accept_upstream_terms=True)
    application = KnownHeightPlanarMapping(options, sam_backend=sam)
    pipeline_result = application.run(scene)
    result = pipeline_result.output
    manifest = pipeline_result.manifest()

    artifacts: list[dict[str, Any]] = []
    summary = _summary(result, manifest)
    _artifact(
        directory,
        artifacts,
        "known-height-result",
        "structured-result",
        "Mapa planar por altura conhecida",
        summary,
        extension="json",
    )
    _artifact(
        directory,
        artifacts,
        "scene-revision",
        "scene-revision",
        "Revisão processada",
        _scene_document(result),
        extension="json",
    )
    _artifact(
        directory,
        artifacts,
        "pipeline-trace",
        "pipeline-trace",
        "Execução dos oito estágios",
        manifest,
        extension="json",
    )
    if result.ranges:
        values = result.ranges[0]
        _artifact(directory, artifacts, "range", "range", "Range radial", _png_bytes(values.range))
        _artifact(directory, artifacts, "range-exact", "range", "Range radial exato", _npy_bytes(values.range), extension="npy")
        _artifact(directory, artifacts, "validity", "mask", "Validade", _png_bytes(values.validity))
        _artifact(directory, artifacts, "conflict", "mask", "Conflitos", _png_bytes(values.conflict))
        _artifact(directory, artifacts, "plane-index", "labels", "Planos resolvidos", _png_bytes(values.plane_index + 1, labels=True))
        _artifact(directory, artifacts, "uncertainty-exact", "uncertainty", "Incerteza métrica", _npy_bytes(values.uncertainty_m), extension="npy")
        if values.validity.any():
            cloud = _point_cloud_payload(
                image,
                values.range,
                values.validity,
                rotation_world_from_camera=result.rotation_world_from_camera,
                camera_center_world=result.camera_center_world,
            )
            _artifact(directory, artifacts, "point-cloud", "point-cloud", "Nuvem métrica", cloud, extension="json")
    metrics = {
        "state": result.state,
        "orientation_state": result.orientation_state,
        "orientation": _orientation_presentation(result),
        "resolved_planes": len(result.planes),
        "unresolved_planes": len(result.unresolved),
        "plane_metrics": _plane_presentations(result),
        "planar_previews": _summary(result, manifest)["previews"],
        "unresolved": dict(result.unresolved),
        "valid_range_pixels": 0
        if not result.ranges
        else int(result.ranges[0].validity.sum()),
    }
    warnings = [
        f"{plane_id}: {reason}" for plane_id, reason in result.unresolved.items()
    ]
    if result.orientation_state != "accepted":
        warnings.append("A orientação precisa de revisão; geometria métrica não publicada")
    return artifacts, metrics, warnings
