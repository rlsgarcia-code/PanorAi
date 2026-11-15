"""Public lightweight relation preview used by interactive clients."""

from __future__ import annotations

import numpy as np

from panorai.geometry import erp_pixels_to_rays
from panorai.reconstruction.planar import PlaneEstimate

from ._models import PlanarPreview, _readonly


def preview_planar_relation(
    *,
    plane_id: str,
    relation: str,
    reference_plane: PlaneEstimate,
    cursor_erp_xy: tuple[float, float],
    erp_shape_hw: tuple[int, int],
    rotation_world_from_camera: np.ndarray,
    camera_center_world: np.ndarray,
    candidate_normal_world: np.ndarray | None = None,
    candidate_normal_camera: np.ndarray | None = None,
    minimum_incidence: float = 0.05,
) -> PlanarPreview:
    """Position a draft plane at one reviewed cursor contact.

    This operation publishes no scientific range or accepted ``PlaneEstimate``.
    It exists so interactive clients never reimplement spherical ray/plane
    intersections or relation projection.
    """

    if relation not in {"orthogonal-to", "parallel-to", "coplanar-with"}:
        raise ValueError("relation must be orthogonal-to, parallel-to or coplanar-with")
    rotation = np.asarray(rotation_world_from_camera, dtype=np.float64)
    center = np.asarray(camera_center_world, dtype=np.float64)
    if rotation.shape != (3, 3) or center.shape != (3,):
        raise ValueError("rotation and camera centre have invalid shapes")
    reference_normal = np.asarray(reference_plane.normal_world, dtype=np.float64)
    if reference_normal.shape != (3,) or reference_plane.offset_m is None:
        raise ValueError("reference plane must be positioned")
    reference_normal /= np.linalg.norm(reference_normal)
    ray_camera = erp_pixels_to_rays(
        np.asarray(cursor_erp_xy, dtype=np.float64)[None], erp_shape_hw
    )[0]
    ray_world = ray_camera @ rotation.T
    denominator = float(ray_world @ reference_normal)
    if abs(denominator) < minimum_incidence:
        raise ValueError("cursor ray is tangent to the reference plane")
    distance = -float(center @ reference_normal + reference_plane.offset_m) / denominator
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("cursor does not intersect the reference plane in front of the camera")
    contact = center + distance * ray_world
    if relation == "coplanar-with":
        normal = reference_normal
        offset = float(reference_plane.offset_m)
    elif relation == "parallel-to":
        normal = reference_normal
        if candidate_normal_world is not None and normal @ candidate_normal_world < 0:
            normal = -normal
        offset = -float(normal @ contact)
    else:
        if candidate_normal_world is not None and candidate_normal_camera is not None:
            raise ValueError("provide candidate orientation in only one frame")
        if candidate_normal_camera is not None:
            candidate_normal_world = np.asarray(candidate_normal_camera, dtype=np.float64) @ rotation.T
        if candidate_normal_world is None:
            raise ValueError("orthogonal preview requires candidate orientation evidence")
        observed = np.asarray(candidate_normal_world, dtype=np.float64)
        if observed.shape != (3,) or np.linalg.norm(observed) < 1e-12:
            raise ValueError("candidate orientation must be a non-degenerate 3-vector")
        observed /= np.linalg.norm(observed)
        normal = observed - reference_normal * float(observed @ reference_normal)
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError("candidate orientation is degenerate for the orthogonal relation")
        normal /= np.linalg.norm(normal)
        if normal @ observed < 0:
            normal = -normal
        offset = -float(normal @ contact)
    return PlanarPreview(
        plane_id=plane_id,
        state="positioned-draft",
        normal_world=_readonly(normal),
        offset_m=offset,
        contact_world_xyz=_readonly(contact[None]),
        publishable=False,
        diagnostics={"relation": relation, "reference_plane_id": reference_plane.plane_id},
    )
