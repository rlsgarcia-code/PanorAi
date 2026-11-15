"""Analytic geometry-v1 oracles for known-height planar mapping.

Fixtures use a +Y-up camera at ``C=[0, 1.6, 0]`` in metres. Expected planes
and ranges are closed-form ray/plane intersections, independent of the
implementation under test.
"""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panorai import EquirectangularImage
from panorai.geometry import erp_pixels_to_rays, rays_to_erp_pixels
from panorai.pipeline import Pipeline
from panorai.reconstruction.known_height_planar import (
    KnownHeightPlanarMapping,
    KnownHeightPlanarOptions,
    KnownHeightPlanarScene,
    PlanarConstraint,
    PlanarOrientationEvidence,
    preview_planar_relation,
)
from panorai.reconstruction.planar import PlaneEstimate, PlanarRegion


SHAPE = (48, 96)
HEIGHT_M = 1.6
VIEW_ID = "single-view"


def _pixels_of(vectors: np.ndarray) -> np.ndarray:
    return np.asarray(rays_to_erp_pixels(vectors, SHAPE).pixels_xy)


def _horizon(normal=np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    first = np.cross(normal, [0.0, 0.0, 1.0])
    if np.linalg.norm(first) < 1e-8:
        first = np.cross(normal, [1.0, 0.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    rays = np.stack((first + second, first - second, -first + second, -first - second))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return _pixels_of(rays)


def _down_pixel() -> tuple[float, float]:
    # ERP pixel centres do not include the exact pole. This supported sample is
    # used only for the hemisphere sign; the optional horizon supplies exact up.
    h, w = SHAPE
    return (w / 2 - 0.5, h - 1.0)


def _floor_line_families() -> tuple[np.ndarray, np.ndarray]:
    """Two orthogonal horizontal line families in the analytic Y=0 plane."""

    centre = np.array([0.0, HEIGHT_M, 0.0])
    along_x = np.stack(
        [
            _pixels_of(np.array([[-2.0, 0.0, z], [2.0, 0.0, z]]) - centre)
            for z in (2.0, 4.0)
        ]
    )
    along_z = np.stack(
        [
            _pixels_of(np.array([[x, 0.0, 1.0], [x, 0.0, 4.0]]) - centre)
            for x in (-1.5, 1.5)
        ]
    )
    return along_x, along_z


def _regions_and_constraints(*, accepted=True, relation_state="accepted"):
    h, w = SHAPE
    pixels = np.stack(
        np.meshgrid(np.arange(w), np.arange(h), indexing="xy"), axis=-1
    ).astype(np.float64)
    rays = erp_pixels_to_rays(pixels, SHAPE)
    centre = np.array([0.0, HEIGHT_M, 0.0])

    floor_distance = -HEIGHT_M / rays[..., 1]
    floor_points = centre + floor_distance[..., None] * rays
    floor = (
        (rays[..., 1] < -0.05)
        & (floor_distance > 0)
        & (np.abs(floor_points[..., 0]) < 2.0)
        & (floor_points[..., 2] > 0.2)
        & (floor_points[..., 2] < 4.0)
    )

    wall_distance = 4.0 / rays[..., 2]
    wall_points = centre + wall_distance[..., None] * rays
    wall = (
        (rays[..., 2] > 0.05)
        & (np.abs(wall_points[..., 0]) < 2.0)
        & (wall_points[..., 1] > 0.0)
        & (wall_points[..., 1] < 3.0)
    )

    ceiling_distance = (3.0 - HEIGHT_M) / rays[..., 1]
    ceiling_points = centre + ceiling_distance[..., None] * rays
    ceiling = (
        (rays[..., 1] > 0.05)
        & (ceiling_distance > 0)
        & (np.abs(ceiling_points[..., 0]) < 2.0)
        & (ceiling_points[..., 2] > 0.2)
        & (ceiling_points[..., 2] < 4.0)
    )

    state = "accepted" if accepted else "draft"
    regions = tuple(
        PlanarRegion(name, VIEW_ID, mask, state=state)
        for name, mask in (("floor", floor), ("wall", wall), ("ceiling", ceiling))
    )
    floor_contact = _pixels_of(
        np.array([[-1.0, 0.0, 4.0], [1.0, 0.0, 4.0]]) - centre
    )
    ceiling_contact = _pixels_of(
        np.array([[-1.0, 3.0, 4.0], [1.0, 3.0, 4.0]]) - centre
    )
    constraints = (
        PlanarConstraint("floor-anchor", "floor", "floor-anchor", state="accepted"),
        PlanarConstraint(
            "wall-floor",
            "wall",
            "vertical-to-floor",
            "floor",
            floor_contact,
            relation_state,
        ),
        PlanarConstraint(
            "ceiling-wall",
            "ceiling",
            "horizontal-to-plane",
            "wall",
            ceiling_contact,
            relation_state,
        ),
    )
    return regions, constraints


def _scene(*, accepted=True, relation_state="accepted", horizon=None, sigma=0.02):
    regions, constraints = _regions_and_constraints(
        accepted=accepted, relation_state=relation_state
    )
    # All-black RGB is intentionally valid evidence; support/validity are
    # independent and explicit.
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    panorama = EquirectangularImage(
        image,
        support_mask=np.ones(SHAPE, bool),
        valid=np.ones(SHAPE, bool),
    )
    return KnownHeightPlanarScene(
        panorama,
        HEIGHT_M,
        _down_pixel(),
        _horizon() if horizon is None else horizon,
        regions,
        constraints,
        sigma,
        scene_id="analytic-room",
        revision="closed-form-v1",
        provenance={"oracle": "analytic ray-plane intersections"},
    )


def _application(**updates):
    return KnownHeightPlanarMapping(
        KnownHeightPlanarOptions(segmentation="manual", **updates)
    )


def test_floor_wall_ceiling_chain_matches_closed_form_planes_and_ranges():
    result = _application()(_scene())
    assert result.state == "accepted"
    assert not result.unresolved
    planes = {plane.plane_id: plane for plane in result.planes}
    np.testing.assert_allclose(planes["floor"].normal_world, [0, 1, 0], atol=1e-12)
    assert planes["floor"].offset_m == 0
    np.testing.assert_allclose(planes["wall"].normal_world, [0, 0, -1], atol=1e-12)
    assert planes["wall"].offset_m == pytest.approx(4.0, abs=1e-12)
    np.testing.assert_allclose(planes["ceiling"].normal_world, [0, 1, 0], atol=1e-12)
    assert planes["ceiling"].offset_m == pytest.approx(-3.0, abs=1e-12)

    maps = result.ranges[0]
    assert maps.validity.any()
    assert not maps.conflict.any()
    pixels = np.stack(
        np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[0]), indexing="xy"),
        axis=-1,
    ).astype(np.float64)
    rays = erp_pixels_to_rays(pixels, SHAPE)
    expected = np.full(SHAPE, np.nan)
    for index, plane_id in enumerate(maps.plane_ids):
        plane = planes[plane_id]
        denominator = rays @ plane.normal_world
        numerator = -(np.array([0.0, HEIGHT_M, 0.0]) @ plane.normal_world + plane.offset_m)
        mask = maps.validity & (maps.plane_index == index)
        expected[mask] = numerator / denominator[mask]
    np.testing.assert_allclose(maps.range[maps.validity], expected[maps.validity], rtol=1e-12)
    np.testing.assert_allclose(
        maps.uncertainty_m[maps.validity],
        np.abs(maps.range[maps.validity] / HEIGHT_M) * 0.02,
        rtol=1e-12,
    )
    assert np.isnan(maps.range[~maps.validity]).all()
    assert np.all(maps.source[maps.validity] == 2)


def test_draft_masks_and_proposed_relations_never_publish_geometry():
    draft = _application()(_scene(accepted=False))
    assert not draft.planes
    assert not draft.ranges[0].validity.any()
    assert "floor" in draft.unresolved

    proposed = _application()(_scene(relation_state="proposed"))
    assert {plane.plane_id for plane in proposed.planes} == {"floor"}
    assert set(proposed.ranges[0].plane_ids) == {"floor"}


def test_reviewed_orientation_and_one_contact_position_an_orthogonal_plane():
    scene = _scene()
    contact = scene.constraints[1].contact_erp_xy[:1]
    orientation = PlanarOrientationEvidence(
        "wall",
        "wall-orientation",
        np.array([0.0, 0.0, -1.0]),
        line_ids=("line-a", "line-b"),
        bundle_ids=("bundle-a", "bundle-b"),
        confidence=0.9,
        state="accepted",
        source="human",
    )
    constraint = PlanarConstraint(
        "wall-floor-generic",
        "wall",
        "orthogonal-to",
        "floor",
        contact,
        "accepted",
        orientation_id=orientation.proposal_id,
    )
    result = _application()(
        replace(
            scene,
            regions=scene.regions[:2],
            constraints=(scene.constraints[0], constraint),
            orientations=(orientation,),
        )
    )
    wall = next(plane for plane in result.planes if plane.plane_id == "wall")
    np.testing.assert_allclose(wall.normal_world, [0, 0, -1], atol=1e-12)
    assert wall.offset_m == pytest.approx(4.0, abs=1e-12)
    assert next(item for item in result.previews if item.plane_id == "wall").publishable


def test_positioned_draft_preview_never_enters_planes_or_ranges():
    scene = _scene()
    orientation = PlanarOrientationEvidence(
        "wall", "wall-draft-orientation", [0, 0, -1], state="accepted", source="human"
    )
    constraint = PlanarConstraint(
        "wall-draft-contact",
        "wall",
        "orthogonal-to",
        "floor",
        scene.constraints[1].contact_erp_xy[:1],
        "proposed",
        orientation_id=orientation.proposal_id,
    )
    wall_draft = replace(scene.regions[1], state="draft")
    result = _application()(
        replace(
            scene,
            regions=(scene.regions[0], wall_draft),
            constraints=(scene.constraints[0], constraint),
            orientations=(orientation,),
        )
    )
    preview = next(item for item in result.previews if item.plane_id == "wall")
    assert preview.state == "positioned-draft"
    assert not preview.publishable
    assert {plane.plane_id for plane in result.planes} == {"floor"}
    assert set(result.ranges[0].plane_ids) == {"floor"}


def test_known_height_floor_is_positioned_in_preview_before_publication():
    scene = _scene(accepted=False)
    proposed_anchor = replace(scene.constraints[0], state="proposed")
    result = _application()(
        replace(
            scene,
            regions=(scene.regions[0],),
            constraints=(proposed_anchor,),
        )
    )

    preview = next(item for item in result.previews if item.plane_id == "floor")
    assert preview.state == "positioned-draft"
    assert not preview.publishable
    np.testing.assert_allclose(preview.normal_world, [0.0, 1.0, 0.0], atol=1e-12)
    assert preview.offset_m == pytest.approx(0.0, abs=1e-12)
    assert preview.footprint_world_xyz.size
    assert preview.diagnostics["source"] == "known-height-floor-anchor"
    assert preview.diagnostics["camera_height_m"] == HEIGHT_M
    assert not result.planes
    assert not result.ranges[0].validity.any()


def test_lightweight_contact_preview_is_non_publishable_closed_form_geometry():
    reference = PlaneEstimate("floor", np.array([0.0, 1.0, 0.0]), 0.0, None, None, "accepted")
    cursor = tuple(
        _pixels_of(np.array([[0.0, -HEIGHT_M, 4.0]]))[0]
    )
    preview = preview_planar_relation(
        plane_id="wall",
        relation="orthogonal-to",
        reference_plane=reference,
        cursor_erp_xy=cursor,
        erp_shape_hw=SHAPE,
        rotation_world_from_camera=np.eye(3),
        camera_center_world=np.array([0.0, HEIGHT_M, 0.0]),
        candidate_normal_world=np.array([0.0, 0.0, -1.0]),
    )
    assert preview.state == "positioned-draft"
    assert not preview.publishable
    np.testing.assert_allclose(preview.contact_world_xyz[0], [0.0, 0.0, 4.0], atol=1e-12)
    assert preview.offset_m == pytest.approx(4.0, abs=1e-12)


def test_inconsistent_horizon_requires_review_and_publishes_no_metric_range():
    angle = np.deg2rad(12.0)
    tilted_up = np.array([0.0, np.cos(angle), np.sin(angle)])
    result = _application()(_scene(horizon=_horizon(tilted_up)))
    assert result.orientation_state == "needs-review"
    assert result.state == "needs-review"
    assert not result.planes
    assert not result.ranges[0].validity.any()
    assert result.orientation_diagnostics["reason"] == "horizon-down-disagreement"


def test_floor_line_families_recover_horizontal_frame_before_geometry():
    scene = replace(
        _scene(),
        horizon_erp_xy=np.empty((0, 2)),
        floor_line_families_erp_xy=_floor_line_families(),
    )
    result = _application()(scene)
    assert result.orientation_state == "accepted"
    assert result.orientation_diagnostics["source"] == "floor-line-families"
    np.testing.assert_allclose(result.gravity_camera, [0.0, 1.0, 0.0], atol=1e-12)
    assert {plane.plane_id for plane in result.planes} == {
        "floor",
        "wall",
        "ceiling",
    }


def test_direct_vanishing_points_are_not_shadowed_by_empty_line_families():
    scene = replace(
        _scene(),
        horizon_erp_xy=np.empty((0, 2)),
        horizontal_vanishing_erp_xy=_pixels_of(
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        ),
        floor_line_families_erp_xy=(
            np.empty((0, 2, 2)),
            np.empty((0, 2, 2)),
        ),
    )
    result = _application()(scene)
    assert result.orientation_state == "accepted"
    assert result.orientation_diagnostics["source"] == "horizontal-vanishing-points"
    np.testing.assert_allclose(result.gravity_camera, [0.0, 1.0, 0.0], atol=1e-12)


def test_incomplete_floor_line_alignment_requires_review_and_publishes_no_geometry():
    scene = replace(
        _scene(),
        horizon_erp_xy=np.empty((0, 2)),
        floor_line_families_erp_xy=(_floor_line_families()[0],),
    )
    result = _application()(scene)
    assert result.orientation_state == "needs-review"
    assert result.orientation_diagnostics["reason"] == "invalid-floor-line-alignment"
    assert not result.planes
    assert not result.ranges[0].validity.any()


def test_short_contact_and_constraint_cycle_remain_explicitly_unresolved():
    scene = _scene()
    short = replace(
        scene.constraints[1],
        contact_erp_xy=_pixels_of(
            np.array([[0.0, 0.0, 4.0], [0.01, 0.0, 4.0]])
            - np.array([0.0, HEIGHT_M, 0.0])
        ),
    )
    result = _application()(replace(scene, constraints=(scene.constraints[0], short)))
    assert result.unresolved["wall"] == "contact-span-too-short"
    assert {plane.plane_id for plane in result.planes} == {"floor"}

    regions = scene.regions + (
        PlanarRegion("copy", VIEW_ID, scene.regions[1].mask, state="accepted"),
    )
    cycle = (
        scene.constraints[0],
        PlanarConstraint("a", "wall", "coplanar-with", "copy", state="accepted"),
        PlanarConstraint("b", "copy", "coplanar-with", "wall", state="accepted"),
    )
    result = _application()(replace(scene, regions=regions, constraints=cycle))
    assert result.unresolved["wall"] == "reference-plane-unresolved-or-cycle"
    assert result.unresolved["copy"] == "reference-plane-unresolved-or-cycle"


def test_absent_height_sigma_is_unavailable_not_false_zero():
    result = _application()(replace(_scene(), camera_height_sigma_m=None))
    maps = result.ranges[0]
    assert maps.validity.any()
    assert np.isnan(maps.uncertainty_m).all()


def test_equivalent_antimeridian_observations_are_normalized_periodically():
    scene = _scene()
    shifted_horizon = np.array(scene.horizon_erp_xy, copy=True)
    shifted_horizon[0, 0] += SHAPE[1]
    shifted_contact = np.array(scene.constraints[1].contact_erp_xy, copy=True)
    shifted_contact[:, 0] += SHAPE[1]
    normalized = replace(
        scene,
        down_erp_xy=(scene.down_erp_xy[0] + SHAPE[1], scene.down_erp_xy[1]),
        horizon_erp_xy=shifted_horizon,
        constraints=(
            scene.constraints[0],
            replace(scene.constraints[1], contact_erp_xy=shifted_contact),
            scene.constraints[2],
        ),
    )

    result = _application()(normalized)

    assert result.orientation_state == "accepted"
    assert {plane.plane_id for plane in result.planes} == {
        "floor",
        "wall",
        "ceiling",
    }
    assert all(-0.5 <= point[0] < SHAPE[1] - 0.5 for point in normalized.horizon_erp_xy)


def test_pipeline_round_trip_and_public_import_do_not_load_optional_backends():
    app = KnownHeightPlanarMapping()
    restored = Pipeline.from_config(app.pipeline.to_config())
    assert restored.describe() == app.pipeline.describe()
    assert restored(_scene()).planes
    assert len(app.named_steps) == 8
    assert app.describe()["interface"].startswith("panorai-known-height-planar")

    root = Path(__file__).resolve().parents[1]
    source = f"""
import sys
sys.path.insert(0, {str(root)!r})
from panorai.reconstruction.known_height_planar import KnownHeightPlanarMapping
assert 'torch' not in sys.modules and 'open3d' not in sys.modules
assert KnownHeightPlanarMapping().pipeline.input_kind == 'known-height-planar-scene'
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
