"""Analytic and integration tests for Experimental planar discovery.

The synthetic ERP contains perspective projections of two declared, non-
orthogonal line directions in the camera-frame plane ``Z=4``.  Expected
directions and plane normal are analytic and independent of the detector.
"""

from dataclasses import replace

import cv2
import numpy as np

from panorai import EquirectangularImage
from panorai.geometry import rays_to_erp_pixels
from panorai.pipeline import Pipeline, registered_pipeline_components
from panorai.reconstruction.planar import (
    PlanarDiscoveryRequest,
    PlanarRegion,
    SphericalLineSeed,
    SphericalPlanarDiscovery,
    SphericalPlanarDiscoveryOptions,
)


SHAPE = (256, 512)
DIRECTION_A = np.array([1.0, 0.0, 0.0])
DIRECTION_B = np.array([0.45, 0.89, 0.0])
DIRECTION_B /= np.linalg.norm(DIRECTION_B)
PLANE_NORMAL = np.array([0.0, 0.0, 1.0])


def _draw_spatial_line(
    image: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> None:
    parameters = np.linspace(-3.0, 3.0, 320)
    points = origin[None] + parameters[:, None] * direction[None]
    rays = points / np.linalg.norm(points, axis=1, keepdims=True)
    pixels = np.asarray(rays_to_erp_pixels(rays, SHAPE).pixels_xy)
    start = 0
    for index in range(1, len(pixels) + 1):
        if index == len(pixels) or abs(pixels[index, 0] - pixels[index - 1, 0]) > SHAPE[1] / 2:
            segment = np.rint(pixels[start:index]).astype(np.int32)
            if len(segment) > 1:
                cv2.polylines(image, [segment], False, (255, 255, 255), 3, cv2.LINE_AA)
            start = index


def _fixture() -> PlanarDiscoveryRequest:
    image = np.zeros((*SHAPE, 3), dtype=np.uint8)
    for y in (-1.2, -0.4, 0.4, 1.2):
        _draw_spatial_line(image, np.array([0.0, y, 4.0]), DIRECTION_A)
    perpendicular = np.array([-DIRECTION_B[1], DIRECTION_B[0], 0.0])
    for offset in (-1.2, -0.4, 0.4, 1.2):
        _draw_spatial_line(
            image,
            np.array([0.0, 0.0, 4.0]) + offset * perpendicular,
            DIRECTION_B,
        )
    support = np.ones(SHAPE, dtype=bool)
    panorama = EquirectangularImage(image, support_mask=support, valid=support)
    return PlanarDiscoveryRequest(
        panorama,
        revision="analytic-non-manhattan-plane-v1",
        provenance={
            "oracle": "two declared directions in camera plane Z=4",
            "units": "direction vectors are dimensionless",
            "grid": "geometry-v1 ERP pixel centres",
        },
    )


def _application() -> SphericalPlanarDiscovery:
    return SphericalPlanarDiscovery(
        SphericalPlanarDiscoveryOptions(
            atlas_face_size=256,
            minimum_bundle_lines=2,
            minimum_bundle_support_deg=4.0,
        )
    )


def _low_contrast_fixture() -> PlanarDiscoveryRequest:
    request = _fixture()
    source = np.asarray(request.panorama.image, dtype=np.float32)
    image = np.rint(104.0 + source * (36.0 / 255.0)).astype(np.uint8)
    support = np.ones(SHAPE, dtype=bool)
    return replace(
        request,
        panorama=EquirectangularImage(image, support_mask=support, valid=support),
        revision="analytic-low-contrast-plane-v1",
        provenance={
            **request.provenance,
            "photometric_oracle": "background 104, declared line peak 140 uint8",
        },
    )


def _nearest_bundle(result, direction):
    return max(
        result.bundles,
        key=lambda bundle: abs(float(bundle.direction_camera @ direction)),
    )


def test_discovery_recovers_non_manhattan_directions_and_never_metric_offset():
    result = _application()(_fixture())

    assert result.state == "needs-review"
    assert len(result.lines) >= 8
    first = _nearest_bundle(result, DIRECTION_A)
    second = _nearest_bundle(result, DIRECTION_B)
    assert abs(float(first.direction_camera @ DIRECTION_A)) > np.cos(np.deg2rad(6))
    assert abs(float(second.direction_camera @ DIRECTION_B)) > np.cos(np.deg2rad(6))
    assert first.bundle_id != second.bundle_id
    assert result.proposals
    assert result.edge_preview_erp_xy.ndim == 2
    assert result.edge_preview_erp_xy.shape[1] == 2
    assert 0 < len(result.edge_preview_erp_xy) <= 20_000
    assert all(proposal.estimate.offset_m is None for proposal in result.proposals)
    assert result.diagnostics["metric_geometry_published"] is False


def test_reviewed_bundle_pair_yields_orientation_only_analytic_plane():
    initial = _application()(_fixture())
    first = _nearest_bundle(initial, DIRECTION_A)
    second = _nearest_bundle(initial, DIRECTION_B)
    request = replace(
        _fixture(),
        plane_bundle_groups={"reviewed-wall": (first.bundle_id, second.bundle_id)},
        revision="analytic-reviewed-v2",
    )

    result = _application()(request)
    proposal = next(item for item in result.proposals if item.proposal_id == "reviewed-wall")

    assert result.state == "reviewed"
    assert proposal.source == "human"
    assert proposal.state == "accepted"
    assert abs(float(proposal.normal_camera @ PLANE_NORMAL)) > np.cos(np.deg2rad(7))
    assert proposal.estimate.normal_world is None
    assert proposal.estimate.offset_m is None
    assert proposal.estimate.diagnostics["normal_frame"] == "camera"


def test_line_decision_and_human_bundle_are_reproducible():
    initial = _application()(_fixture())
    automatic = _nearest_bundle(initial, DIRECTION_A)
    selected = automatic.line_ids[:2]
    request = replace(
        _fixture(),
        line_decisions={initial.lines[-1].line_id: "rejected"},
        bundle_memberships={"human-a": selected},
        revision="analytic-human-bundle-v2",
    )
    first = _application()(request)
    second = _application()(request)

    human = next(bundle for bundle in first.bundles if bundle.bundle_id == "human-a")
    assert human.source == "human"
    assert human.state == "accepted"
    assert human.line_ids == selected
    assert [line.line_id for line in first.lines] == [line.line_id for line in second.lines]
    np.testing.assert_allclose(
        [bundle.direction_camera for bundle in first.bundles],
        [bundle.direction_camera for bundle in second.bundles],
        atol=0,
        rtol=0,
    )


def test_estimated_direction_suggests_and_replays_compatible_nonmember_lines():
    initial = _application()(_fixture())
    automatic = _nearest_bundle(initial, DIRECTION_A)
    request = replace(
        _fixture(),
        bundle_memberships={"reviewed-a": automatic.line_ids[:2]},
        revision="analytic-direction-expansion-v2",
    )

    expanded = _application()(request)
    reviewed = next(bundle for bundle in expanded.bundles if bundle.bundle_id == "reviewed-a")

    assert reviewed.suggestions
    candidate = min(reviewed.suggestions, key=lambda item: item.residual_deg)
    assert candidate.line_id not in reviewed.line_ids
    assert candidate.residual_deg <= expanded.diagnostics["suggestion_residual_deg"]
    assert candidate.state == "suggested"
    assert 0 <= candidate.confidence <= 1

    rejected = _application()(
        replace(
            request,
            bundle_line_decisions={"reviewed-a": {candidate.line_id: "rejected"}},
            revision="analytic-direction-expansion-v3",
        )
    )
    replayed = next(bundle for bundle in rejected.bundles if bundle.bundle_id == "reviewed-a")
    decision = next(item for item in replayed.suggestions if item.line_id == candidate.line_id)
    assert decision.state == "rejected"

    accepted = _application()(
        replace(
            request,
            bundle_memberships={
                "reviewed-a": tuple((*automatic.line_ids[:2], candidate.line_id))
            },
            revision="analytic-direction-expansion-v4",
        )
    )
    refitted = next(bundle for bundle in accepted.bundles if bundle.bundle_id == "reviewed-a")
    assert candidate.line_id in refitted.line_ids
    assert candidate.line_id not in {item.line_id for item in refitted.suggestions}


def test_manual_lines_and_plane_line_association_share_the_public_contract():
    initial = _application()(_fixture())
    first = _nearest_bundle(initial, DIRECTION_A)
    second = _nearest_bundle(initial, DIRECTION_B)
    seed = SphericalLineSeed(
        "manual-reviewed-line",
        initial.lines[0].erp_endpoints_xy + np.array([[0.25, 0.0], [0.25, 0.0]]),
    )
    selected = tuple(first.line_ids[:2] + second.line_ids[:2])
    request = replace(
        _fixture(),
        manual_line_seeds=(seed,),
        plane_line_groups={"wall-a": selected},
        revision="manual-and-associated-v3",
    )

    result = _application()(request)

    manual = next(line for line in result.lines if line.line_id == seed.line_id)
    assert manual.source == "human"
    assert manual.state == "accepted"
    proposal = next(item for item in result.proposals if item.plane_id == "wall-a")
    assert proposal.source == "human"
    assert proposal.estimate.offset_m is None
    assert abs(float(proposal.normal_camera @ PLANE_NORMAL)) > np.cos(np.deg2rad(7))


def test_local_contrast_recovers_declared_low_contrast_line_directions():
    options = SphericalPlanarDiscoveryOptions(
        atlas_face_size=256,
        canny_low=25,
        canny_high=105,
        local_contrast_clip_limit=2.0,
        minimum_segment_px=14,
        minimum_edge_support=0.3,
        minimum_bundle_lines=2,
        minimum_bundle_support_deg=4.0,
    )

    result = SphericalPlanarDiscovery(options)(_low_contrast_fixture())

    first = _nearest_bundle(result, DIRECTION_A)
    second = _nearest_bundle(result, DIRECTION_B)
    assert abs(float(first.direction_camera @ DIRECTION_A)) > np.cos(np.deg2rad(7))
    assert abs(float(second.direction_camera @ DIRECTION_B)) > np.cos(np.deg2rad(7))
    assert result.diagnostics["local_contrast_clip_limit"] == 2.0


def test_manual_line_is_never_evicted_by_the_automatic_line_budget():
    initial = _application()(_fixture())
    seed = SphericalLineSeed(
        "manual-budget-proof",
        initial.lines[-1].erp_endpoints_xy + np.array([[0.2, 0.0], [0.2, 0.0]]),
    )
    options = replace(_application().options, maximum_lines=1)

    result = SphericalPlanarDiscovery(options)(
        replace(_fixture(), manual_line_seeds=(seed,), revision="manual-budget-v2")
    )

    assert len([line for line in result.lines if line.source == "automatic"]) == 1
    manual = next(line for line in result.lines if line.line_id == seed.line_id)
    assert manual.source == "human"
    assert result.diagnostics["deduplicated_lines_retained"] == 2


def test_region_support_is_nearest_neighbour_and_independent_of_black_intensity():
    request = _fixture()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, : SHAPE[1] // 2] = True
    region = PlanarRegion("left", "single-view", mask, state="draft")
    result = _application()(replace(request, regions=(region,)))

    assert result.lines
    assert all(0.0 <= line.region_support["left"] <= 1.0 for line in result.lines)
    assert np.asarray(request.panorama.validity("image")).all()
    assert np.count_nonzero(request.panorama.image == 0) > 0

def test_discovery_pipeline_is_safe_registered_and_round_trips():
    application = _application()
    config = application.pipeline.to_config()
    restored = Pipeline.from_config(config)

    assert list(restored.named_steps) == list(application.named_steps)
    registered = registered_pipeline_components()
    for stage in application.named_steps.values():
        assert f"panorai.reconstruction.planar.{type(stage).__name__}" in registered
    assert "panorai.reconstruction.planar.SphericalPlanarDiscoveryOptions" in registered


def test_black_pixels_remain_valid_evidence_domain():
    request = _fixture()
    image = np.asarray(request.panorama.image)
    assert np.count_nonzero(image) < image.size / 10
    assert np.asarray(request.panorama.validity("image")).all()
    assert np.asarray(request.panorama.support_mask).all()

    result = _application()(request)

    assert result.lines
    assert result.diagnostics["edge_pixels_in_atlas"] > 0
