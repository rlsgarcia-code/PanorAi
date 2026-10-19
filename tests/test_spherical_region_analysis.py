from __future__ import annotations

from dataclasses import replace
import math
import sys

import numpy as np
import pytest
from scipy import ndimage

import panorai as pa
from panorai.core.numpy import spherical_solid_angle_weights
from panorai.data import RasterFieldSpec
from panorai.evaluation import evaluate_binary_regions
from panorai.geometry import erp_pixels_to_rays
from panorai.ops import spherical as S
from panorai.pipeline import AnalyzeRegions, SphericalWatershed


def _rays(shape_hw: tuple[int, int]) -> np.ndarray:
    rows, columns = np.indices(shape_hw)
    return erp_pixels_to_rays(np.stack((columns, rows), axis=-1), shape_hw)


def test_connected_components_join_seam_and_respect_invalid_barrier() -> None:
    mask = np.zeros((12, 24), dtype=bool)
    mask[4:8, :3] = True
    mask[4:8, -3:] = True
    mask[2:10, 10:14] = True
    validity = np.ones_like(mask)

    labels = S.connected_components(mask, validity=validity, connectivity=8)

    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[5, 0] == labels[5, -1]
    assert labels[5, 0] != labels[5, 11]

    validity[:, 12] = False
    split = S.connected_components(mask, validity=validity, connectivity=4)
    assert len(set(np.unique(split)) - {0}) == 3
    assert np.all(split[:, 12] == 0)


def test_connected_components_matches_planar_oracle_away_from_seam() -> None:
    mask = np.zeros((18, 36), dtype=bool)
    mask[4:8, 6:10] = True
    mask[11:15, 22:28] = True

    actual = S.connected_components(mask, connectivity=8)
    expected, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))

    np.testing.assert_array_equal(actual, expected.astype(np.int32))


def test_polar_region_has_one_component_and_north_centroid() -> None:
    mask = np.zeros((24, 48), dtype=bool)
    mask[:3, :] = True

    labels = S.connected_components(mask)
    measurement = S.measure_regions(labels).regions[0]

    assert labels.max() == 1
    np.testing.assert_allclose(
        measurement.centroid_bearing_xyz,
        np.asarray((0.0, 1.0, 0.0)),
        atol=2e-15,
    )


def test_geodesic_distance_transform_matches_great_circle_oracle() -> None:
    shape = (24, 48)
    target = np.zeros(shape, dtype=bool)
    target[shape[0] // 2, 0] = True
    rays = _rays(shape)
    center = rays[shape[0] // 2, 0]
    dot = np.clip(rays @ center, -1.0, 1.0)
    expected = np.rad2deg(np.arccos(dot))
    expected[np.isclose(dot, 1.0, atol=1e-12, rtol=0.0)] = 0.0
    expected[np.isclose(dot, -1.0, atol=1e-12, rtol=0.0)] = 180.0

    actual = S.geodesic_distance_transform(target)

    np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=0.0)
    assert actual[shape[0] // 2, -1] < 360.0 / shape[1] + 1e-12


def test_distance_transform_preserves_invalidity_as_nan() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[5, 10] = True
    validity = np.ones_like(mask)
    validity[:, 4] = False

    result = S.geodesic_distance_transform(mask, validity=validity)

    assert np.isnan(result[:, 4]).all()
    assert np.isfinite(result[validity]).all()
    with pytest.raises(ValueError, match="no target"):
        S.geodesic_distance_transform(mask, validity=~mask)


def test_region_boundaries_do_not_treat_longitude_seam_as_image_edge() -> None:
    mask = np.zeros((12, 24), dtype=bool)
    mask[4:8, :3] = True
    mask[4:8, -3:] = True

    boundary = S.region_boundaries(mask, connectivity=4)

    assert not boundary[5, 0]
    assert not boundary[5, -1]
    assert boundary[5, 2]
    assert boundary[5, -3]


def test_fill_holes_uses_solid_angle_and_never_crosses_missing_data() -> None:
    shape = (18, 36)
    mask = np.ones(shape, dtype=bool)
    mask[8:10, 17:19] = False
    mask[3:5, 4:6] = False
    validity = np.ones(shape, dtype=bool)
    validity[3, 3] = False
    weights = spherical_solid_angle_weights(shape)
    threshold = float(weights[8:10, 17:19].sum()) * 1.01

    result = S.fill_holes(
        mask,
        max_hole_solid_angle_sr=threshold,
        validity=validity,
    )

    assert result[8:10, 17:19].all()
    assert not result[3:5, 4:6].any()
    with pytest.raises(TypeError):
        S.fill_holes(mask, max_hole_solid_angle_sr=True)


def test_marker_watershed_is_periodic_deterministic_and_keeps_markers() -> None:
    shape = (8, 16)
    gradient = np.zeros(shape, dtype=np.float64)
    markers = np.zeros(shape, dtype=np.int32)
    markers[4, 1] = 7
    markers[4, 9] = 3
    validity = np.ones(shape, dtype=bool)
    validity[:, 5] = False

    first = S.watershed(
        gradient,
        markers,
        validity=validity,
        connectivity=4,
    )
    second = S.watershed(
        gradient,
        markers,
        validity=validity,
        connectivity=4,
    )

    np.testing.assert_array_equal(first, second)
    assert first[4, 1] == 7
    assert first[4, 9] == 3
    assert np.all(first[:, 5] == 0)
    assert first[4, -1] == 7


def test_region_measurements_cap_annulus_and_adjacency_have_analytic_oracles() -> None:
    shape = (36, 72)
    rays = _rays(shape)
    north = np.asarray((0.0, 1.0, 0.0))
    angle = np.rad2deg(np.arccos(np.clip(rays @ north, -1.0, 1.0)))
    labels = np.zeros(shape, dtype=np.int32)
    labels[angle <= 30.0] = 1
    labels[(angle > 35.0) & (angle <= 50.0)] = 2
    weights = spherical_solid_angle_weights(shape)

    measured = S.measure_regions(labels)

    assert tuple(measured.by_label) == (1, 2)
    for label in (1, 2):
        expected = float(weights[labels == label].sum())
        assert measured.by_label[label].solid_angle_sr == pytest.approx(expected)
        assert measured.by_label[label].support_fraction == pytest.approx(
            expected / (4.0 * math.pi)
        )
        assert measured.by_label[label].enclosing_cap.radius_deg <= 55.0
    np.testing.assert_allclose(
        measured.by_label[1].centroid_bearing_xyz,
        north,
        atol=2e-15,
    )

    split = np.ones((12, 24), dtype=np.int32)
    split[:, 12:] = 2
    graph = S.region_adjacency_graph(split)
    assert graph.labels == (1, 2)
    assert len(graph.edges) == 1
    assert graph.edges[0].shared_boundary_rad == pytest.approx(2.0 * math.pi)


def test_boundary_conversion_lifts_unit_bearings_and_length() -> None:
    labels = np.zeros((12, 24), dtype=np.int32)
    labels[3:9, 5:12] = 4

    result = S.extract_boundaries(labels)

    assert len(result.boundaries) == 1
    boundary = result.boundaries[0]
    assert boundary.label == 4
    assert boundary.angular_length_rad > 0.0
    np.testing.assert_allclose(
        np.linalg.norm(boundary.bearings_xyz, axis=1),
        1.0,
        atol=1e-12,
    )
    assert not boundary.bearings_xyz.flags.writeable


def test_ordered_contours_are_closed_continuous_and_seam_periodic() -> None:
    labels = np.zeros((12, 24), dtype=np.int32)
    labels[4:8, :3] = 1
    labels[4:8, -3:] = 1

    result = S.trace_contours(labels)

    assert result.shape_hw == labels.shape
    assert len(result.contours) == 1
    contour = result.contours[0]
    assert contour.label == 1
    assert contour.closed
    assert len(contour.bearings_xyz) == 20
    assert not contour.bearings_xyz.flags.writeable
    adjacent = np.rad2deg(
        np.arccos(
            np.clip(
                np.sum(
                    contour.bearings_xyz * np.roll(contour.bearings_xyz, -1, axis=0),
                    axis=1,
                ),
                -1.0,
                1.0,
            )
        )
    )
    assert float(adjacent.max()) <= 15.0 + 1e-12
    assert contour.angular_length_rad == pytest.approx(
        float(np.deg2rad(adjacent).sum()), abs=1e-12
    )
    with pytest.raises(ValueError, match="interface"):
        replace(contour, interface="panorai-spherical-region-analysis/unknown")


def test_polar_contour_has_one_loop_without_artificial_pole_edge() -> None:
    labels = np.zeros((12, 24), dtype=np.int32)
    labels[:3, :] = 1

    contour = S.trace_contours(labels).contours[0]

    assert contour.closed
    assert len(contour.bearings_xyz) == labels.shape[1]
    latitude_edge = math.pi / 2.0 - 3.0 * math.pi / labels.shape[0]
    expected_small_circle = 2.0 * math.pi * math.cos(latitude_edge)
    assert contour.angular_length_rad == pytest.approx(
        expected_small_circle,
        rel=0.01,
    )


def test_contours_open_at_missing_data_unless_support_boundary_is_requested() -> None:
    labels = np.zeros((10, 20), dtype=np.int32)
    labels[3:7, 6:12] = 1
    validity = np.ones_like(labels, dtype=bool)
    validity[:, 11] = False

    open_result = S.trace_contours(labels, validity=validity)
    closed_result = S.trace_contours(
        labels,
        validity=validity,
        include_support_boundary=True,
    )

    assert len(open_result.contours) == 1
    assert not open_result.contours[0].closed
    assert len(closed_result.contours) == 1
    assert closed_result.contours[0].closed


def test_intrinsic_skeleton_is_a_seam_periodic_subset_and_rejects_full_sphere() -> None:
    mask = np.zeros((18, 36), dtype=bool)
    mask[6:12, :4] = True
    mask[6:12, -4:] = True

    skeleton = S.skeletonize(mask)

    assert skeleton.dtype == np.bool_
    assert np.any(skeleton)
    assert np.all(~skeleton | mask)
    assert S.connected_components(skeleton).max() == 1
    assert np.any(skeleton[:, 0])
    assert np.any(skeleton[:, -1])
    with pytest.raises(ValueError, match="supported domain"):
        S.skeletonize(np.ones_like(mask))
    partial_support = np.ones_like(mask)
    partial_support[:, 0] = False
    with pytest.raises(ValueError, match="supported domain"):
        S.skeletonize(np.ones_like(mask), support=partial_support)


def test_contours_and_skeleton_are_equivariant_to_longitude_roll() -> None:
    shape = (24, 48)
    shift = 11
    mask = np.zeros(shape, dtype=bool)
    mask[7:18, :6] = True
    mask[7:18, -5:] = True

    contour = S.trace_contours(mask.astype(np.int32)).contours[0]
    rolled_contour = S.trace_contours(
        np.roll(mask, shift, axis=1).astype(np.int32)
    ).contours[0]
    yaw = shift * 2.0 * math.pi / shape[1]
    rotation = np.asarray(
        (
            (math.cos(yaw), 0.0, math.sin(yaw)),
            (0.0, 1.0, 0.0),
            (-math.sin(yaw), 0.0, math.cos(yaw)),
        )
    )
    rotated_bearings = contour.bearings_xyz @ rotation.T
    similarities = rotated_bearings @ rolled_contour.bearings_xyz.T

    assert contour.closed and rolled_contour.closed
    assert contour.angular_length_rad == pytest.approx(
        rolled_contour.angular_length_rad,
        abs=1e-12,
    )
    assert np.all(np.max(similarities, axis=1) > 1.0 - 1e-12)
    np.testing.assert_array_equal(
        S.skeletonize(np.roll(mask, shift, axis=1)),
        np.roll(S.skeletonize(mask), shift, axis=1),
    )


def test_skeleton_configured_operation_composes_and_persists(tmp_path) -> None:
    mask = np.zeros((18, 36), dtype=bool)
    mask[5:13, 12:24] = True
    panorama = pa.EquirectangularImage.from_field(
        "mask",
        mask,
        spec=RasterFieldSpec.categorical(),
    )
    pipeline = pa.Pipeline(pa.OnSphere(S.Skeletonize(), input="mask", output="mask"))

    expected = pipeline(panorama)
    restored = pa.Pipeline.load(pipeline.save(tmp_path / "skeleton"))
    actual = restored(panorama)

    np.testing.assert_array_equal(actual.values, expected.values)
    assert isinstance(restored.named_steps["on_sphere"].operation, S.Skeletonize)
    assert expected.grid.transforms[-1].operation == "skeletonize"


def test_complete_region_analysis_is_cacheable_without_losing_contour_order(
    tmp_path,
) -> None:
    labels = np.zeros((12, 24), dtype=np.int32)
    labels[4:8, :3] = 1
    labels[4:8, -3:] = 1
    panorama = pa.EquirectangularImage.constant(0, height=12, width=24).with_labels(
        labels
    )
    pipeline = pa.Pipeline(
        AnalyzeRegions(include_contours=True),
        memory=tmp_path / "cache",
    )

    cold = pipeline.run(panorama)
    warm = pipeline.run(panorama)

    assert cold.manifest()["cache"]["hit"] is False
    assert warm.manifest()["cache"]["hit"] is True
    assert warm.output.contours is not None
    assert cold.output.contours is not None
    np.testing.assert_array_equal(
        warm.output.contours.contours[0].bearings_xyz,
        cold.output.contours.contours[0].bearings_xyz,
    )


def test_non_axis_rotation_preserves_component_count_and_area_with_lattice_error() -> (
    None
):
    shape = (72, 144)
    rays = _rays(shape)
    center = np.asarray((0.3, 0.4, 0.866), dtype=np.float64)
    center /= np.linalg.norm(center)
    mask = np.rad2deg(np.arccos(np.clip(rays @ center, -1.0, 1.0))) <= 18.0
    angle = math.radians(29.0)
    rotation = np.asarray(
        (
            (math.cos(angle), 0.0, math.sin(angle)),
            (0.0, 1.0, 0.0),
            (-math.sin(angle), 0.0, math.cos(angle)),
        )
    )
    rotated = S.rotate(mask.astype(np.uint8), rotation, interpolation="nearest").astype(
        bool
    )

    original_labels = S.connected_components(mask)
    rotated_labels = S.connected_components(rotated)
    original = S.measure_regions(original_labels).regions[0]
    transformed = S.measure_regions(rotated_labels).regions[0]

    assert original_labels.max() == rotated_labels.max() == 1
    assert transformed.solid_angle_sr == pytest.approx(
        original.solid_angle_sr,
        rel=0.04,
    )


def test_configured_operations_compose_with_container_pipeline_and_persistence(
    tmp_path,
) -> None:
    mask = np.zeros((18, 36), dtype=bool)
    mask[6:12, :3] = True
    mask[6:12, -3:] = True
    support = np.ones_like(mask)
    validity = support.copy()
    panorama = pa.EquirectangularImage.from_field(
        "mask",
        mask,
        spec=RasterFieldSpec.categorical(),
        support=support,
        valid=validity,
    )
    operation = S.ConnectedComponents(connectivity=8)
    pipeline = pa.Pipeline(pa.OnSphere(operation, input="mask", output="labels"))
    analysis_pipeline = pa.Pipeline(
        pa.OnSphere(operation, input="mask", output="labels"),
        AnalyzeRegions(include_contours=True),
    )

    field = pipeline(panorama)
    composed_analysis = analysis_pipeline(panorama)
    container = panorama.apply_spherical(
        operation,
        input="mask",
        output="labels",
        spec=RasterFieldSpec.categorical(),
    )

    assert field.kind == "labels"
    assert field.values.max() == 1
    np.testing.assert_array_equal(container.labels, field.values)
    assert field.grid.transforms[-1].operation == "connected_components"
    assert len(composed_analysis.measurements.regions) == 1
    assert composed_analysis.contours is not None
    assert len(composed_analysis.contours.contours) == 1
    restored = pa.Pipeline.load(pipeline.save(tmp_path / "regions"))
    actual = restored(panorama)
    np.testing.assert_array_equal(actual.values, field.values)
    assert isinstance(
        restored.named_steps["on_sphere"].operation, S.ConnectedComponents
    )
    assert evaluate_binary_regions(field.values > 0, mask).iou == 1.0

    analyzed_panorama = panorama.with_field(
        "labels",
        field.values,
        spec=RasterFieldSpec.categorical(),
        support=field.support,
        valid=field.validity,
    )
    analysis = pa.Pipeline(AnalyzeRegions())(analyzed_panorama)
    assert len(analysis.measurements.regions) == 1
    assert analysis.adjacency.labels == (1,)
    assert analysis.boundaries is not None


def test_watershed_stage_reads_two_typed_fields_and_is_persistable(tmp_path) -> None:
    shape = (10, 20)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    gradient = np.zeros(shape, dtype=np.float64)
    markers = np.zeros(shape, dtype=np.int32)
    markers[5, 2] = 1
    markers[5, 12] = 2
    panorama = (
        pa.EquirectangularImage.fromarray(image)
        .with_field(
            "features",
            gradient,
            spec=RasterFieldSpec.continuous(),
        )
        .with_labels(markers)
    )
    stage = SphericalWatershed()
    pipeline = pa.Pipeline(stage)

    field = pipeline(panorama)

    assert field.kind == "labels"
    assert field.values[5, 2] == 1
    assert field.values[5, 12] == 2
    restored = pa.Pipeline.load(pipeline.save(tmp_path / "watershed"))
    np.testing.assert_array_equal(restored(panorama).values, field.values)


def test_region_analysis_import_does_not_load_torch() -> None:
    before = set(sys.modules)
    __import__("panorai.ops.spherical")
    loaded = set(sys.modules) - before
    assert not any(name == "torch" or name.startswith("torch.") for name in loaded)
