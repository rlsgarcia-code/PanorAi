from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from panorai import EquirectangularImage
from panorai.ops import spherical as S


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/_static/tutorials/poly-haven-studio-erp.jpg"
SOURCE_SHA256 = "9b6e2b7521e2cf35d998f4087840c4bf98b480467bd753d719533228c984be8a"
WORKING_SIZE_WH = (360, 180)
LUMINANCE_QUANTILE = 0.90
MAX_HOLE_SOLID_ANGLE_SR = 0.05


def _source_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    panorama = EquirectangularImage.open(SOURCE).resize(WORKING_SIZE_WH)
    rgb = np.asarray(panorama.data, dtype=np.uint8)
    normalized = rgb.astype(np.float64) / 255.0
    luminance = (
        0.2126 * normalized[..., 0]
        + 0.7152 * normalized[..., 1]
        + 0.0722 * normalized[..., 2]
    )
    smoothed = np.asarray(S.gaussian_blur(luminance, ksize=7, sigma=1.5))
    candidate = smoothed >= np.quantile(smoothed, LUMINANCE_QUANTILE)
    return rgb, luminance, candidate


def _gradient_and_markers(luminance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    smoothed = S.gaussian_blur(luminance, ksize=7, sigma=1.5)
    gradient = np.asarray(S.gradient(smoothed, operator="scharr").magnitude)
    markers = np.zeros(luminance.shape, dtype=np.int32)
    windows = (
        (20, 90, 0, 180),
        (20, 90, 180, 360),
        (90, 160, 0, 180),
        (90, 160, 180, 360),
    )
    for label, (row_start, row_stop, column_start, column_stop) in enumerate(
        windows,
        start=1,
    ):
        window = gradient[row_start:row_stop, column_start:column_stop]
        local = np.unravel_index(int(np.argmin(window)), window.shape)
        markers[row_start + local[0], column_start + local[1]] = label
    return gradient, markers


def _measurement_signature(labels: np.ndarray) -> list[tuple[int, float, float]]:
    measurements = S.measure_regions(labels)
    return [
        (
            region.label,
            region.solid_angle_sr,
            region.angular_perimeter_rad,
        )
        for region in measurements.regions
    ]


def test_every_region_operation_runs_on_checksum_pinned_real_erp() -> None:
    rgb, luminance, candidate = _source_fields()
    assert rgb.shape == (180, 360, 3)
    assert rgb.dtype == np.uint8
    assert candidate.dtype == np.bool_
    assert np.isclose(candidate.mean(), 0.10)

    components = S.connected_components(candidate, connectivity=8)
    assert int(components.max()) == 25
    component_measurements = S.measure_regions(components)
    largest = max(
        component_measurements.regions,
        key=lambda region: (region.solid_angle_sr, -region.label),
    )
    target = components == largest.label
    distance = S.geodesic_distance_transform(target, units="degrees")
    assert np.isfinite(distance).all()
    assert float(distance.min()) == 0.0
    assert 80.0 < float(distance.max()) < 180.0

    filled = S.fill_holes(
        candidate,
        max_hole_solid_angle_sr=MAX_HOLE_SOLID_ANGLE_SR,
        connectivity=8,
    )
    assert np.count_nonzero(filled & ~candidate) == 192
    assert np.all(filled[candidate])

    gradient, markers = _gradient_and_markers(luminance)
    partition = S.watershed(gradient, markers, connectivity=8)
    unique, counts = np.unique(partition, return_counts=True)
    np.testing.assert_array_equal(unique, (1, 2, 3, 4))
    np.testing.assert_array_equal(counts, (20004, 13311, 18598, 12887))

    boundaries = S.region_boundaries(partition, connectivity=4)
    measurements = S.measure_regions(partition)
    adjacency = S.region_adjacency_graph(partition)
    lifted = S.extract_boundaries(partition)
    assert np.count_nonzero(boundaries) > 1000
    assert len(measurements.regions) == 4
    assert math.isclose(
        sum(region.solid_angle_sr for region in measurements.regions),
        4.0 * math.pi,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert len(adjacency.edges) == 4
    assert len(lifted.boundaries) == 4
    for boundary in lifted.boundaries:
        assert boundary.bearings_xyz.shape[1] == 3
        assert boundary.angular_length_rad > 0.0


def test_real_image_region_results_are_longitude_roll_equivariant() -> None:
    _, luminance, candidate = _source_fields()
    shift = 37
    rolled_candidate = np.roll(candidate, shift, axis=1)
    components = S.connected_components(candidate, connectivity=8)
    rolled_components = S.connected_components(rolled_candidate, connectivity=8)

    original_areas = sorted(
        region.solid_angle_sr for region in S.measure_regions(components).regions
    )
    rolled_areas = sorted(
        region.solid_angle_sr for region in S.measure_regions(rolled_components).regions
    )
    np.testing.assert_allclose(rolled_areas, original_areas, rtol=0.0, atol=1e-12)

    largest_label = max(
        S.measure_regions(components).regions,
        key=lambda region: region.solid_angle_sr,
    ).label
    target = components == largest_label
    distance = S.geodesic_distance_transform(target, units="degrees")
    rolled_distance = S.geodesic_distance_transform(
        np.roll(target, shift, axis=1),
        units="degrees",
    )
    np.testing.assert_allclose(
        rolled_distance,
        np.roll(distance, shift, axis=1),
        rtol=0.0,
        atol=1e-12,
    )

    filled = S.fill_holes(
        candidate,
        max_hole_solid_angle_sr=MAX_HOLE_SOLID_ANGLE_SR,
    )
    rolled_filled = S.fill_holes(
        rolled_candidate,
        max_hole_solid_angle_sr=MAX_HOLE_SOLID_ANGLE_SR,
    )
    np.testing.assert_array_equal(rolled_filled, np.roll(filled, shift, axis=1))

    gradient, markers = _gradient_and_markers(luminance)
    partition = S.watershed(gradient, markers, connectivity=8)
    rolled_partition = S.watershed(
        np.roll(gradient, shift, axis=1),
        np.roll(markers, shift, axis=1),
        connectivity=8,
    )
    np.testing.assert_array_equal(
        rolled_partition,
        np.roll(partition, shift, axis=1),
    )
    np.testing.assert_array_equal(
        S.region_boundaries(rolled_partition),
        np.roll(S.region_boundaries(partition), shift, axis=1),
    )

    original_signature = _measurement_signature(partition)
    rolled_signature = _measurement_signature(rolled_partition)
    np.testing.assert_allclose(
        rolled_signature,
        original_signature,
        rtol=0.0,
        atol=1e-12,
    )
    original_edges = S.region_adjacency_graph(partition).edges
    rolled_edges = S.region_adjacency_graph(rolled_partition).edges
    assert tuple(edge.labels for edge in rolled_edges) == tuple(
        edge.labels for edge in original_edges
    )
    np.testing.assert_allclose(
        [edge.shared_boundary_rad for edge in rolled_edges],
        [edge.shared_boundary_rad for edge in original_edges],
        rtol=0.0,
        atol=1e-12,
    )
