from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from panorai.evaluation import evaluate_surface_reconstruction
from panorai.reconstruction import SurfaceMesh


def _square(*, z: float = 0.0, normal_z: float = 1.0) -> SurfaceMesh:
    vertices = np.asarray(
        (
            (-1.0, -1.0, z),
            (1.0, -1.0, z),
            (1.0, 1.0, z),
            (-1.0, 1.0, z),
            (0.0, 0.0, z),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32)
    normals = np.repeat(np.asarray(((0.0, 0.0, normal_z),)), len(vertices), axis=0)
    return SurfaceMesh(
        vertices_xyz=vertices,
        faces=faces,
        normals_xyz=normals,
        vertex_weight=np.ones(len(vertices), dtype=np.float32),
        vertex_view_count=np.full(len(vertices), 3, dtype=np.uint16),
    )


def test_parallel_square_distance_is_exact_point_to_triangle_oracle() -> None:
    reference = _square(z=0.0)
    prediction = _square(z=0.125)

    report = evaluate_surface_reconstruction(
        prediction,
        reference,
        distance_threshold=0.1,
    )

    assert report.accuracy_mean == pytest.approx(0.125, abs=1e-12)
    assert report.accuracy_median == pytest.approx(0.125, abs=1e-12)
    assert report.completeness_mean_distance == pytest.approx(0.125, abs=1e-12)
    assert report.completeness_median_distance == pytest.approx(0.125, abs=1e-12)
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.fscore == 0.0
    assert report.normal_consistency == pytest.approx(1.0, abs=1e-12)
    assert report.median_normal_error_deg == pytest.approx(0.0, abs=1e-12)


def test_surface_evaluator_detects_opposite_normal_orientation() -> None:
    reference = _square()
    prediction = _square(normal_z=-1.0)

    report = evaluate_surface_reconstruction(
        prediction,
        reference,
        distance_threshold=0.01,
    )

    assert report.fscore == 1.0
    assert report.normal_consistency == pytest.approx(-1.0, abs=1e-12)
    assert report.median_normal_error_deg == pytest.approx(180.0, abs=1e-12)


def test_reference_observability_is_external_and_reported_separately() -> None:
    reference = _square()
    observed = np.asarray((True, False, True, False, True), dtype=bool)

    report = evaluate_surface_reconstruction(
        reference,
        reference,
        distance_threshold=0.01,
        reference_observed=observed,
    )

    assert report.observed_surface_coverage == pytest.approx(0.6)
    assert report.reference_sample_count == 3
    assert report.recall == 1.0


def test_surface_evaluator_rejects_frame_units_and_degenerate_faces() -> None:
    mesh = _square()
    with pytest.raises(ValueError, match="share units and frame"):
        evaluate_surface_reconstruction(
            mesh,
            replace(mesh, units="cm"),
            distance_threshold=0.1,
        )

    degenerate = SurfaceMesh(
        vertices_xyz=np.asarray(((0.0, 0.0, 0.0),) * 3),
        faces=np.asarray(((0, 1, 2),)),
        normals_xyz=np.asarray(((0.0, 0.0, 1.0),) * 3),
        vertex_weight=np.ones(3),
        vertex_view_count=np.ones(3),
    )
    with pytest.raises(ValueError, match="degenerate faces"):
        evaluate_surface_reconstruction(
            mesh,
            degenerate,
            distance_threshold=0.1,
        )
