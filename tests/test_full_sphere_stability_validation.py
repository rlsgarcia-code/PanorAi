from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    evaluate_result,
    make_candidate,
    registered_pair,
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import PairData


def _pair(shape: tuple[int, int] = (6, 12)) -> PairData:
    image = np.zeros((*shape, 3), dtype=np.uint8)
    support = np.ones(shape, dtype=bool)
    radial = np.full(shape, 2.0, dtype=np.float32)
    return PairData(
        "fixture",
        "pair",
        image,
        image.copy(),
        support,
        support.copy(),
        radial,
        radial.copy(),
        np.eye(3),
        np.asarray((0.1, 0.0, 0.0)),
        {"license": "analytic fixture"},
    )


def test_solid_angle_weights_are_larger_at_equator_than_poles():
    weights = solid_angle_weights((8, 16))

    assert weights.shape == (8, 16)
    assert np.all(weights > 0.0)
    assert weights[3, 0] > weights[0, 0]
    np.testing.assert_allclose(weights[:, 0], weights[:, -1])


def test_candidate_is_the_real_experimental_full_sphere_facade():
    candidate = make_candidate(
        face_size=32,
        face_fov_deg=100.0,
        min_range_m=0.3,
        max_range_m=12.0,
        hypotheses=12,
        filter_backend="numpy",
    )

    assert candidate.application_id == "panorai.applications.FullSphereDenseStereo"
    assert candidate.stability == "experimental"
    assert candidate.options.planner.face_size == 32
    assert candidate.options.stereo.num_hypotheses == 12


def test_registered_pair_keeps_metric_directed_pose():
    pair = _pair()

    registered = registered_pair(pair)

    assert registered.reference_id == "fixture:reference"
    assert len(registered.sources) == 1
    source = registered.sources[0]
    assert source.pose.scale_status == "metric"
    assert source.pose.units == "m"
    np.testing.assert_allclose(
        source.pose.translation_source_from_reference,
        pair.translation_source_from_reference,
    )


def test_evaluation_uses_fixed_population_and_prediction_validity_separately():
    pair = _pair()
    validity = np.ones(pair.reference_range.shape, dtype=bool)
    validity[2, 3] = False
    estimated = pair.reference_range.copy()
    estimated[2, 3] = np.nan
    result = SimpleNamespace(
        radial_range=estimated,
        validity=validity,
        sampling_support=np.ones_like(validity),
    )

    evaluation, non_sky, population = evaluate_result(
        pair,
        result,
        min_range_m=0.3,
        max_range_m=12.0,
    )

    overall = evaluation["metrics"]["overall"]
    assert np.all(non_sky)
    assert np.all(population)
    assert overall["population"] == population.size
    assert overall["predicted_valid"] == population.size - 1
    assert overall["coverage"] < 1.0
    assert overall["delta_1_25"] == overall["coverage"]
