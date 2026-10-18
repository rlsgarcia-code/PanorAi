from __future__ import annotations

import numpy as np
import pytest

from panorai.applications import (
    ADCensusStereoDepth,
    Application,
    CensusMGMStereoDepth,
    ELASStereoDepth,
    PatchMatchStereoDepth,
    StereoBMDepth,
    StereoSGBMDepth,
)
from panorai.stereo.classical import (
    ADCensusConfig,
    CensusMGMConfig,
    ELASConfig,
    PatchMatchConfig,
    RectifiedStereoGeometry,
    RectifiedStereoPair,
    RefineStereoDisparity,
    StereoDisparityCandidate,
    StereoBMConfig,
    StereoSGBMConfig,
)
from panorai.pipeline import Pipeline


DISPARITY = 6


def _translated_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    reference = rng.random((32, 96), dtype=np.float32)
    source = np.zeros_like(reference)
    source[:, :-DISPARITY] = reference[:, DISPARITY:]
    source[:, -DISPARITY:] = rng.random(
        (reference.shape[0], DISPARITY), dtype=np.float32
    )
    return reference, source


def _applications():
    return (
        StereoBMDepth(
            StereoBMConfig(max_disparity=15, block_size=9, uniqueness_ratio=0)
        ),
        ADCensusStereoDepth(
            ADCensusConfig(
                max_disparity=12,
                aggregation_arm=2,
                aggregation_passes=1,
                paths=4,
            )
        ),
        StereoSGBMDepth(
            StereoSGBMConfig(max_disparity=15, block_size=5, uniqueness_ratio=0)
        ),
        CensusMGMStereoDepth(
            CensusMGMConfig(max_disparity=12, aggregation_window=3, variant="mgm")
        ),
        ELASStereoDepth(
            ELASConfig(max_disparity=12, aggregation_window=3, support_step=4)
        ),
        PatchMatchStereoDepth(
            PatchMatchConfig(max_disparity=12, iterations=1, patch_radius=1)
        ),
    )


@pytest.mark.parametrize("application", _applications())
def test_six_reference_pipelines_recover_closed_form_translation(application) -> None:
    reference, source = _translated_pair()

    result = application(reference, source)
    composed = application.as_pipeline()(RectifiedStereoPair(reference, source))

    evaluation = result.validity[:, 20:-10]
    recovered = result.disparity[:, 20:-10][evaluation]
    assert recovered.size > 0.45 * evaluation.size
    assert np.median(recovered) == pytest.approx(DISPARITY, abs=0.35)
    assert np.mean(np.abs(recovered - DISPARITY) < 1.0) > 0.85
    assert np.all(np.isnan(result.disparity[~result.validity]))
    assert np.all(result.confidence[~result.validity] == 0.0)
    np.testing.assert_array_equal(result.disparity, composed.disparity)
    np.testing.assert_array_equal(result.validity, composed.validity)
    np.testing.assert_array_equal(result.confidence, composed.confidence)


def test_pipeline_exposes_reusable_cost_aggregation_and_optimization() -> None:
    application = ADCensusStereoDepth(
        ADCensusConfig(max_disparity=12, aggregation_arm=2, paths=4)
    )

    names = tuple(application.as_pipeline().named_steps)

    assert names == (
        "matching_cost",
        "cost_aggregation",
        "optimization",
        "selection",
        "validation",
    )

    restored = Pipeline.from_config(application.as_pipeline().to_config())
    assert tuple(restored.named_steps) == names


def test_metric_geometry_returns_radial_not_optical_axis_range() -> None:
    reference, source = _translated_pair()
    geometry = RectifiedStereoGeometry(
        focal_length_x=80.0,
        focal_length_y=80.0,
        baseline=0.30,
    )
    result = StereoSGBMDepth(StereoSGBMConfig(max_disparity=15, uniqueness_ratio=0))(
        reference, source, geometry=geometry
    )
    assert result.radial_range is not None
    y = reference.shape[0] // 2
    x = reference.shape[1] // 2
    expected = geometry.focal_length_x * geometry.baseline / DISPARITY
    assert result.radial_range[y, x] == pytest.approx(expected, rel=0.03)


def test_metric_validation_rejects_non_positive_disparity() -> None:
    reference = np.linspace(0.0, 1.0, 16 * 32, dtype=np.float32).reshape(16, 32)
    pair = RectifiedStereoPair(
        reference,
        reference,
        geometry=RectifiedStereoGeometry(focal_length_x=20.0, baseline=0.5),
    )
    zeros = np.zeros(pair.shape_hw, dtype=np.float32)
    candidate = StereoDisparityCandidate(
        pair=pair,
        left=zeros,
        right=zeros,
        left_cost=zeros,
        left_second_cost=np.ones(pair.shape_hw, dtype=np.float32),
        backend_validity=np.ones(pair.shape_hw, dtype=bool),
        metadata={},
    )

    result = (
        RefineStereoDisparity(
            method="regression",
            uniqueness_margin=0.0,
            left_right_tolerance=0.0,
        )
        .execute(candidate)
        .value
    )

    assert not np.any(result.validity)
    assert np.all(np.isnan(result.radial_range))


def test_support_is_distinct_from_zero_intensity_and_invalidates_output() -> None:
    reference, source = _translated_pair()
    reference[:, 40:44] = 0.0
    source[:, 34:38] = 0.0
    support = np.ones(reference.shape, dtype=bool)
    support[:, 50:54] = False

    result = ADCensusStereoDepth(
        ADCensusConfig(max_disparity=12, aggregation_arm=2, paths=4)
    )(reference, source, reference_support=support)

    assert np.any(result.validity[:, 40:44])
    assert not np.any(result.validity[:, 50:54])


def test_rectified_pair_rejects_non_boolean_support() -> None:
    reference, source = _translated_pair()
    with pytest.raises(TypeError, match="must be boolean"):
        RectifiedStereoPair(
            reference,
            source,
            reference_support=np.ones(reference.shape, dtype=np.uint8),
        )


def test_registered_classical_application_round_trip(tmp_path) -> None:
    application = CensusMGMStereoDepth(
        CensusMGMConfig(max_disparity=12, variant="tmgm")
    )

    restored = Application.load(application.save(tmp_path / "mgm.json"))

    assert restored == application
