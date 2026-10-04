from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from panorai.estimators import (
    ModelCompetitionReport,
    ModelEvidence,
    PoseStabilityReport,
    RelativePoseQualityReport,
    TranslationOrientationReport,
)
from scripts.evaluate_cross_domain_pose_calibration import (
    CalibrationRow,
    evaluate_rows,
)


def _report(score: float) -> RelativePoseQualityReport:
    evidence = ModelEvidence("essential", 10, 0.5, 0.5, 0.1)
    return RelativePoseQualityReport(
        num_correspondences=20,
        num_inliers=10,
        inlier_ratio=0.5,
        occupied_cells_a=4,
        occupied_cells_b=4,
        coverage_entropy_a=0.5,
        coverage_entropy_b=0.5,
        median_residual_deg=0.1,
        p90_residual_deg=0.2,
        median_parallax_deg=2.0,
        cheirality_ratio=0.8,
        translation_orientation=TranslationOrientationReport(
            4, 10, 8, 2, 0.8, 0.6, 2.0, False
        ),
        stability=PoseStabilityReport(2, 2, 0.1, 0.2, 0.3, 0.4),
        model_competition=ModelCompetitionReport(
            evidence,
            replace(evidence, model="rotation-only"),
            replace(evidence, model="spherical-homography"),
            "essential",
            0.2,
        ),
        raw_quality_score=score,
        accepted=True,
        rejection_reasons=(),
    )


def _rows() -> list[CalibrationRow]:
    rows: list[CalibrationRow] = []
    for domain_index, domain in enumerate(("matterport360", "stanford2d3d")):
        for group_index in range(3):
            for sample_index, score in enumerate((0.1, 0.4, 0.7, 0.9)):
                rows.append(
                    CalibrationRow(
                        sample_id=f"{domain}-{group_index}-{sample_index}",
                        domain=domain,
                        group_id=f"{domain}-group-{group_index}",
                        success=bool(score >= (0.6 + 0.1 * domain_index)),
                        report=_report(score),
                    )
                )
    return rows


def test_cross_domain_evaluation_is_deterministic_and_disjoint() -> None:
    first = evaluate_rows(_rows(), bootstrap_repetitions=50, seed=17)
    second = evaluate_rows(_rows(), bootstrap_repetitions=50, seed=17)

    assert first == second
    assert first["schema"] == "panorai-cross-domain-pose-calibration/v1"
    assert set(first["cross_domain"]) == {
        "matterport360_to_stanford2d3d",
        "stanford2d3d_to_matterport360",
    }
    for result in first["cross_domain"].values():
        assert result["sample_id_overlap"] == 0
        assert result["calibration_group_count"] == 3
        assert result["evaluation_group_count"] == 3
        assert result["spatial_group_bootstrap"]["repetitions"] == 50
        assert np.isfinite(result["calibrated"]["brier_score"])


def test_cross_domain_evaluation_rejects_missing_domain_or_bad_bootstrap() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="both domains"):
        evaluate_rows(
            [row for row in rows if row.domain == "matterport360"],
            bootstrap_repetitions=5,
        )
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_rows(rows, bootstrap_repetitions=0)


def test_group_held_out_metrics_do_not_train_on_the_held_group() -> None:
    result = evaluate_rows(_rows(), bootstrap_repetitions=5, seed=3)
    for domain in ("matterport360", "stanford2d3d"):
        within = result["within_domain_group_held_out"][domain]
        assert within["protocol"] == "leave-one-spatial-group-out"
        assert within["group_count"] == 3
        assert within["count"] == 12
        assert 0.0 <= within["metrics"]["brier_score"] <= 1.0
