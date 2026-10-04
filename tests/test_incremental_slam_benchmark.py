from __future__ import annotations

import pytest

from scripts.benchmark_incremental_erp_slam import (
    _StageTimings,
    nearest_rank_percentile,
    summarize_samples,
    summarize_values,
)


def test_nearest_rank_and_summary_preserve_every_raw_sample():
    values = [4.0, 1.0, 5.0, 2.0, 3.0]
    assert nearest_rank_percentile(values, 0.5) == 3.0
    assert nearest_rank_percentile(values, 0.95) == 5.0
    summary = summarize_values(values)
    assert summary == {
        "count": 5,
        "raw": values,
        "minimum": 1.0,
        "median": 3.0,
        "p95_nearest_rank": 5.0,
        "maximum": 5.0,
    }
    with pytest.raises(ValueError, match="cannot be empty"):
        summarize_values([])
    with pytest.raises(ValueError, match="probability"):
        nearest_rank_percentile(values, 0.0)


def test_stage_timings_preserve_calls_and_make_inclusive_semantics_explicit():
    timings = _StageTimings()
    with timings.measure("outer"):
        with timings.measure("inner"):
            sum(range(50))
    record = timings.to_dict()
    assert record["outer"]["call_count"] == 1
    assert record["inner"]["call_count"] == 1
    assert record["outer"]["total_seconds"] >= record["inner"]["total_seconds"]
    assert record["outer"]["calls_seconds"] == [
        record["outer"]["total_seconds"]
    ]


def _sample(seconds: float, *, strict: bool = True):
    return {
        "measured_seconds": seconds + 0.5,
        "slam_seconds": seconds,
        "peak_rss_bytes_process_high_watermark": 1000.0 + seconds,
        "stage_timings_inclusive": {
            "feature_extraction": {
                "total_seconds": seconds * 0.6,
                "call_count": 3,
                "calls_seconds": [seconds * 0.2] * 3,
            }
        },
        "result": {"registered_frame_count": 3},
        "correctness": {
            "strict_success": strict,
            "median_rotation_error_deg": 0.4,
            "median_translation_direction_error_deg": 0.5,
        },
    }


def test_sample_summary_keeps_correctness_as_performance_gate():
    summary = summarize_samples([_sample(3.0), _sample(1.0), _sample(2.0)])
    assert summary["all_strictly_correct"] is True
    assert summary["slam_seconds"]["raw"] == [3.0, 1.0, 2.0]
    assert summary["slam_seconds"]["median"] == 2.0
    assert summary["seconds_per_registered_frame"]["median"] == pytest.approx(
        2.0 / 3.0
    )
    assert summary["registered_frames_per_second"]["median"] == 1.5
    assert summary["stage_seconds_inclusive"]["feature_extraction"][
        "median"
    ] == pytest.approx(1.2)
    failed = summarize_samples([_sample(1.0), _sample(2.0, strict=False)])
    assert failed["all_strictly_correct"] is False
