from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_geometry.py"
SPEC = importlib.util.spec_from_file_location("benchmark_geometry", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_nearest_rank_summary_keeps_raw_samples() -> None:
    summary = benchmark.summarize_seconds([0.4, 0.1, 0.3, 0.2])

    assert summary["raw"] == [0.4, 0.1, 0.3, 0.2]
    assert summary["median"] == pytest.approx(0.25)
    assert summary["p95_nearest_rank"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("512x1024:256", (512, 1024, 256)), ("17X31:9", (17, 31, 9))],
)
def test_resolution_parser(text: str, expected: tuple[int, int, int]) -> None:
    assert benchmark.parse_resolution(text) == expected


def test_case_matrix_records_support_gaps_instead_of_fabricating_cases() -> None:
    cases = benchmark.build_cases(
        [(32, 64, 16)],
        ["panorai", "py360convert", "pyequilib"],
        "cpu",
    )

    assert not any(
        case.library == "panorai" and case.backend == "numpy" and case.batch == 8
        for case in cases
    )
    assert not any(
        case.library == "py360convert" and case.operation == "gnomonic_to_erp"
        for case in cases
    )
    assert any(
        case.library == "pyequilib"
        and case.backend == "torch"
        and case.batch == 8
        and case.operation == "cubemap_to_erp"
        for case in cases
    )
    assert len({case.case_id for case in cases}) == len(cases)


def test_default_matrix_has_required_resolutions_and_interpolations() -> None:
    cases = benchmark.build_cases(benchmark.DEFAULT_RESOLUTIONS, ["panorai"], "cpu")

    assert {(case.erp_height, case.erp_width, case.face_size) for case in cases} == set(
        benchmark.DEFAULT_RESOLUTIONS
    )
    assert {case.interpolation for case in cases} == {"nearest", "bilinear"}
    assert {case.operation for case in cases} == set(benchmark.OPERATIONS)
    assert {case.batch for case in cases if case.backend == "torch"} == {1, 8}


def test_correctness_guard_allows_documented_fill_only_outside_support() -> None:
    data = np.full((4, 8, 3), np.nan, dtype=np.float32)
    support = np.zeros((4, 8), dtype=bool)
    support[1:3, 2:6] = True
    data[support] = 1.0
    result = SimpleNamespace(data=data, support_mask=support)
    case = benchmark.Case("panorai", "numpy", 1, "gnomonic_to_erp", "bilinear", 4, 8, 2)

    guard = benchmark._validate_output(case, result, data.shape)

    assert guard["passed"] is True
    assert guard["finite_on_support"] is True
    assert guard["nonfinite_outside_support_allowed"] is True
