from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
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


@pytest.mark.parametrize("operation", ["erp_to_cubemap", "cubemap_to_erp"])
def test_py360convert_adapter_does_not_prewarm_timed_transform(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls = {"e2c": 0, "c2e": 0}
    fixture_shapes: list[tuple[int, int]] = []
    real_gradient = benchmark._gradient_hwc

    def tracked_gradient(height: int, width: int) -> np.ndarray:
        fixture_shapes.append((height, width))
        return real_gradient(height, width)

    monkeypatch.setattr(benchmark, "_gradient_hwc", tracked_gradient)
    fake = ModuleType("py360convert")

    def e2c(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        calls["e2c"] += 1
        return {name: np.zeros((4, 4, 3)) for name in "FRBLUD"}

    def c2e(cube: object, *args: object, **kwargs: object) -> np.ndarray:
        calls["c2e"] += 1
        assert isinstance(cube, dict)
        assert tuple(cube) == ("F", "R", "B", "L", "U", "D")
        return np.zeros((8, 16, 3))

    fake.e2c = e2c  # type: ignore[attr-defined]
    fake.c2e = c2e  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "py360convert", fake)
    case = benchmark.Case("py360convert", "numpy", 1, operation, "bilinear", 8, 16, 4)

    timed_operation, _ = benchmark._py360convert_operation(case)

    assert calls == {"e2c": 0, "c2e": 0}
    assert ((8, 16) in fixture_shapes) is (operation == "erp_to_cubemap")
    timed_operation()
    assert calls["e2c" if operation == "erp_to_cubemap" else "c2e"] == 1


@pytest.mark.parametrize("operation", ["erp_to_cubemap", "cubemap_to_erp"])
def test_pyequilib_adapter_does_not_prewarm_timed_transform(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls = {"equi2cube": 0, "cube2equi": 0}
    fixture_shapes: list[tuple[int, int]] = []
    real_gradient = benchmark._gradient_hwc

    def tracked_gradient(height: int, width: int) -> np.ndarray:
        fixture_shapes.append((height, width))
        return real_gradient(height, width)

    monkeypatch.setattr(benchmark, "_gradient_hwc", tracked_gradient)
    fake = ModuleType("equilib")

    def equi2cube(*args: object, **kwargs: object) -> np.ndarray:
        calls["equi2cube"] += 1
        return np.zeros((1, 3, 4, 24))

    def cube2equi(cube: object, *args: object, **kwargs: object) -> np.ndarray:
        calls["cube2equi"] += 1
        assert np.asarray(cube).shape == (3, 4, 24)
        return np.zeros((3, 8, 16))

    fake.equi2cube = equi2cube  # type: ignore[attr-defined]
    fake.cube2equi = cube2equi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "equilib", fake)
    case = benchmark.Case("pyequilib", "numpy", 1, operation, "bilinear", 8, 16, 4)

    timed_operation, _ = benchmark._pyequilib_operation(case)

    assert calls == {"equi2cube": 0, "cube2equi": 0}
    assert ((8, 16) in fixture_shapes) is (operation == "erp_to_cubemap")
    timed_operation()
    expected = "equi2cube" if operation == "erp_to_cubemap" else "cube2equi"
    assert calls[expected] == 1


def test_panorai_cubemap_backprojection_only_allocates_face_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_shapes: list[tuple[int, int]] = []
    real_gradient = benchmark._gradient_hwc

    def tracked_gradient(height: int, width: int) -> np.ndarray:
        fixture_shapes.append((height, width))
        return real_gradient(height, width)

    monkeypatch.setattr(benchmark, "_gradient_hwc", tracked_gradient)
    case = benchmark.Case("panorai", "numpy", 1, "cubemap_to_erp", "bilinear", 8, 16, 4)

    benchmark._panorai_operation(case)

    assert fixture_shapes == [(4, 4)] * 6


def test_worker_starts_rss_measurement_after_synchronized_fixture_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"prepared": False, "synchronized": False, "operated": False}

    def adapter(case: benchmark.Case):
        state["prepared"] = True

        def operation() -> np.ndarray:
            state["operated"] = True
            return np.zeros((2, 4, 3))

        return operation, (2, 4, 3)

    rss_observations: list[tuple[bool, bool, bool]] = []

    def rss() -> int:
        rss_observations.append(
            (state["prepared"], state["synchronized"], state["operated"])
        )
        return 100 + 10 * len(rss_observations)

    monkeypatch.setattr(benchmark, "_panorai_operation", adapter)
    monkeypatch.setattr(
        benchmark, "_sync_device", lambda case: state.__setitem__("synchronized", True)
    )
    monkeypatch.setattr(benchmark, "_rss_bytes", rss)
    case = benchmark.Case("panorai", "numpy", 1, "erp_to_gnomonic", "nearest", 2, 4, 2)

    result = benchmark.run_worker(case, warmup=0, repetitions=1)

    assert rss_observations[0] == (True, True, False)
    assert result["rss_before_bytes"] == 110
    assert result["rss_peak_delta_bytes"] == 10
