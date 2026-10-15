from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "composed_spherical_api" / "benchmark.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "composed_pipeline_benchmark", BENCHMARK
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_retains_raw_samples_and_nearest_statistics() -> None:
    summary = _module()._summarize(
        [1.0, 2.0, 10.0],
        rss_before=100,
        rss_after=140,
    )
    assert summary["samples_ms"] == [1.0, 2.0, 10.0]
    assert summary["median_ms"] == 2.0
    assert summary["p95_ms"] > summary["median_ms"]
    assert summary["peak_rss_growth_bytes"] == 40


def test_benchmark_requires_artifact_and_installed_origin_evidence() -> None:
    source = BENCHMARK.read_text(encoding="utf-8")
    compile(source, str(BENCHMARK), "exec")
    for required in (
        "--wheel",
        "--source-commit",
        "package_origin",
        "wheel_sha256",
        "samples_ms",
        "median_ms",
        "p95_ms",
        "peak_rss_bytes_after",
        "cold_construct_and_run",
        "warm_pipeline_run",
        "cache_miss",
        "cache_hit",
    ):
        assert required in source
