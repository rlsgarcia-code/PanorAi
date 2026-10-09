#!/usr/bin/env python3
"""Validate and summarize a complete controlled timing run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from run_controlled_timing_benchmark import (  # type: ignore[import-not-found]
        DATASETS,
        EXPECTED_VERSION,
        OVERLAP_BINS,
        STATUS_SCHEMA,
    )
    from run_resumable_population_replay import (  # type: ignore[import-not-found]
        _load_valid_result,
        _sha256,
        _slug,
    )
except ImportError:
    from benchmarks.two_view_pose_probability.run_controlled_timing_benchmark import (
        DATASETS,
        EXPECTED_VERSION,
        OVERLAP_BINS,
        STATUS_SCHEMA,
    )
    from benchmarks.two_view_pose_probability.run_resumable_population_replay import (
        _load_valid_result,
        _sha256,
        _slug,
    )


OBSERVATION_SCHEMA = "panorai-controlled-timing-observation/v1"
SUMMARY_SCHEMA = "panorai-controlled-timing-summary/v1"
TIMING_FIELDS = (
    "detection_pair",
    "detection_per_image",
    "patches_pair",
    "descriptor_pair",
    "image_ready_per_image",
    "matching",
    "pose",
    "pair_total",
)
COLORS = {
    "matterport360": "#24557a",
    "stanford2d3d": "#16827c",
    "p74_native_polar": "#d1793f",
}
DISPLAY_NAMES = {
    "matterport360": "Matterport360",
    "stanford2d3d": "Stanford2D3D",
    "p74_native_polar": "P74",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def collect_observations(
    *,
    run_dir: Path,
    selection: list[dict[str, Any]],
    expected_source_commit: str,
    repetitions: int,
) -> list[dict[str, Any]]:
    status = _read_json(run_dir / "status.json")
    expected_total = len(selection) * repetitions
    if status.get("schema") != STATUS_SCHEMA or status.get("state") != "complete":
        raise ValueError("controlled timing run is not complete")
    if status.get("completed_timed_observations") != expected_total:
        raise ValueError("controlled timing completion count is inconsistent")
    observations = []
    for repetition in range(1, repetitions + 1):
        for selected in selection:
            path = (
                run_dir
                / "results"
                / f"repetition-{repetition}"
                / f"{_slug(selected)}.json"
            )
            result = _load_valid_result(
                path,
                selected,
                expected_package_version=EXPECTED_VERSION,
                expected_source_commit=expected_source_commit,
            )
            if result is None:
                raise ValueError(f"missing or invalid controlled result: {path}")
            if result.get("resolution_hw") != [1024, 2048]:
                raise ValueError(f"unexpected controlled resolution: {path}")
            timings = result.get("timings_seconds", {})
            if any(
                not np.isfinite(float(timings.get(field, np.nan)))
                or float(timings[field]) < 0.0
                for field in TIMING_FIELDS
            ):
                raise ValueError(f"invalid timing value: {path}")
            peak_rss = float(result.get("peak_rss_mib", np.nan))
            if not np.isfinite(peak_rss) or peak_rss <= 0.0:
                raise ValueError(f"invalid peak RSS: {path}")
            observations.append(
                {
                    "schema": OBSERVATION_SCHEMA,
                    "repetition": repetition,
                    "dataset_id": selected["dataset_id"],
                    "pair_id": selected["pair_id"],
                    "independence_component_id": selected[
                        "independence_component_id"
                    ],
                    "overlap_bin": selected["overlap_bin"],
                    "registered_cloud_overlap_min": selected[
                        "registered_cloud_overlap_min"
                    ],
                    "timings_seconds": {
                        field: float(timings[field]) for field in TIMING_FIELDS
                    },
                    "peak_rss_mib": peak_rss,
                    "counts": result["counts"],
                    "system": result["system"],
                }
            )
    return observations


def _metric_summary(
    rows: list[dict[str, Any]], field: str, *, nested_timing: bool
) -> dict[str, Any]:
    def value(row: dict[str, Any]) -> float:
        if nested_timing:
            return float(row["timings_seconds"][field])
        return float(row[field])

    raw = np.asarray([value(row) for row in rows], dtype=np.float64)
    by_pair: dict[str, list[float]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(value(row))
    pair_medians = np.asarray(
        [np.median(values) for values in by_pair.values()], dtype=np.float64
    )
    return {
        "unit": "MiB" if field == "peak_rss_mib" else "seconds",
        "observation_median": float(np.median(raw)),
        "observation_p95": float(np.quantile(raw, 0.95)),
        "pair_median_median": float(np.median(pair_medians)),
        "pair_median_p95": float(np.quantile(pair_medians, 0.95)),
        "raw_observations": int(raw.size),
        "unique_pairs": int(pair_medians.size),
    }


def summarize_cells(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    for dataset in DATASETS:
        for overlap_bin in OVERLAP_BINS:
            rows = [
                row
                for row in observations
                if row["dataset_id"] == dataset
                and row["overlap_bin"] == overlap_bin
            ]
            if not rows:
                raise ValueError(f"empty controlled timing cell: {dataset}/{overlap_bin}")
            repetitions = {int(row["repetition"]) for row in rows}
            pairs = {str(row["pair_id"]) for row in rows}
            if len(rows) != 15 or len(repetitions) != 3 or len(pairs) != 5:
                raise ValueError(
                    f"controlled timing cell is incomplete: {dataset}/{overlap_bin}"
                )
            cells.append(
                {
                    "dataset_id": dataset,
                    "overlap_bin": overlap_bin,
                    "unique_pairs": len(pairs),
                    "independence_components": len(
                        {row["independence_component_id"] for row in rows}
                    ),
                    "repetitions": len(repetitions),
                    "raw_observations": len(rows),
                    "overlap_median": float(
                        np.median(
                            [row["registered_cloud_overlap_min"] for row in rows]
                        )
                    ),
                    "metrics": {
                        field: _metric_summary(rows, field, nested_timing=True)
                        for field in TIMING_FIELDS
                    }
                    | {
                        "peak_rss_mib": _metric_summary(
                            rows, "peak_rss_mib", nested_timing=False
                        )
                    },
                }
            )
    return cells


def _render(cells: list[dict[str, Any]], output: Path) -> None:
    x = np.arange(len(OVERLAP_BINS), dtype=np.float64)
    labels = ("<10", "10–25", "25–50", "50–70", "≥70")
    panels = (
        ("pair_total", "Complete pair", "seconds / pair"),
        ("detection_pair", "Spherical detection", "seconds / pair"),
        ("pose", "R,t estimation", "seconds / pair"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for axis, (field, title, ylabel) in zip(axes, panels, strict=True):
        for dataset in DATASETS:
            rows = [row for row in cells if row["dataset_id"] == dataset]
            median = np.asarray(
                [row["metrics"][field]["observation_median"] for row in rows]
            )
            p95 = np.asarray(
                [row["metrics"][field]["observation_p95"] for row in rows]
            )
            axis.plot(
                x,
                median,
                marker="o",
                linewidth=2.0,
                color=COLORS[dataset],
                label=DISPLAY_NAMES[dataset],
            )
            axis.vlines(x, median, p95, color=COLORS[dataset], alpha=0.55)
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.set_xlabel("registered-cloud overlap (%)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Controlled PanorAi 3.5.0 two-view runtime: median and P95",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("controlled timing summary output must not exist")
    selection = _read_jsonl(args.selection)
    observations = collect_observations(
        run_dir=args.run_dir,
        selection=selection,
        expected_source_commit=args.expected_source_commit,
        repetitions=args.repetitions,
    )
    cells = summarize_cells(observations)
    args.output_dir.mkdir(parents=True)
    observations_path = args.output_dir / "controlled-timing-observations.jsonl"
    _atomic_text(
        observations_path,
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in observations
        ),
    )
    figure_path = args.output_dir / "controlled-runtime-overlap-response.png"
    _render(cells, figure_path)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete controlled idle-host timing benchmark",
        "package_version": EXPECTED_VERSION,
        "expected_source_commit": args.expected_source_commit,
        "unique_pairs": len(
            {(row["dataset_id"], row["pair_id"]) for row in observations}
        ),
        "raw_observations": len(observations),
        "repetitions": args.repetitions,
        "inference_note": (
            "three repetitions of a pair are repeated engineering measurements, "
            "not independent scenes; no probability-model predictor uses runtime"
        ),
        "sources": {
            "run_status_sha256": _sha256(args.run_dir / "status.json"),
            "selection_sha256": _sha256(args.selection),
            "observations_sha256": _sha256(observations_path),
            "figure_sha256": _sha256(figure_path),
        },
        "cells": cells,
    }
    _atomic_text(
        args.output_dir / "controlled-timing-summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
