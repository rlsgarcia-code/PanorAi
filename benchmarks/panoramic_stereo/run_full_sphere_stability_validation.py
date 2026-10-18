#!/usr/bin/env python3
"""Evaluate full-sphere stereo on licensed public data.

The datasets and generated rasters remain external to the repository. Dataset
depth and provided metric pose define the evaluation population; neither is
available to the RGB matcher.
"""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_real_domain_validation import (
    PairData,
    json_safe,
    load_public_pair,
)
from benchmarks.panoramic_stereo.stability_protocol import (
    build_evaluation_population,
    same_surface_observability,
    summarize_confidence_selectivity,
    summarize_range_prediction,
)
from panorai.applications import FullSphereDenseStereo
from panorai.stereo import SphericalStereoOptions
from panorai.stereo.panoramic import (
    FullSphereStereoOptions,
    OverlappingCubemapPlanner,
    PanoramicStereoPose,
    RegisteredPanoramaSet,
    StereoSourceView,
)


SCHEMA = "panorai-public-full-sphere-stereo-stability/v1"
PUBLIC_DATASETS = ("stanford2d3d", "matterport360")
DATASETS = PUBLIC_DATASETS


def solid_angle_weights(shape_hw: tuple[int, int]) -> np.ndarray:
    """Return ERP pixel weights proportional to represented solid angle."""

    height, width = shape_hw
    latitude = (
        math.pi / 2.0 - math.pi * (np.arange(height, dtype=np.float64) + 0.5) / height
    )
    return np.broadcast_to(np.cos(latitude)[:, None], (height, width)).copy()


def make_candidate(
    *,
    face_size: int,
    face_fov_deg: float,
    min_range_m: float,
    max_range_m: float,
    hypotheses: int,
    filter_backend: str,
    matching_cost: str = "intensity-gradient",
    local_aggregation: str = "box",
    regularizer: str = "sgm4",
    erp_feature_tap_deg: float | None = None,
    face_cost_tap_deg: float | None = None,
) -> FullSphereDenseStereo:
    stereo = SphericalStereoOptions(
        min_range=min_range_m,
        max_range=max_range_m,
        num_hypotheses=hypotheses,
        pole_margin_fraction=0.0,
        bidirectional_consistency=False,
        pyramid_levels=1,
        filter_backend=filter_backend,
    )
    return FullSphereDenseStereo(
        FullSphereStereoOptions(
            planner=OverlappingCubemapPlanner(
                face_size=face_size,
                face_fov_deg=face_fov_deg,
            ),
            stereo=stereo,
            min_parallax_deg=0.25,
            min_source_evidence=1,
            source_aggregation="median",
            matching_cost=matching_cost,
            local_aggregation=local_aggregation,
            regularizer=regularizer,
            erp_feature_tap_deg=erp_feature_tap_deg,
            face_cost_tap_deg=face_cost_tap_deg,
            min_range_consensus_sources=0,
        )
    )


def registered_pair(pair: PairData) -> RegisteredPanoramaSet:
    translation = np.asarray(pair.translation_source_from_reference, dtype=np.float64)
    pose = PanoramicStereoPose(
        rotation_source_from_reference=pair.rotation_source_from_reference,
        translation_source_from_reference=translation,
        baseline=float(np.linalg.norm(translation)),
        units="m",
        scale_status="metric",
        provenance="dataset-provided metric pose",
    )
    return RegisteredPanoramaSet(
        reference_id=f"{pair.dataset_id}:reference",
        reference=pair.reference_rgb,
        reference_validity=pair.reference_support,
        sources=(
            StereoSourceView(
                source_id=f"{pair.dataset_id}:source",
                panorama=pair.source_rgb,
                validity=pair.source_support,
                pose=pose,
            ),
        ),
    )


def evaluate_result(
    pair: PairData,
    result: Any,
    *,
    min_range_m: float,
    max_range_m: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    explicit_non_sky = (
        pair.reference_support
        & np.isfinite(pair.reference_range)
        & (pair.reference_range > 0.05)
    )
    same_surface = same_surface_observability(
        pair.reference_range,
        pair.source_range,
        source_support=pair.source_support,
        rotation_source_from_reference=pair.rotation_source_from_reference,
        translation_source_from_reference=pair.translation_source_from_reference,
    )
    population = build_evaluation_population(
        pair.reference_range,
        explicit_non_sky=explicit_non_sky,
        geometric_support=(
            pair.reference_support & np.asarray(result.sampling_support, dtype=bool)
        ),
        observable=same_surface,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        non_sky_policy=(
            "positive dataset-depth proxy; no RGB/prediction-derived sky mask"
        ),
    )
    metrics = summarize_range_prediction(
        pair.reference_range,
        result.radial_range,
        population=population.mask,
        prediction_validity=result.validity,
        sample_weights=solid_angle_weights(pair.reference_range.shape),
    )
    return (
        {"population_audit": population.audit, "metrics": metrics},
        explicit_non_sky,
        population.mask,
    )


def render_panel(
    path: Path,
    pair: PairData,
    result: Any,
    explicit_non_sky: np.ndarray,
    population: np.ndarray,
    record: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_truth = explicit_non_sky & np.isfinite(pair.reference_range)
    lower, upper = np.percentile(pair.reference_range[valid_truth], (2.0, 98.0))
    error = np.full(pair.reference_range.shape, np.nan, dtype=np.float32)
    evaluated = population & result.validity
    error[evaluated] = np.abs(
        result.radial_range[evaluated] - pair.reference_range[evaluated]
    )
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    panels = (
        (pair.reference_rgb, "reference ERP", None),
        (pair.source_rgb, "source ERP", None),
        (
            np.ma.masked_where(~valid_truth, pair.reference_range),
            "dataset radial range [m]",
            "turbo",
        ),
        (
            np.ma.masked_where(~result.validity, result.radial_range),
            "estimated radial range [m]",
            "turbo",
        ),
        (explicit_non_sky, "explicit non-sky proxy", "gray"),
        (population, "fixed evaluation population", "gray"),
        (result.validity, "prediction validity", "gray"),
        (np.ma.masked_invalid(error), "absolute error [m]", "magma"),
    )
    for axis, (value, title, cmap) in zip(axes.flat, panels, strict=True):
        kwargs: dict[str, Any] = {"cmap": cmap} if cmap is not None else {}
        if "radial range" in title:
            kwargs.update(vmin=lower, vmax=upper)
        axis.imshow(value, **kwargs)
        axis.set_title(title)
        axis.axis("off")
    overall = record["metrics"]["overall"]
    figure.suptitle(
        f"{pair.dataset_id} · FullSphereDenseStereo · "
        f"coverage={overall['coverage']:.3f} · "
        f"δ<1.25={overall['delta_1_25']:.3f}"
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_dataset(
    pair: PairData,
    candidate: FullSphereDenseStereo,
    output_dir: Path,
    *,
    min_range_m: float,
    max_range_m: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = candidate.run(registered_pair(pair))
    elapsed = time.perf_counter() - started
    evaluation, explicit_non_sky, population = evaluate_result(
        pair,
        result,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )
    confidence_selectivity = summarize_confidence_selectivity(
        pair.reference_range,
        result.radial_range,
        result.confidence,
        population=population,
        prediction_validity=result.validity,
        sample_weights=solid_angle_weights(pair.reference_range.shape),
    )
    panel_path = output_dir / f"{pair.dataset_id}-audit.png"
    arrays_path = output_dir / f"{pair.dataset_id}-result.npz"
    record = {
        "dataset_id": pair.dataset_id,
        "pair_id": pair.pair_id,
        "status": "completed",
        "elapsed_seconds": elapsed,
        "application_stability": candidate.stability,
        "baseline_m": float(np.linalg.norm(pair.translation_source_from_reference)),
        "population_audit": evaluation["population_audit"],
        "metrics": evaluation["metrics"],
        "confidence_selectivity": confidence_selectivity,
        "diagnostics": dict(result.diagnostics),
        "panel": str(panel_path.resolve()),
        "arrays": str(arrays_path.resolve()),
    }
    np.savez_compressed(
        arrays_path,
        estimated_radial_range_m=result.radial_range,
        prediction_validity=result.validity,
        confidence=result.confidence,
        matching_confidence=result.matching_confidence,
        consistency_confidence=result.consistency_confidence,
        disagreement_m=result.disagreement,
        sampling_support=result.sampling_support,
        application_observable=result.observable,
        max_parallax_deg=result.max_parallax_deg,
        source_count=result.source_count,
        source_consensus_count=result.source_consensus_count,
        contribution_count=result.contribution_count,
        reference_radial_range_m=pair.reference_range,
        explicit_non_sky=explicit_non_sky,
        evaluation_population=population,
    )
    render_panel(
        panel_path,
        pair,
        result,
        explicit_non_sky,
        population,
        record,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stanford-root",
        type=Path,
        default=Path("data/stanford2d3d"),
    )
    parser.add_argument(
        "--matterport-root",
        type=Path,
        default=Path("data/matterport360"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(PUBLIC_DATASETS)
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--face-size", type=int, default=128)
    parser.add_argument("--face-fov-deg", type=float, default=100.0)
    parser.add_argument("--min-range-m", type=float, default=0.3)
    parser.add_argument("--max-range-m", type=float, default=12.0)
    parser.add_argument("--hypotheses", type=int, default=48)
    parser.add_argument(
        "--filter-backend", choices=("auto", "numpy", "native"), default="auto"
    )
    parser.add_argument(
        "--matching-cost",
        choices=("intensity-gradient", "ad-census"),
        default="intensity-gradient",
    )
    parser.add_argument("--local-aggregation", choices=("none", "box"), default="box")
    parser.add_argument(
        "--regularizer", choices=("none", "sgm4", "tmgm"), default="sgm4"
    )
    args = parser.parse_args()
    if args.width != 2 * args.height:
        raise ValueError("ERP width must be twice its height")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "stanford2d3d": args.stanford_root,
        "matterport360": args.matterport_root,
    }
    candidate = make_candidate(
        face_size=args.face_size,
        face_fov_deg=args.face_fov_deg,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        hypotheses=args.hypotheses,
        filter_backend=args.filter_backend,
        matching_cost=args.matching_cost,
        local_aggregation=args.local_aggregation,
        regularizer=args.regularizer,
    )
    records: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for dataset_id in args.datasets:
        print(f"[{dataset_id}] loading pair", flush=True)
        pair = load_public_pair(
            dataset_id, roots[dataset_id], (args.height, args.width)
        )
        provenance[dataset_id] = pair.provenance
        record = run_dataset(
            pair,
            candidate,
            args.output_dir,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
        )
        records.append(record)
        print(json.dumps(json_safe(record), indent=2), flush=True)
    summary = {
        "schema": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence_target": "dirty-source-checkout-licensed-local-public-datasets",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "application": candidate.describe(),
        "configuration": {
            "erp_shape_hw": [args.height, args.width],
            "face_size": args.face_size,
            "face_fov_deg": args.face_fov_deg,
            "min_range_m": args.min_range_m,
            "max_range_m": args.max_range_m,
            "hypotheses": args.hypotheses,
            "filter_backend": args.filter_backend,
            "matching_cost": args.matching_cost,
            "local_aggregation": args.local_aggregation,
            "regularizer": args.regularizer,
            "fixed_denominator": True,
            "solid_angle_weighted": True,
            "same_surface_tolerance": "max(0.10 m, 5%)",
        },
        "options": asdict(candidate.options),
        "provenance": provenance,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
