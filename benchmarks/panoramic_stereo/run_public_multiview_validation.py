#!/usr/bin/env python3
"""Prospective public multiview validation for full-sphere dense stereo.

The tracked manifest is frozen from pose/baseline/registered-overlap evidence
before this runner observes a dense-stereo output. Dataset bytes and generated
rasters remain external to the repository and distribution.
"""

# ruff: noqa: E402 -- the source-checkout root precedes PanorAi imports

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.panoramic_stereo.run_full_sphere_stability_validation import (
    make_candidate,
    solid_angle_weights,
)
from benchmarks.panoramic_stereo.run_real_domain_validation import (
    PairData,
    json_safe,
    load_public_pair_selection,
    sha256,
)
from benchmarks.panoramic_stereo.stability_protocol import (
    build_evaluation_population,
    same_surface_observability,
    summarize_range_prediction,
)
from panorai.applications import FullSphereDenseStereo
from panorai.stereo.panoramic import (
    PanoramicStereoPose,
    RegisteredPanoramaSet,
    StereoSourceView,
)


SCHEMA = "panorai-public-multiview-full-sphere-stereo/v1"
SELECTION_SCHEMA = "panorai-public-multiview-stereo-selection/v1"
DEFAULT_MANIFEST = Path(__file__).with_name("public_multiview_manifest.json")
DATASETS = ("stanford2d3d", "matterport360")
CONFIGURATIONS = (
    "single-primary",
    "two-source",
    "two-source-consensus-2",
)
POPULATIONS = ("observable-union", "observable-intersection")
BASELINE_AUDIT_TOLERANCE_M = 5e-6


@dataclass(frozen=True, slots=True)
class PublicTriplet:
    """One fixed reference and two public source relations."""

    dataset_id: str
    triplet_id: str
    pairs: tuple[PairData, PairData]
    selection: Mapping[str, Any]


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the frozen outcome-blind triplet contract and fail closed."""

    payload = json.loads(path.read_text())
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected public multiview selection schema")
    if payload.get("frozen_before_dense_evaluation") is not True:
        raise ValueError("multiview selection must be frozen before inference")
    policy = payload.get("selection_policy")
    if not isinstance(policy, dict):
        raise TypeError("selection_policy must be an object")
    if policy.get("dense_output_used_for_selection") is not False:
        raise ValueError("triplet selection must be dense-output blind")
    if policy.get("source_count_per_reference") != 2:
        raise ValueError("the frozen protocol requires two sources per reference")
    minimum_overlap = float(policy["minimum_directional_cloud_overlap"])
    minimum_baseline = float(policy["minimum_baseline_m"])
    maximum_baseline = float(policy["maximum_baseline_m"])
    triplets = payload.get("triplets")
    if not isinstance(triplets, list) or not triplets:
        raise ValueError("selection manifest must contain triplets")
    dataset_counts: Counter[str] = Counter()
    triplet_ids: set[str] = set()
    pair_ids: set[str] = set()
    view_ids: set[str] = set()
    for index, triplet in enumerate(triplets):
        if not isinstance(triplet, dict):
            raise TypeError(f"triplet {index} must be an object")
        required = (
            "dataset_id",
            "triplet_id",
            "group",
            "spatial_group_id",
            "reference",
            "sources",
        )
        missing = [name for name in required if name not in triplet]
        if missing:
            raise ValueError(f"triplet {index} is missing: {', '.join(missing)}")
        dataset_id = str(triplet["dataset_id"])
        if dataset_id not in DATASETS:
            raise ValueError(f"unsupported manifest dataset: {dataset_id}")
        triplet_id = str(triplet["triplet_id"])
        if triplet_id in triplet_ids:
            raise ValueError(f"duplicate triplet_id: {triplet_id}")
        triplet_ids.add(triplet_id)
        group = str(triplet["group"])
        reference_id = f"{dataset_id}::{group}::{triplet['reference']}"
        if reference_id in view_ids:
            raise ValueError(f"reused view identifier: {reference_id}")
        view_ids.add(reference_id)
        sources = triplet["sources"]
        if not isinstance(sources, list) or len(sources) != 2:
            raise ValueError(f"triplet {triplet_id} must contain exactly two sources")
        overlaps: list[float] = []
        for source in sources:
            if not isinstance(source, dict):
                raise TypeError(f"triplet {triplet_id} source must be an object")
            for name in (
                "pair_id",
                "source",
                "baseline_m",
                "minimum_directional_cloud_overlap",
            ):
                if name not in source:
                    raise ValueError(f"triplet {triplet_id} source is missing {name}")
            pair_id = str(source["pair_id"])
            if pair_id in pair_ids:
                raise ValueError(f"duplicate pair_id: {pair_id}")
            pair_ids.add(pair_id)
            source_id = f"{dataset_id}::{group}::{source['source']}"
            if source_id in view_ids:
                raise ValueError(f"reused view identifier: {source_id}")
            view_ids.add(source_id)
            baseline = float(source["baseline_m"])
            overlap = float(source["minimum_directional_cloud_overlap"])
            if not minimum_baseline <= baseline <= maximum_baseline:
                raise ValueError(f"baseline outside frozen policy for {pair_id}")
            if not minimum_overlap <= overlap <= 1.0:
                raise ValueError(f"overlap outside frozen policy for {pair_id}")
            overlaps.append(overlap)
        if overlaps != sorted(overlaps, reverse=True):
            raise ValueError(f"sources for {triplet_id} must be overlap-ranked")
        dataset_counts[dataset_id] += 1
    if dict(dataset_counts) != {"stanford2d3d": 3, "matterport360": 3}:
        raise ValueError("frozen protocol requires three triplets per dataset")
    return payload


def load_public_triplet(
    selection: Mapping[str, Any],
    root: Path,
    shape_hw: tuple[int, int],
) -> PublicTriplet:
    """Load a frozen reference and its two source relations."""

    dataset_id = str(selection["dataset_id"])
    common = {
        "group": selection["group"],
        "reference": selection["reference"],
        "selection": "frozen outcome-blind VAL-108 multiview manifest",
    }
    pairs = tuple(
        load_public_pair_selection(
            dataset_id,
            root,
            shape_hw,
            {
                **common,
                "pair_id": source["pair_id"],
                "source": source["source"],
                "baseline_m": source["baseline_m"],
                "minimum_directional_cloud_overlap": source[
                    "minimum_directional_cloud_overlap"
                ],
            },
        )
        for source in selection["sources"]
    )
    first, second = pairs
    for name in (
        "reference_rgb",
        "reference_support",
        "reference_range",
    ):
        if not np.array_equal(
            getattr(first, name), getattr(second, name), equal_nan=True
        ):
            raise RuntimeError(f"triplet reference mismatch in {name}")
    for pair, source in zip(pairs, selection["sources"], strict=True):
        baseline = float(np.linalg.norm(pair.translation_source_from_reference))
        if not math.isclose(
            baseline,
            float(source["baseline_m"]),
            rel_tol=0.0,
            abs_tol=BASELINE_AUDIT_TOLERANCE_M,
        ):
            raise RuntimeError(f"baseline mismatch for {pair.pair_id}")
    return PublicTriplet(
        dataset_id=dataset_id,
        triplet_id=str(selection["triplet_id"]),
        pairs=(first, second),
        selection=dict(selection),
    )


def registered_triplet(
    triplet: PublicTriplet,
    *,
    source_count: int,
) -> RegisteredPanoramaSet:
    """Build one public registered set without changing metric pose scale."""

    pairs = triplet.pairs[:source_count]
    sources = []
    for index, pair in enumerate(pairs):
        translation = np.asarray(
            pair.translation_source_from_reference, dtype=np.float64
        )
        pose = PanoramicStereoPose(
            rotation_source_from_reference=pair.rotation_source_from_reference,
            translation_source_from_reference=translation,
            baseline=float(np.linalg.norm(translation)),
            units="m",
            scale_status="metric",
            provenance="dataset-provided metric pose",
        )
        sources.append(
            StereoSourceView(
                source_id=f"{triplet.triplet_id}:source-{index}",
                panorama=pair.source_rgb,
                validity=pair.source_support,
                pose=pose,
            )
        )
    reference = triplet.pairs[0]
    return RegisteredPanoramaSet(
        reference_id=f"{triplet.triplet_id}:reference",
        reference=reference.reference_rgb,
        reference_validity=reference.reference_support,
        sources=tuple(sources),
    )


def _array_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _label_error_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    validity: np.ndarray,
    population: np.ndarray,
    weights: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
    hypotheses: int,
) -> dict[str, float | int | None]:
    population_weight = float(weights[population].sum())
    valid = population & validity & np.isfinite(prediction) & (prediction > 0.0)
    valid_weight = float(weights[valid].sum())
    if population_weight <= 0.0:
        return {
            "population": 0,
            "predicted_valid": 0,
            "gross_gt_2_steps_fixed_population_fraction": None,
            "gross_gt_2_steps_conditional_fraction": None,
        }
    start = 1.0 / max_range_m
    step = (1.0 / min_range_m - start) / (hypotheses - 1)

    def indices(value: np.ndarray) -> np.ndarray:
        return np.clip(
            np.rint((np.reciprocal(value) - start) / step),
            0,
            hypotheses - 1,
        ).astype(np.int32)

    truth_index = indices(truth[valid])
    prediction_index = indices(prediction[valid])
    gross = np.abs(prediction_index - truth_index) > 2
    gross_weight = float(weights[valid][gross].sum())
    return {
        "population": int(population.sum()),
        "predicted_valid": int(valid.sum()),
        "gross_gt_2_steps_fixed_population_fraction": gross_weight / population_weight,
        "gross_gt_2_steps_conditional_fraction": (
            gross_weight / valid_weight if valid_weight > 0.0 else None
        ),
    }


def _populations(
    triplet: PublicTriplet,
    sampling_support: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    reference = triplet.pairs[0]
    explicit_non_sky = (
        reference.reference_support
        & np.isfinite(reference.reference_range)
        & (reference.reference_range > 0.05)
    )
    visibility = tuple(
        same_surface_observability(
            reference.reference_range,
            pair.source_range,
            source_support=pair.source_support,
            rotation_source_from_reference=pair.rotation_source_from_reference,
            translation_source_from_reference=pair.translation_source_from_reference,
        )
        for pair in triplet.pairs
    )
    visible_union = np.logical_or.reduce(visibility)
    visible_intersection = np.logical_and.reduce(visibility)
    masks: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    for name, observable in (
        ("observable-union", visible_union),
        ("observable-intersection", visible_intersection),
    ):
        population = build_evaluation_population(
            reference.reference_range,
            explicit_non_sky=explicit_non_sky,
            geometric_support=reference.reference_support & sampling_support,
            observable=observable,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            non_sky_policy=(
                "positive dataset-depth proxy; no RGB/prediction-derived sky mask"
            ),
        )
        masks[name] = population.mask
        audits[name] = population.audit
    return masks, audits, explicit_non_sky


def run_configuration(
    triplet: PublicTriplet,
    base_candidate: FullSphereDenseStereo,
    configuration: str,
    output_dir: Path,
    *,
    min_range_m: float,
    max_range_m: float,
    hypotheses: int,
) -> dict[str, Any]:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unsupported configuration: {configuration}")
    source_count = 1 if configuration == "single-primary" else 2
    consensus = 2 if configuration == "two-source-consensus-2" else 0
    candidate = FullSphereDenseStereo(
        replace(
            base_candidate.options,
            min_range_consensus_sources=consensus,
        )
    )
    started = time.perf_counter()
    result = candidate.run(registered_triplet(triplet, source_count=source_count))
    elapsed = time.perf_counter() - started
    populations, audits, explicit_non_sky = _populations(
        triplet,
        result.sampling_support,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )
    reference = triplet.pairs[0]
    weights = solid_angle_weights(reference.reference_range.shape)
    population_records: dict[str, Any] = {}
    for name, population in populations.items():
        population_records[name] = {
            "population_audit": audits[name],
            "metrics": summarize_range_prediction(
                reference.reference_range,
                result.radial_range,
                population=population,
                prediction_validity=result.validity,
                sample_weights=weights,
            ),
            "label_error": _label_error_metrics(
                reference.reference_range,
                result.radial_range,
                result.validity,
                population,
                weights,
                min_range_m=min_range_m,
                max_range_m=max_range_m,
                hypotheses=hypotheses,
            ),
        }
    arrays = {
        "estimated_radial_range_m": result.radial_range,
        "prediction_validity": result.validity,
        "confidence": result.confidence,
        "matching_confidence": result.matching_confidence,
        "consistency_confidence": result.consistency_confidence,
        "disagreement_m": result.disagreement,
        "sampling_support": result.sampling_support,
        "application_observable": result.observable,
        "max_parallax_deg": result.max_parallax_deg,
        "source_count": result.source_count,
        "source_consensus_count": result.source_consensus_count,
        "contribution_count": result.contribution_count,
        "reference_radial_range_m": reference.reference_range,
        "explicit_non_sky": explicit_non_sky,
        **{
            f"population_{name.replace('-', '_')}": value
            for name, value in populations.items()
        },
    }
    arrays_path = output_dir / f"{configuration}.npz"
    np.savez_compressed(arrays_path, **arrays)
    return {
        "configuration": configuration,
        "source_count": source_count,
        "minimum_consensus_sources": consensus,
        "elapsed_seconds": elapsed,
        "array_payload_sha256": _array_payload_sha256(arrays),
        "arrays": str(arrays_path.resolve()),
        "populations": population_records,
        "diagnostics": dict(result.diagnostics),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fixed-denominator measures by domain/configuration/population."""

    output: dict[str, Any] = {}
    for dataset_id in DATASETS:
        output[dataset_id] = {}
        for configuration in CONFIGURATIONS:
            matching = [
                record
                for record in records
                if record["dataset_id"] == dataset_id
                and record["configuration"] == configuration
            ]
            if not matching:
                continue
            output[dataset_id][configuration] = {}
            for population_name in POPULATIONS:
                entries = [item["populations"][population_name] for item in matching]
                overall = [item["metrics"]["overall"] for item in entries]
                population_weight = sum(
                    float(item["population_weight"]) for item in overall
                )
                predicted_weight = sum(
                    float(item["predicted_valid_weight"]) for item in overall
                )
                correct_weight = sum(
                    float(item["delta_1_25"]) * float(item["population_weight"])
                    for item in overall
                )
                gross_weight = sum(
                    float(
                        item["label_error"][
                            "gross_gt_2_steps_fixed_population_fraction"
                        ]
                    )
                    * float(overall_item["population_weight"])
                    for item, overall_item in zip(entries, overall, strict=True)
                )
                output[dataset_id][configuration][population_name] = {
                    "triplet_count": len(entries),
                    "micro_coverage": predicted_weight / population_weight,
                    "micro_delta_1_25": correct_weight / population_weight,
                    "micro_gross_gt_2_steps_fixed_population_fraction": (
                        gross_weight / population_weight
                    ),
                    "micro_gross_gt_2_steps_conditional_fraction": (
                        gross_weight / predicted_weight
                        if predicted_weight > 0.0
                        else None
                    ),
                    "median_triplet_coverage": float(
                        np.median([item["coverage"] for item in overall])
                    ),
                    "median_triplet_delta_1_25": float(
                        np.median([item["delta_1_25"] for item in overall])
                    ),
                }
    return output


def deterministic_evidence(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    configuration: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return replay-comparable evidence without paths, clocks or runtimes."""

    stable_records = []
    for record in records:
        stable_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"arrays", "elapsed_seconds"}
            }
        )
    return {
        "schema": SCHEMA,
        "selection": {
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": sha256(manifest_path),
        },
        "configuration": dict(configuration),
        "aggregates": aggregate_records(stable_records),
        "records": stable_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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
        "--source-aggregation",
        choices=("median", "weighted-mean", "best"),
        default="median",
    )
    args = parser.parse_args()
    if args.width != 2 * args.height:
        raise ValueError("ERP width must be twice its height")
    manifest = load_manifest(args.manifest)
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
    )
    candidate = FullSphereDenseStereo(
        replace(
            candidate.options,
            source_aggregation=args.source_aggregation,
        )
    )
    records: list[dict[str, Any]] = []
    for selection in manifest["triplets"]:
        triplet = load_public_triplet(
            selection,
            roots[str(selection["dataset_id"])],
            (args.height, args.width),
        )
        triplet_dir = args.output_dir / triplet.triplet_id
        triplet_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{triplet.dataset_id}] {triplet.triplet_id}", flush=True)
        for configuration_name in CONFIGURATIONS:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                record = run_configuration(
                    triplet,
                    candidate,
                    configuration_name,
                    triplet_dir,
                    min_range_m=args.min_range_m,
                    max_range_m=args.max_range_m,
                    hypotheses=args.hypotheses,
                )
            record.update(
                dataset_id=triplet.dataset_id,
                triplet_id=triplet.triplet_id,
            )
            records.append(record)
            overall = record["populations"]["observable-union"]["metrics"]["overall"]
            print(
                f"  {configuration_name}: coverage={overall['coverage']:.3f} "
                f"delta_1_25={overall['delta_1_25']:.3f}",
                flush=True,
            )
    configuration = {
        "erp_shape_hw": [args.height, args.width],
        "face_size": args.face_size,
        "face_fov_deg": args.face_fov_deg,
        "min_range_m": args.min_range_m,
        "max_range_m": args.max_range_m,
        "hypotheses": args.hypotheses,
        "filter_backend": args.filter_backend,
        "matching_cost": "intensity-gradient",
        "local_aggregation": "box",
        "regularizer": "sgm4",
        "solid_angle_weighted": True,
        "fixed_denominator": True,
        "sky_excluded": True,
        "source_aggregation": candidate.options.source_aggregation,
        "range_consensus_tolerance_steps": (
            candidate.options.range_consensus_tolerance_steps
        ),
    }
    evidence = deterministic_evidence(
        manifest_path=args.manifest,
        manifest=manifest,
        configuration=configuration,
        records=records,
    )
    (args.output_dir / "evidence.json").write_text(
        json.dumps(json_safe(evidence), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    summary = {
        **evidence,
        "created_at": datetime.now().astimezone().isoformat(),
        "evidence_target": "dirty-source-checkout-licensed-local-public-datasets",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "options": asdict(candidate.options),
        "runtime_seconds": sum(float(item["elapsed_seconds"]) for item in records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
