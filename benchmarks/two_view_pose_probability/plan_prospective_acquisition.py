#!/usr/bin/env python3
"""Plan E8 groups, candidate pairs, and panorama counts before collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from scipy.stats import binom

try:
    from select_release_rule import exact_one_sided_lower
except ImportError:
    from benchmarks.two_view_pose_probability.select_release_rule import (
        exact_one_sided_lower,
    )


SCHEMA = "panorai-two-view-prospective-acquisition-plan/v1"
READINESS_SCHEMA = "panorai-two-view-prospective-readiness/v1"
DEFAULT_DOMAIN_LABELS = (
    "matterport-like-indoor",
    "stanford-like-repetitive-indoor",
    "p74-like-industrial",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    fields = list(rows[0]) if rows else []
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def capped_selected_distribution(
    groups: int,
    candidates_per_group: int,
    selection_probability: float,
    *,
    cap_per_group: int = 3,
) -> np.ndarray:
    """Exact distribution for the sum of capped binomial group yields."""
    if groups <= 0 or candidates_per_group <= 0 or cap_per_group <= 0:
        raise ValueError("groups, candidates, and cap must be positive")
    if not 0.0 < selection_probability < 1.0:
        raise ValueError("selection probability must be inside (0, 1)")
    probabilities = np.asarray(
        [
            *(
                binom.pmf(index, candidates_per_group, selection_probability)
                for index in range(cap_per_group)
            ),
            binom.sf(
                cap_per_group - 1,
                candidates_per_group,
                selection_probability,
            ),
        ],
        dtype=np.float64,
    )
    distribution = np.ones(1, dtype=np.float64)
    for _ in range(groups):
        distribution = np.convolve(distribution, probabilities)
    return distribution


def _minimum_images_for_pairs(pair_count: int) -> int:
    images = 2
    while images * (images - 1) // 2 < pair_count:
        images += 1
    return images


def _plan_row(
    *,
    domains: int,
    groups_per_domain: int,
    candidates_per_group: int,
    selected_target_per_domain: int,
    selection_probability: float,
) -> dict[str, Any]:
    distribution = capped_selected_distribution(
        groups_per_domain,
        candidates_per_group,
        selection_probability,
    )
    domain_probability = float(distribution[selected_target_per_domain:].sum())
    minimum_images = _minimum_images_for_pairs(candidates_per_group)
    return {
        "domains": domains,
        "groups_per_domain": groups_per_domain,
        "total_groups": domains * groups_per_domain,
        "candidates_per_group": candidates_per_group,
        "total_candidate_pairs": domains
        * groups_per_domain
        * candidates_per_group,
        "minimum_unique_images_per_group": minimum_images,
        "minimum_unique_images_total": domains * groups_per_domain * minimum_images,
        "maximum_unique_images_total_if_pairs_are_disjoint": domains
        * groups_per_domain
        * candidates_per_group
        * 2,
        "selected_target_per_domain": selected_target_per_domain,
        "selected_target_total": domains * selected_target_per_domain,
        "selection_probability": selection_probability,
        "probability_domain_reaches_target": domain_probability,
        "probability_all_domains_reach_target": domain_probability**domains,
    }


def _minimum_candidates(
    *,
    domains: int,
    groups_per_domain: int,
    selected_target_per_domain: int,
    selection_probability: float,
    joint_probability_target: float,
    maximum_candidates_per_group: int,
) -> dict[str, Any]:
    for candidates in range(1, maximum_candidates_per_group + 1):
        row = _plan_row(
            domains=domains,
            groups_per_domain=groups_per_domain,
            candidates_per_group=candidates,
            selected_target_per_domain=selected_target_per_domain,
            selection_probability=selection_probability,
        )
        if row["probability_all_domains_reach_target"] >= joint_probability_target:
            return row
    raise RuntimeError("search range cannot satisfy the acquisition target")


def plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("prospective acquisition output directory must not exist")
    readiness = _read_json(args.readiness_report)
    if readiness.get("schema") != READINESS_SCHEMA:
        raise ValueError("candidate readiness schema is invalid")
    if readiness.get("status") != "DRAFT_REQUIRES_EXPLICIT_AUTHORIZATION_AND_E8":
        raise ValueError("candidate is not an outcome-disclosed E8 draft")
    evaluation = readiness["retrospective_evaluation_hypothesis"]
    selected = int(evaluation["selected_pairs"])
    population = int(evaluation["population_pairs"])
    if not 0 < selected < population:
        raise ValueError("retrospective selection count is invalid")
    observed_probability = selected / population
    conservative_probability = exact_one_sided_lower(selected, population)

    domains = len(args.domain_labels)
    if args.selected_target % domains:
        raise ValueError("selected target must divide evenly across domains")
    selected_per_domain = args.selected_target // domains
    if args.groups_per_domain * args.cap_per_group < selected_per_domain:
        raise ValueError("group cap makes the per-domain target impossible")
    if args.cap_per_group != 3:
        raise ValueError("the frozen E8 gate requires a cap of three")

    scenario_rows = []
    for probability_name, probability in (
        ("observed", observed_probability),
        ("one-sided-95-lower", conservative_probability),
    ):
        for groups_per_domain in args.group_grid:
            if groups_per_domain * args.cap_per_group < selected_per_domain:
                continue
            row = _minimum_candidates(
                domains=domains,
                groups_per_domain=groups_per_domain,
                selected_target_per_domain=selected_per_domain,
                selection_probability=probability,
                joint_probability_target=args.accrual_probability,
                maximum_candidates_per_group=args.maximum_candidates_per_group,
            )
            row["selection_probability_scenario"] = probability_name
            scenario_rows.append(row)

    recommended = _minimum_candidates(
        domains=domains,
        groups_per_domain=args.groups_per_domain,
        selected_target_per_domain=selected_per_domain,
        selection_probability=conservative_probability,
        joint_probability_target=args.accrual_probability,
        maximum_candidates_per_group=args.maximum_candidates_per_group,
    )
    proposed_images_per_group = max(
        args.proposed_images_per_group,
        int(recommended["minimum_unique_images_per_group"]),
    )
    possible_pairs_per_group = (
        proposed_images_per_group * (proposed_images_per_group - 1) // 2
    )
    if possible_pairs_per_group < recommended["candidates_per_group"]:
        raise AssertionError("proposed image count cannot generate planned pairs")

    report = {
        "schema": SCHEMA,
        "status": "PLANNING_ONLY_NO_COLLECTION_AUTHORIZED",
        "candidate_readiness": {
            "artifact_name": args.readiness_report.name,
            "sha256": _sha256(args.readiness_report),
            "candidate_sha256": readiness["candidate"]["sha256"],
        },
        "unit": {
            "estimator_input": "exactly two spherical panoramas",
            "independence": "capture group; images may not cross groups",
            "primary_selected_pair_cap_per_group": args.cap_per_group,
        },
        "selection_rate": {
            "selected_pairs": selected,
            "candidate_pairs": population,
            "observed": observed_probability,
            "exact_one_sided_95_lower": conservative_probability,
            "use_for_recommended_plan": "exact one-sided 95% lower",
        },
        "balanced_domain_design": {
            "labels": args.domain_labels,
            "selected_target_total": args.selected_target,
            "selected_target_per_domain": selected_per_domain,
            "note": "domain label is evaluation metadata, never a model input",
        },
        "recommended_plan": {
            **recommended,
            "joint_accrual_probability_target": args.accrual_probability,
            "proposed_unique_images_per_group": proposed_images_per_group,
            "proposed_unique_images_total": proposed_images_per_group
            * recommended["total_groups"],
            "possible_unordered_pairs_per_group": possible_pairs_per_group,
            "pair_registration_policy": (
                "register only capture-eligible pairs before prediction; reuse "
                "within a group is allowed and handled by group-level inference"
            ),
        },
        "scenario_grid": scenario_rows,
        "limitations": [
            "Accrual probability is not pose-accuracy power.",
            "The binomial model assumes a common selection rate and conditional independence within planning groups; E8 analysis remains group based.",
            "The selection-rate lower bound comes from opened retrospective evaluation and is used only for resource planning.",
            "No reference pose may be inspected to decide whether collection continues.",
        ],
    }
    args.output_dir.mkdir(parents=True)
    _atomic_csv(args.output_dir / "acquisition-scenarios.csv", scenario_rows)
    report["scenario_csv_sha256"] = _sha256(
        args.output_dir / "acquisition-scenarios.csv"
    )
    _atomic_json(args.output_dir / "acquisition-plan.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-target", type=int, default=120)
    parser.add_argument("--groups-per-domain", type=int, default=20)
    parser.add_argument("--cap-per-group", type=int, default=3)
    parser.add_argument("--accrual-probability", type=float, default=0.90)
    parser.add_argument("--maximum-candidates-per-group", type=int, default=200)
    parser.add_argument("--proposed-images-per-group", type=int, default=16)
    parser.add_argument(
        "--group-grid", type=int, nargs="+", default=[14, 20, 24, 25, 30, 40]
    )
    parser.add_argument(
        "--domain-labels", nargs="+", default=list(DEFAULT_DOMAIN_LABELS)
    )
    return parser


def main() -> int:
    report = plan(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
