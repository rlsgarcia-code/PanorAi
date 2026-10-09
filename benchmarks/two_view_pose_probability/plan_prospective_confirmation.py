#!/usr/bin/env python3
"""Plan E8 independent-group confirmation for the frozen selective rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.stats import beta

SCHEMA = "panorai-two-view-prospective-power-plan/v1"
SEED = 18018
REPETITIONS = 30_000
PAIRS_PER_GROUP = 3
GROUP_GRID = tuple(range(5, 151, 5))
TRUE_PRECISION_SCENARIOS = (0.95, 0.97, 0.98, 0.99)
ICC_SCENARIOS = (0.05, 0.10, 0.20)
MARGINAL_CATASTROPHIC_RATE = 0.001
MINIMUM_OBSERVED_PRECISION = 0.95
MINIMUM_EXACT_LOWER = 0.90
TARGET_POWER = 0.80
DESIGN_TRUE_PRECISION = 0.97
DESIGN_ICC = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _scenario_seed(mean_precision: float, icc: float, groups: int) -> int:
    text = f"{SEED}|{mean_precision:.4f}|{icc:.4f}|{groups}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def simulate_power(
    *,
    mean_precision: float,
    icc: float,
    groups: int,
    repetitions: int = REPETITIONS,
) -> float:
    """Beta-binomial group simulation with catastrophic errors among failures."""
    if not 0.0 < mean_precision < 1.0 or not 0.0 < icc < 1.0:
        raise ValueError("mean_precision and icc must be strictly between zero and one")
    concentration = 1.0 / icc - 1.0
    alpha = mean_precision * concentration
    beta_shape = (1.0 - mean_precision) * concentration
    generator = np.random.default_rng(_scenario_seed(mean_precision, icc, groups))
    group_probabilities = generator.beta(alpha, beta_shape, size=(repetitions, groups))
    successes = generator.binomial(PAIRS_PER_GROUP, group_probabilities).sum(axis=1)
    count = groups * PAIRS_PER_GROUP
    failures = count - successes
    conditional_catastrophic = min(
        MARGINAL_CATASTROPHIC_RATE / (1.0 - mean_precision), 1.0
    )
    catastrophic = generator.binomial(failures, conditional_catastrophic)
    lower = beta.ppf(0.05, successes, count - successes + 1)
    passes = (
        (successes / count >= MINIMUM_OBSERVED_PRECISION)
        & (lower >= MINIMUM_EXACT_LOWER)
        & (catastrophic == 0)
    )
    return float(np.mean(passes))


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_path = args.release_evaluation.resolve()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("verdict") != "NO_GO":
        raise ValueError("power plan expects the current retrospective NO_GO evidence")
    rows = []
    for mean_precision in TRUE_PRECISION_SCENARIOS:
        for icc in ICC_SCENARIOS:
            reached = None
            for groups in GROUP_GRID:
                power = simulate_power(
                    mean_precision=mean_precision, icc=icc, groups=groups
                )
                rows.append(
                    {
                        "true_precision": mean_precision,
                        "intraclass_correlation": icc,
                        "independent_groups": groups,
                        "selected_pairs": groups * PAIRS_PER_GROUP,
                        "estimated_power": power,
                    }
                )
                if reached is None and power >= TARGET_POWER:
                    reached = groups
                    break
    design = next(
        row
        for row in rows
        if row["true_precision"] == DESIGN_TRUE_PRECISION
        and row["intraclass_correlation"] == DESIGN_ICC
        and row["estimated_power"] >= TARGET_POWER
    )
    target_selected = int(design["selected_pairs"])
    screening = {}
    for dataset, metrics in evaluation["datasets"].items():
        coverage = float(metrics["pair_coverage"])
        screening[dataset] = {
            "retrospective_pair_coverage": coverage,
            "candidate_pairs_for_target_at_observed_coverage": (
                math.ceil(target_selected / coverage) if coverage > 0.0 else None
            ),
            "warning": "planning diagnostic only; prospective coverage may differ",
        }
    result = {
        "schema": SCHEMA,
        "status": "frozen planning assumptions; data collection not started",
        "confirmation_gate": {
            "minimum_observed_precision": MINIMUM_OBSERVED_PRECISION,
            "minimum_exact_one_sided_95_lower": MINIMUM_EXACT_LOWER,
            "maximum_catastrophic_accepted": 0,
            "target_power": TARGET_POWER,
        },
        "simulation": {
            "model": "beta-binomial group precision; catastrophic outcomes sampled among imprecise poses",
            "seed": SEED,
            "repetitions_per_cell": REPETITIONS,
            "selected_pairs_per_independent_group": PAIRS_PER_GROUP,
            "marginal_catastrophic_rate": MARGINAL_CATASTROPHIC_RATE,
            "true_precision_scenarios": list(TRUE_PRECISION_SCENARIOS),
            "intraclass_correlation_scenarios": list(ICC_SCENARIOS),
        },
        "primary_design_scenario": {
            "true_precision": DESIGN_TRUE_PRECISION,
            "intraclass_correlation": DESIGN_ICC,
            "minimum_independent_groups": int(design["independent_groups"]),
            "target_selected_pairs": target_selected,
            "estimated_power": float(design["estimated_power"]),
            "maximum_selected_pairs_per_group": PAIRS_PER_GROUP,
        },
        "screening_burden": screening,
        "source_release_evaluation": {
            "path": str(evaluation_path),
            "sha256": _sha256(evaluation_path),
        },
        "limitations": [
            "The beta-binomial is a planning model, not proof of future performance.",
            "The assumed 97% true precision and 0.20 ICC must be reported with the result.",
            "Accrual is by selected pairs; candidate-pair count is not fixed because gate coverage is uncertain.",
            "All pairs must be predicted before reference poses are opened.",
            "No image may occur in more than one prospective split or independent group.",
        ],
        "power_grid": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "prospective-power-plan.json", result)
    csv_path = args.output_dir / "prospective-power-grid.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
