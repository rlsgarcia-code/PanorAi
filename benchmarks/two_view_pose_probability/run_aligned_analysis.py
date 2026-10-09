#!/usr/bin/env python3
"""Run the sealed aligned-frontend VAL-018 analysis in its frozen order."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

if __package__:
    from benchmarks.two_view_pose_probability.build_aligned_population_table import (
        run as build_aligned_table,
    )
    from benchmarks.two_view_pose_probability.plan_prospective_confirmation import (
        run as plan_prospective_confirmation,
    )
    from benchmarks.two_view_pose_probability.render_paper_results import (
        run as render_paper_results,
    )
    from benchmarks.two_view_pose_probability.run_probability_models import (
        evaluate as evaluate_models,
        fit_predict,
        fit_predict_lodo,
    )
    from benchmarks.two_view_pose_probability.select_release_rule import (
        ALIGNED_POST_MODEL,
        NoQualifyingRuleError,
        evaluate_rule,
        freeze_rule,
    )
else:  # Direct execution from the repository root.
    from build_aligned_population_table import run as build_aligned_table
    from plan_prospective_confirmation import run as plan_prospective_confirmation
    from render_paper_results import run as render_paper_results
    from run_probability_models import evaluate as evaluate_models
    from run_probability_models import fit_predict, fit_predict_lodo
    from select_release_rule import (
        ALIGNED_POST_MODEL,
        NoQualifyingRuleError,
        evaluate_rule,
        freeze_rule,
    )


SCHEMA = "panorai-two-view-pose-aligned-analysis-run/v1"
ANALYSIS_STAGE_ORDER = (
    "aligned-table",
    "fit-models",
    "fit-lodo-models",
    "evaluate-models",
    "evaluate-lodo-models",
    "freeze-selective-rule",
    "evaluate-selective-rule",
    "plan-prospective-confirmation",
    "render-paper-results",
)


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


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _run_stage(
    output_dir: Path,
    status: dict[str, Any],
    name: str,
    operation: Callable[[], Any],
) -> Any:
    if name not in ANALYSIS_STAGE_ORDER:
        raise ValueError(f"unknown analysis stage: {name}")
    status["state"] = "running"
    status["current_stage"] = name
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_dir / "status.json", status)
    log_path = output_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as stream:
            with redirect_stdout(stream):
                result = operation()
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = {"type": type(error).__name__, "message": str(error)}
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(output_dir / "status.json", status)
        raise
    status["completed_stages"].append(name)
    status["current_stage"] = None
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_dir / "status.json", status)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _prepare_output_dir(args.output_dir)
    base_analysis_table = args.base_analysis_table.resolve()
    results_dir = args.results_dir.resolve()
    status: dict[str, Any] = {
        "schema": SCHEMA,
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "current_stage": None,
        "completed_stages": [],
        "model_profile": "aligned",
        "post_precision_model": ALIGNED_POST_MODEL,
        "expected_package_version": args.expected_package_version,
        "expected_source_commit": args.expected_source_commit,
    }
    _atomic_json(output_dir / "status.json", status)

    table_dir = output_dir / "aligned-table"
    _run_stage(
        output_dir,
        status,
        "aligned-table",
        lambda: build_aligned_table(
            argparse.Namespace(
                base_analysis_table=base_analysis_table,
                results_dir=results_dir,
                output_dir=table_dir,
                expected_package_version=args.expected_package_version,
                expected_source_commit=args.expected_source_commit,
            )
        ),
    )
    features = table_dir / "features.jsonl"
    training_outcomes = table_dir / "outcomes-development-calibration.jsonl"
    evaluation_outcomes = table_dir / "outcomes-evaluation.jsonl"

    models_dir = output_dir / "models"
    _run_stage(
        output_dir,
        status,
        "fit-models",
        lambda: fit_predict(
            argparse.Namespace(
                features=features,
                training_outcomes=training_outcomes,
                output_dir=models_dir,
                model_profile="aligned",
            )
        ),
    )
    lodo_models_dir = output_dir / "models-lodo"
    _run_stage(
        output_dir,
        status,
        "fit-lodo-models",
        lambda: fit_predict_lodo(
            argparse.Namespace(
                features=features,
                training_outcomes=training_outcomes,
                output_dir=lodo_models_dir,
                model_profile="aligned",
            )
        ),
    )

    evaluation_dir = output_dir / "evaluation"
    _run_stage(
        output_dir,
        status,
        "evaluate-models",
        lambda: evaluate_models(
            argparse.Namespace(
                predictions=models_dir / "predictions.jsonl",
                model_card=models_dir / "model-card.json",
                evaluation_outcomes=evaluation_outcomes,
                output_dir=evaluation_dir,
            )
        ),
    )
    lodo_evaluation_dir = output_dir / "evaluation-lodo"
    _run_stage(
        output_dir,
        status,
        "evaluate-lodo-models",
        lambda: evaluate_models(
            argparse.Namespace(
                predictions=lodo_models_dir / "predictions.jsonl",
                model_card=lodo_models_dir / "model-card.json",
                evaluation_outcomes=evaluation_outcomes,
                output_dir=lodo_evaluation_dir,
            )
        ),
    )

    release_dir = output_dir / "release-rule"
    release_rule = release_dir / "release-rule.json"
    rule_qualified = True
    try:
        _run_stage(
            output_dir,
            status,
            "freeze-selective-rule",
            lambda: freeze_rule(
                argparse.Namespace(
                    features=features,
                    predictions=models_dir / "predictions.jsonl",
                    training_outcomes=training_outcomes,
                    post_model=ALIGNED_POST_MODEL,
                    output=release_rule,
                )
            ),
        )
    except NoQualifyingRuleError:
        rule_qualified = False
        status.pop("error", None)
        status["state"] = "running"
        status["current_stage"] = None
        status["completed_stages"].append("freeze-selective-rule")
        status["scientific_result"] = "no calibration-grid rule qualified"
        _atomic_json(output_dir / "status.json", status)

    release_evaluation: Path | None = None
    prospective_plan: Path | None = None
    if rule_qualified:
        _run_stage(
            output_dir,
            status,
            "evaluate-selective-rule",
            lambda: evaluate_rule(
                argparse.Namespace(
                    rule=release_rule,
                    features=features,
                    predictions=models_dir / "predictions.jsonl",
                    evaluation_outcomes=evaluation_outcomes,
                    output_dir=release_dir,
                )
            ),
        )
        release_evaluation = release_dir / "release-rule-evaluation.json"
        prospective_dir = output_dir / "prospective-plan"
        _run_stage(
            output_dir,
            status,
            "plan-prospective-confirmation",
            lambda: plan_prospective_confirmation(
                argparse.Namespace(
                    release_evaluation=release_evaluation,
                    output_dir=prospective_dir,
                )
            ),
        )
        prospective_plan = prospective_dir / "prospective-power-plan.json"

    paper_dir = output_dir / "paper-results"
    _run_stage(
        output_dir,
        status,
        "render-paper-results",
        lambda: render_paper_results(
            argparse.Namespace(
                analysis_table=table_dir / "analysis-table.jsonl",
                evaluation=evaluation_dir / "evaluation.json",
                lodo_evaluation=lodo_evaluation_dir / "evaluation.json",
                release_rule=release_rule if rule_qualified else None,
                release_evaluation=release_evaluation,
                output_dir=paper_dir,
            )
        ),
    )

    artifacts = {
        "aligned_table_manifest": _artifact(table_dir / "manifest.json"),
        "model_card": _artifact(models_dir / "model-card.json"),
        "predictions": _artifact(models_dir / "predictions.jsonl"),
        "evaluation": _artifact(evaluation_dir / "evaluation.json"),
        "lodo_model_card": _artifact(lodo_models_dir / "model-card.json"),
        "lodo_predictions": _artifact(lodo_models_dir / "predictions.jsonl"),
        "lodo_evaluation": _artifact(lodo_evaluation_dir / "evaluation.json"),
        "paper_results": _artifact(paper_dir / "paper-results.json"),
    }
    if rule_qualified:
        artifacts["release_rule"] = _artifact(release_rule)
        artifacts["release_evaluation"] = _artifact(release_evaluation)
        artifacts["prospective_plan"] = _artifact(prospective_plan)
    else:
        artifacts["release_rule_search"] = _artifact(
            release_dir / "release-rule-search.json"
        )

    manifest = {
        "schema": SCHEMA,
        "status": "complete retrospective aligned analysis",
        "interpretation": (
            "Retrospective evidence only; prospective E8 confirmation remains "
            "mandatory before any release-reliability claim."
        ),
        "model_profile": "aligned",
        "post_precision_model": ALIGNED_POST_MODEL,
        "rule_qualified_on_calibration": rule_qualified,
        "expected_package_version": args.expected_package_version,
        "expected_source_commit": args.expected_source_commit,
        "inputs": {
            "base_analysis_table": _artifact(base_analysis_table),
            "results_directory": str(results_dir),
        },
        "completed_stages": status["completed_stages"],
        "artifacts": artifacts,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    status["state"] = "complete"
    status["current_stage"] = None
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    status["manifest"] = _artifact(output_dir / "manifest.json")
    _atomic_json(output_dir / "status.json", status)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-analysis-table", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-package-version", default="3.5.0")
    parser.add_argument("--expected-source-commit", required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
