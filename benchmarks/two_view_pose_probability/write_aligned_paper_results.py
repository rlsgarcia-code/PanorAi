#!/usr/bin/env python3
"""Write a paper-ready VAL-018 result document from sealed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


DATASET_ORDER = ("matterport360", "stanford2d3d", "p74_native_polar")
DATASET_LABELS = {
    "matterport360": "Matterport360",
    "stanford2d3d": "Stanford2D3D",
    "p74_native_polar": "P74",
}
CAPTURE_ACCEPT_MODEL = "capture-accept-overlap-baseline"
CAPTURE_PRECISE_MODEL = "capture-precise-given-accept-overlap-baseline"
CAPTURE_USABLE_MODEL = "capture-usable-overlap-baseline"
RAW_POST_MODEL = "post-precise-raw-score"
ALIGNED_POST_MODEL = "post-precise-aligned-orientation"
HISTORICAL_POST_MODEL = "post-precise-common"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = list(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _post_model(evaluation: dict[str, Any]) -> str:
    models = evaluation["models"]
    if ALIGNED_POST_MODEL in models:
        return ALIGNED_POST_MODEL
    if HISTORICAL_POST_MODEL in models:
        return HISTORICAL_POST_MODEL
    raise ValueError("evaluation has no supported post-processing model")


def _outcome_counts(rows: list[dict[str, Any]]) -> list[list[Any]]:
    result = []
    for dataset in DATASET_ORDER:
        selected = [row for row in rows if row["dataset_id"] == dataset]
        returned = sum(bool(row["outcomes"]["returned"]) for row in selected)
        accepted = sum(bool(row["outcomes"]["accepted"]) for row in selected)
        precise = sum(bool(row["outcomes"]["precise"]) for row in selected)
        usable = sum(bool(row["outcomes"]["usable"]) for row in selected)
        catastrophic = sum(
            bool(row["outcomes"]["catastrophic_accepted"]) for row in selected
        )
        result.append(
            [
                DATASET_LABELS[dataset],
                _fmt(len(selected)),
                f"{returned}/{len(selected)} ({returned / len(selected):.1%})",
                f"{accepted}/{len(selected)} ({accepted / len(selected):.1%})",
                f"{precise}/{len(selected)} ({precise / len(selected):.1%})",
                f"{usable}/{len(selected)} ({usable / len(selected):.1%})",
                _fmt(catastrophic),
            ]
        )
    return result


def _evaluation_rows(
    evaluation: dict[str, Any], model_ids: list[tuple[str, str]]
) -> list[list[Any]]:
    result = []
    for model_id, label in model_ids:
        source = evaluation["models"].get(model_id)
        if source is None:
            continue
        for dataset in DATASET_ORDER:
            metrics = source["datasets"].get(dataset)
            if metrics is None:
                continue
            result.append(
                [
                    label,
                    DATASET_LABELS[dataset],
                    _fmt(metrics["count"]),
                    _fmt(metrics["brier"]),
                    _fmt(metrics["log_loss"]),
                    _fmt(metrics["ece_10"]),
                    _fmt(metrics["predicted_mean"]),
                    _fmt(metrics["prevalence"]),
                    _fmt(metrics["independence_components"]),
                ]
            )
    return result


def _lodo_rows(evaluation: dict[str, Any], post_model: str) -> list[list[Any]]:
    result = []
    for base_model, label in (
        (CAPTURE_ACCEPT_MODEL, "Capture acceptance"),
        (post_model, "Post precision"),
    ):
        for dataset in DATASET_ORDER:
            model_id = f"{base_model}--lodo-{dataset}"
            metrics = evaluation["models"][model_id]["datasets"][dataset]
            result.append(
                [
                    label,
                    DATASET_LABELS[dataset],
                    _fmt(metrics["count"]),
                    _fmt(metrics["brier"]),
                    _fmt(metrics["log_loss"]),
                    _fmt(metrics["ece_10"]),
                    _fmt(metrics["independence_components"]),
                ]
            )
    return result


def _overlap_rows(paper: dict[str, Any]) -> list[list[Any]]:
    result = []
    for row in paper["overlap_response"]:
        result.append(
            [
                DATASET_LABELS[row["dataset_id"]],
                row["overlap_bin"],
                _fmt(row["pairs"]),
                _fmt(row["independence_components"]),
                _fmt(row["returned_rate"]),
                _fmt(row["accepted_rate"]),
                _fmt(row["usable_rate"]),
                _fmt(row["catastrophic_accepted"]),
            ]
        )
    return result


def _runtime_rows(paper: dict[str, Any]) -> list[list[Any]]:
    result = []
    for dataset in DATASET_ORDER:
        engineering = paper["engineering_summary"]["datasets"][dataset]
        total = engineering["timing_seconds"]["pair_total_seconds"]
        detection = engineering["timing_seconds"]["detection_pair_seconds"]
        memory = engineering["peak_rss_mib"]
        result.append(
            [
                DATASET_LABELS[dataset],
                _fmt(engineering["pairs"]),
                _fmt(detection["median"], 2),
                _fmt(detection["p95"], 2),
                _fmt(total["median"], 2),
                _fmt(total["p95"], 2),
                _fmt(memory["p95"], 1),
                _fmt(memory["maximum"], 1),
            ]
        )
    return result


def _surface_rows(paper: dict[str, Any]) -> list[list[Any]]:
    surface = paper.get("capture_probability_surface") or []
    return [
        [
            row["overlap_bin"],
            f"{row['baseline_low_m']:.2f}–{row['baseline_high_m']:.2f}",
            _fmt(row["support_components"]),
            _fmt(row["p_accept"]),
            _fmt(row["p_precise_given_accept"]),
            _fmt(row["p_usable"]),
        ]
        for row in surface
        if row["supported"]
    ]


def _release_section(
    rule: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    search: dict[str, Any] | None,
) -> tuple[str, str]:
    if rule is None:
        if search is None:
            raise ValueError("release rule and release-rule search are both absent")
        return (
            "NO_QUALIFYING_CALIBRATION_RULE",
            "No threshold pair in the frozen calibration grid satisfied all "
            "precision, exact-bound, catastrophic-error, and group-support "
            f"requirements. {len(search['candidate_grid'])} candidates were "
            "retained in the rule-search artifact; evaluation outcomes were not "
            "used to rescue a failed calibration rule.",
        )
    if evaluation is None:
        raise ValueError("release evaluation is required when a rule exists")
    threshold = rule["thresholds"]
    envelope = rule["capture_envelope"]
    rows = []
    for dataset in DATASET_ORDER:
        metrics = evaluation["datasets"][dataset]
        rows.append(
            [
                DATASET_LABELS[dataset],
                f"{metrics['selected_pairs']}/{metrics['eligible_pairs']}",
                _fmt(metrics["selected_components"]),
                _fmt(metrics["selected_precision"]),
                _fmt(metrics["exact_one_sided_95_lower"]),
                _fmt(metrics["catastrophic_accepted"]),
                "yes" if metrics["passes_release_evidence_gate"] else "no",
            ]
        )
    body = (
        f"The calibration-only rule requires overlap ≥"
        f"{envelope['registered_cloud_overlap_min']:.0%}, baseline in "
        f"[{envelope['baseline_m_closed_interval'][0]:.3f}, "
        f"{envelope['baseline_m_closed_interval'][1]:.3f}] m, capture usable "
        f"probability ≥{threshold['capture_usable_probability']:.2f}, and post "
        f"precision probability ≥{threshold['post_precision_probability']:.2f}.\n\n"
        + _table(
            (
                "Dataset",
                "Selected/eligible",
                "Groups",
                "Precision",
                "Exact lower 95%",
                "Catastrophic",
                "Gate",
            ),
            rows,
        )
    )
    return str(evaluation["verdict"]), body


def run(args: argparse.Namespace) -> dict[str, Any]:
    census_path = args.census.resolve()
    table_path = args.analysis_table.resolve()
    card_path = args.model_card.resolve()
    evaluation_path = args.evaluation.resolve()
    lodo_path = args.lodo_evaluation.resolve()
    paper_path = args.paper_results.resolve()
    manifest_path = args.aligned_table_manifest.resolve()
    census = json.loads(census_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(table_path)
    model_card = json.loads(card_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    lodo = json.loads(lodo_path.read_text(encoding="utf-8"))
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    json.loads(manifest_path.read_text(encoding="utf-8"))
    rule = (
        json.loads(args.release_rule.read_text(encoding="utf-8"))
        if args.release_rule is not None
        else None
    )
    release_evaluation = (
        json.loads(args.release_evaluation.read_text(encoding="utf-8"))
        if args.release_evaluation is not None
        else None
    )
    search = (
        json.loads(args.release_rule_search.read_text(encoding="utf-8"))
        if args.release_rule_search is not None
        else None
    )
    prospective = (
        json.loads(args.prospective_plan.read_text(encoding="utf-8"))
        if args.prospective_plan is not None
        else None
    )
    post_model = _post_model(evaluation)
    verdict, release_body = _release_section(rule, release_evaluation, search)
    totals = census["totals"]
    census_rows = [
        [
            DATASET_LABELS[dataset],
            _fmt(census["datasets"][dataset]["unique_images"]),
            _fmt(census["datasets"][dataset]["unique_pairs"]),
            _fmt(census["datasets"][dataset]["independence_components"]),
            _fmt(census["datasets"][dataset]["components_by_split"]["evaluation"]),
        ]
        for dataset in DATASET_ORDER
    ]
    prospective_text = (
        f"The frozen primary design requires "
        f"{prospective['primary_design_scenario']['minimum_independent_groups']} "
        f"new groups and {prospective['primary_design_scenario']['target_selected_pairs']} "
        "selected pairs."
        if prospective is not None
        else "E8 is not authorized until a replacement rule qualifies on calibration; "
        "the frozen planning target remains at least 40 new groups and 120 selected pairs."
    )
    figures = "\n".join(
        f"- `{name}`: [{Path(path).name}]"
        f"({Path(os.path.relpath(path, start=args.output.resolve().parent))})"
        for name, path in sorted(paper["figures"].items())
    )
    supported_surface = _surface_rows(paper)
    document = f"""# PanorAi 3.5.0 spherical two-view R,t: aligned retrospective results

Status: **{verdict}**. This is retrospective validation, not prospective
confirmation or a release-reliability claim. Dense stereo, bundle adjustment,
and multiview estimation are outside this analysis; every estimate uses exactly
two spherical panoramas.

## Study question and artifact

The study estimates two complementary quantities:

```text
p_usable_capture = P(accepted | capture) × P(precise | accepted, capture)
p_precise_post = P(precise | returned, algorithm evidence)
```

The aligned replay used PanorAi `{args.panorai_version}`, source commit
`{args.source_commit}`, native spherical DoG batch-two detection, explicit
validity masks, four tangent-patch workers, calibrated tangent RootSIFT, and the
frozen spherical R,t estimator. The primary post model is `{post_model}`.

## Evidence base and leakage control

{_table(("Dataset", "Unique images", "Pairs", "Independent groups", "Evaluation groups"), census_rows)}

Total: {totals['unique_images']:,} unique images, {totals['unique_pairs']:,}
unordered pairs, and {totals['independence_components']} independence
components. Duplicate/reversed pairs, shared-image split leakage, component
split leakage, and group split leakage are all zero. Fitting commands received
development/calibration outcomes only; evaluation outcomes were opened by a
separate evaluation stage.

## Aligned estimator response

{_table(("Dataset", "Pairs", "Returned", "Accepted", "Precise", "Usable", "Catastrophic accepts"), _outcome_counts(rows))}

`precise` means rotation error ≤1° and oriented translation-direction error
≤5°. `usable` means accepted and precise. Translation magnitude is not scored,
because central two-view geometry recovers translation direction only up to
scale.

## Response versus registered-cloud overlap

{_table(("Dataset", "Overlap", "Pairs", "Groups", "Return rate", "Accept rate", "Usable rate", "Catastrophic"), _overlap_rows(paper))}

Overlap is a scanner-assisted capture variable only when the two depth clouds
and their shared coordinate transforms are available before RGB pose
estimation. It must not be advertised as an RGB-only observable without a
separately validated online proxy.

## Supported capture probability surface

{_table(("Overlap", "Baseline (m)", "Support groups", "P(accept)", "P(precise|accept)", "P(usable)"), supported_surface) if supported_surface else "No overlap × baseline cell reached the frozen five-group support minimum."}

Cells with fewer than five independent groups are withheld, rather than
interpolated into capture advice.

## Calibrated component-held-out evaluation

{_table(("Model", "Dataset", "n", "Brier", "Log loss", "ECE", "Predicted", "Observed", "Groups"), _evaluation_rows(evaluation, [(CAPTURE_ACCEPT_MODEL, "Capture acceptance"), (CAPTURE_PRECISE_MODEL, "Capture conditional precision"), (RAW_POST_MODEL, "Post raw score"), (post_model, "Post primary")]))}

The capture usable product is evaluated directly against `accepted AND
precise`; it is not assumed calibrated merely because its two factors were
calibrated separately.

{_table(("Dataset", "n", "Brier", "Log loss", "ECE", "Predicted", "Observed", "Groups"), [[DATASET_LABELS[dataset], _fmt(metrics['count']), _fmt(metrics['brier']), _fmt(metrics['log_loss']), _fmt(metrics['ece_10']), _fmt(metrics['predicted_mean']), _fmt(metrics['prevalence']), _fmt(metrics['independence_components'])] for dataset, metrics in evaluation['usable_products'][CAPTURE_USABLE_MODEL]['datasets'].items()])}

## Leave-one-dataset-out transfer

No outcome from the named target dataset was opened during its fit.

{_table(("Model", "Held-out dataset", "n", "Brier", "Log loss", "ECE", "Groups"), _lodo_rows(lodo, post_model))}

Stanford2D3D and P74 each have only three independent groups in the complete
corpus and one evaluation group. Their results diagnose transfer and failure
modes; they do not support standalone group-generalized reliability claims.

## Replay wall-time diagnostic

Resolution: {paper['engineering_summary']['resolution']} per panorama. Times
refer to a complete two-panorama pair. These observations were collected on a
shared host with independently observed contention and are therefore **not a
controlled performance benchmark**.

{_table(("Dataset", "Pairs", "Detection median", "Detection P95", "Total median", "Total P95", "RSS P95 MiB", "RSS max MiB"), _runtime_rows(paper))}

Runtime and memory are engineering outcomes only; they are not probability
model inputs. A separate frozen timing protocol on an idle host is required for
comparison with the 3–5 s-per-panorama reference range.

## Selective operating rule

{release_body}

The retrospective verdict cannot itself authorize a release. A favorable
`GO_FOR_PROSPECTIVE_CONFIRMATION` result authorizes only E8 collection.

## Capture recommendation

1. Use exactly two calibrated spherical panoramas for the claimed estimator.
2. With registered depth in a shared frame, require at least 50% minimum
   bidirectional cloud overlap as an eligibility boundary; prefer ≥70% where
   the table shows independent support, but do not treat either value as a
   guarantee.
3. Plan or measure baseline independently of the reference pose, and use it
   only inside the calibrated scene-scale support. Record representative scene
   distance or predicted parallax when available.
4. Prefer static structure distributed broadly over the sphere, depth
   variation, and non-negligible parallax. Repetitive structure can preserve a
   strong-looking match set while reversing translation direction.
5. Preserve explicit masks; black pixels are not validity evidence.
6. Release a pose only when public quality acceptance, the supported capture
   envelope, and the frozen post-processing probability gate all agree.
7. Reacquire rather than extrapolate outside any calibrated support region.

For pure RGB capture, the 50% cloud-overlap rule is unavailable until an online
RGB overlap proxy is independently validated.

## Prospective confirmation still required

{prospective_text} The confirmatory gate also requires observed selected-pose
precision ≥95%, a one-sided exact 95% lower bound ≥90%, zero catastrophic
accepts, at most three selected pairs per independent group, and predictions
sealed before reference poses are opened.

## Figures

{figures}

## Reproducibility record

- aligned table manifest: `{manifest_path}` (`{_sha256(manifest_path)}`)
- census: `{census_path}` (`{_sha256(census_path)}`)
- aligned pair table: `{table_path}` (`{_sha256(table_path)}`)
- model card: `{card_path}` (`{_sha256(card_path)}`), profile `{model_card.get('model_profile', 'historical')}`
- evaluation: `{evaluation_path}` (`{_sha256(evaluation_path)}`)
- leave-one-dataset-out evaluation: `{lodo_path}` (`{_sha256(lodo_path)}`)
- paper-results data: `{paper_path}` (`{_sha256(paper_path)}`)
- exact package requirement: PanorAi `{args.panorai_version}` from `{args.source_commit}`

## Limitations

- This is retrospective validation on frozen groups, not E8 confirmation.
- The scanner-assisted overlap model is not an RGB-only deployment model.
- Stanford2D3D and P74 have insufficient independent groups for standalone
  probability claims.
- Baseline in the archived corpora is a retrospective surrogate for a
  quantity that must be planned or tracked independently at deployment.
- No result in this document validates dense stereo, translation magnitude,
  bundle adjustment, or multiview estimation.
- Population-replay wall times were affected by shared-host contention and are
  diagnostic only; performance claims require the separate controlled timing
  protocol.
"""
    output_path = args.output.resolve()
    _atomic_text(output_path, document)
    result = {
        "schema": "panorai-two-view-pose-paper-document/v1",
        "status": verdict,
        "output": {"path": str(output_path), "sha256": _sha256(output_path)},
        "primary_post_model": post_model,
        "pair_count": len(rows),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--analysis-table", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--lodo-evaluation", type=Path, required=True)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--aligned-table-manifest", type=Path, required=True)
    parser.add_argument("--release-rule", type=Path)
    parser.add_argument("--release-evaluation", type=Path)
    parser.add_argument("--release-rule-search", type=Path)
    parser.add_argument("--prospective-plan", type=Path)
    parser.add_argument("--panorai-version", default="3.5.0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
