# Prospective E8 candidate amendment

Status: **DRAFT_ONLY_NOT_AUTHORIZED**. The retrospective release verdict
remains **NO_GO**.

## Purpose

The calibration-selected rule used post-processing probability threshold 0.50.
It failed on the untouched evaluation split because six selected Stanford2D3D
poses were catastrophic. This amendment identifies a stricter hypothesis for a
new prospective trial; it does not reinterpret the failed evaluation as a
release success.

The proposed change keeps the exact PanorAi 3.5.0 frontend, two capture models,
post-processing model, capture envelope, and `p_usable >= 0.50` requirement. It
changes only the post-processing precision threshold from 0.50 to 0.90.

## Why 0.90 is admissible for prospective testing

The 0.90 threshold was one of the five thresholds in the frozen calibration
grid before evaluation outcomes were opened. Its independently recomputed
calibration result was:

| Calibration result | Value |
| --- | ---: |
| Selected pairs | 39/270 |
| Independent components | 8 |
| Precise selected pairs | 39/39 |
| Selected precision | 100% |
| Exact one-sided 95% lower bound | 92.61% |
| Catastrophic selected poses | 0 |

The original deterministic tie-break chose 0.50 because it retained one more
calibration pair. Choosing 0.90 now is informed by the already-open evaluation
split. It is therefore post-evaluation hypothesis generation even though the
threshold itself was pre-specified. Only genuinely prospective E8 data can
test it.

## Retrospective hypothesis-generation result

| Dataset | Selected | Precise | Precision | Exact lower 95% | Catastrophic |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matterport360 | 10/270 | 10 | 100% | 74.11% | 0 |
| Stanford2D3D | 22/150 | 21 | 95.45% | 80.19% | 0 |
| P74 | 2/15 | 2 | 100% | 22.36% | 0 |
| Pooled descriptive only | 34/435 | 33 | 97.06% | 86.79% | 0 |

The pooled point estimate exceeds 95%, but its exact lower bound remains below
90%. The domains also have unequal and insufficient independent-group support.
These numbers justify testing the hypothesis; they do not satisfy the release
gate.

## Capture burden

The retrospective selection rate is 34/435, or 7.82%. If that rate transferred
unchanged, approximately 1,536 candidate pairs would have to be processed to
accrue 120 selected pairs. E8 still requires at least 40 new independent groups
and at most three primary selected pairs from any group. Collection should
therefore register more candidate pairs than the final primary sample and must
not stop after inspecting reference outcomes.

## Frozen draft identity

The generated draft uses the exact PanorAi 3.5.0 wheel from source commit
`03c5b36b28225b24d3909286bf53250d7b532aa3` and wheel SHA-256
`e861dafbaa5991aef77dd512b3ef1bf6fdc7967d10fbd236d850bab5c1a5f8a7`.

| Artifact | SHA-256 |
| --- | --- |
| Candidate draft | `0578488d6b658c4989113d4121035d15bb1a19078f1bc73f3a3dcfe0ac1cc815` |
| Optimized-route configuration | `e3072b7bd34602c245eac2695c3030adfed4b92e314706407041510ece531ba2` |
| Capture-model snapshot | `9ffedc15f37d9af82dd4882b61dd7598050946bdeae872885eed3d773a8f288e` |
| Post-model snapshot | `9999857efb9fa404bc51f5fb31bc1e50760f1899c6c32bc334e278c8add55051` |
| Selective-rule amendment | `2adaa273b5d6224d65545bbae5a6032e0d0974371ecfc2b37a6dfa7b4e16e208` |
| Readiness report | `f781179207bbfea92c7fdcb6dcacdc1cd52f9094552fb44d39b73b5e069e77fb` |

The candidate contains `authorization: "DRAFT_ONLY_NOT_AUTHORIZED"`; the E8
seal executor must reject it. Authorization requires a separate, explicit
decision after reviewing the lower coverage and acquisition burden.

## Reproduction

Run `prepare_prospective_candidate_draft.py` with the frozen aligned features,
predictions, development/calibration outcomes, evaluation outcomes, model card,
release rule, controlled route validation, and the exact wheel identity. The
command fails if input hashes differ from the release rule, the requested
threshold was not a qualifying frozen-grid candidate, or the route is not the
native batch-two/four-worker/explicit-mask configuration.

```bash
python benchmarks/two_view_pose_probability/prepare_prospective_candidate_draft.py \
  --features /path/to/aligned-table/features.jsonl \
  --predictions /path/to/models/predictions.jsonl \
  --training-outcomes /path/to/outcomes-development-calibration.jsonl \
  --evaluation-outcomes /path/to/outcomes-evaluation.jsonl \
  --release-rule /path/to/release-rule.json \
  --model-card /path/to/model-card.json \
  --route-validation /path/to/route-validation.json \
  --wheel-sha256 e861dafbaa5991aef77dd512b3ef1bf6fdc7967d10fbd236d850bab5c1a5f8a7 \
  --expected-package-version 3.5.0 \
  --expected-source-commit 03c5b36b28225b24d3909286bf53250d7b532aa3 \
  --capture-threshold 0.50 \
  --post-threshold 0.90 \
  --output-dir /path/to/new-prospective-candidate-draft
```
