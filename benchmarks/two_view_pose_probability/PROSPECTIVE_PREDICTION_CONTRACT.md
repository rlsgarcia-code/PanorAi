# Prospective probability and primary-selection contract

Status: executable outcome-blind contract. It does not authorize E8.

## Inputs

`build_prospective_predictions.py` consumes:

- the exact candidate and acquisition-plan artifacts;
- the frozen image/pair registry;
- one frontend result per registered pair;
- frozen capture-model, post-model, and selective-rule snapshots.

Frontend rows contain only pair/group/domain identity, `returned`, public
quality `accepted`, and the declared post-processing diagnostics. Reference
rotation, translation, pose errors, derived outcomes, and ground truth are
recursively forbidden.

## Probabilities

The two capture models use only registered-cloud overlap and baseline. They
produce:

- `p_accept_capture`;
- `p_precise_capture_given_accept`;
- `p_usable_capture`, computed exactly as their product.

The post-processing model is evaluated only when the estimator returned a
pose. It uses the frozen 14-feature profile:

- raw public quality score;
- match and inlier counts;
- inlier ratio;
- median parallax and cheirality ratio;
- coverage entropy in both panoramas;
- translation stability P90;
- essential-model score margin;
- raw and parallax-weighted translation-orientation margins;
- median translation-orientation triangulation angle;
- translation-orientation ambiguity flag.

For a pair without a returned pose, `p_precise_post` is serialized as zero and
`post_probability_available` is false. Such a pair cannot be accepted or
selected. Runtime, memory, dataset identity, and reference-derived fields are
not model inputs.

## Selective decision

The draft rule selects a pair only when all conditions hold:

1. the public estimator quality decision accepted the returned pose;
2. `p_usable_capture >= 0.50`;
3. `p_precise_post >= 0.90`.

All selected and withheld pairs remain in the prediction file. The primary
confirmatory subset is chosen without reference outcomes, separately inside
each domain:

1. descending post-processing precision probability;
2. descending usable capture probability;
3. candidate-bound SHA-256 tie break;
4. at most three primary pairs from one group;
5. stop after the frozen per-domain primary target.

This ordering maximizes predicted reliability without allowing manual pair
choice or ground-truth inspection. Extra qualifying pairs remain selected
diagnostics but do not increase the primary sample.

## Audit and freeze

Audit mode can be used with the unauthorized draft to verify feature
completeness, probability computation, selection counts, group caps, and domain
accrual:

```bash
python benchmarks/two_view_pose_probability/build_prospective_predictions.py \
  audit \
  --candidate /path/to/candidate-draft.json \
  --acquisition-plan /path/to/acquisition-plan.json \
  --registry /path/to/prospective-registry.jsonl \
  --frontend /path/to/frontend-results.jsonl \
  --capture-models /path/to/capture-models.json \
  --post-model /path/to/post-model.json \
  --selective-rule /path/to/selective-rule-amendment.json \
  --output-dir /path/to/new-prediction-audit
```

Freeze mode additionally requires the exact candidate authorization
`GO_FOR_PROSPECTIVE_CONFIRMATION`, an `AUTHORIZED_FOR_COLLECTION` acquisition
plan bound to that candidate, and the complete selected target in every domain.
Failure is transactional: no partial output directory is created.

```bash
python benchmarks/two_view_pose_probability/build_prospective_predictions.py \
  freeze \
  --candidate /path/to/authorized-candidate.json \
  --acquisition-plan /path/to/authorized-acquisition-plan.json \
  --registry /path/to/prospective-registry.jsonl \
  --frontend /path/to/frontend-results.jsonl \
  --capture-models /path/to/capture-models.json \
  --post-model /path/to/post-model.json \
  --selective-rule /path/to/selective-rule-amendment.json \
  --output-dir /path/to/new-frozen-predictions
```

The resulting `predictions.jsonl` is accepted by
`run_prospective_confirmation.py seal`, which binds it before reference poses
can be opened.
