# Prospective E8 authorization

Status: **GO_FOR_PROSPECTIVE_CONFIRMATION** for the exact candidate and
acquisition design named below. The retrospective release verdict remains
**NO_GO** until a sealed prospective trial passes E8 and is independently
reviewed.

## Authorized scope

The explicit user authorization applies only to:

- PanorAi 3.5.0 from source commit
  `03c5b36b28225b24d3909286bf53250d7b532aa3`;
- the already frozen native spherical frontend, batch of two panoramas,
  four-worker tangent patch extraction, and explicit validity masks;
- `p_usable_capture >= 0.50` and `p_precise_post >= 0.90`;
- three balanced capture domains, 20 new independent groups per domain;
- 16 panoramas and 52 preregistered candidate pairs per group;
- 60 groups, 960 panoramas, and 3,120 candidate pairs in total;
- 40 primary selected pairs per domain, 120 total, capped at three per group;
- the unchanged E8 precision and safety gate.

Every R,t estimate remains a two-view estimate: one candidate row contains
exactly two spherical panoramas. The larger image count creates independent
capture groups and enough candidate pairs to support selective evaluation; it
does not turn the estimator into a multiview method.

## What authorization means

Authorization permits outcome-blind registration, collection, prediction,
sealing, and later confirmatory evaluation under the exact hashed artifacts.
It does not:

- erase the six catastrophic Stanford2D3D poses that made the original
  retrospective rule `NO_GO`;
- validate the post-evaluation 0.90 threshold on old data;
- permit reference poses or errors to influence registration, accrual, ranking,
  or stopping;
- permit substitutions of the wheel, frontend, models, thresholds, domains,
  sample sizes, or group cap without a new amendment and authorization.

## Immutable materialization

`authorize_prospective_candidate.py` validates both drafts and every referenced
hash before creating a new authorization bundle. It never edits the historical
drafts. The bundle contains the source drafts, byte-identical model and route
artifacts, an authorization record, an authorized candidate, an authorized
acquisition plan, and a hash manifest.

The authorization recorded on 2026-10-09 has ID
`AUTH-20261009-PPOST090-BALANCED60` and the following identities:

| Artifact | SHA-256 |
| --- | --- |
| Authorization record | `96b290872e7d66c840c3735239f3a5f78c174dedad14577c4453b0d8947d3484` |
| Authorized candidate | `b7f42cb4187716f52c86885ad0f67e69717472cd503fe7829d15696a472fa8ea` |
| Authorized acquisition plan | `3d50c25cf212dc95c15b0d7a4abea0d2b2000f4daf4af40ba7824b972c48521f` |

The source candidate and plan retained their original SHA-256 identities,
`0578488d6b658c4989113d4121035d15bb1a19078f1bc73f3a3dcfe0ac1cc815`
and
`4fb6b71e46dfe6d51aca8bda09e23b9f72c937d9e47c65406894512f23d56133`,
respectively.

```bash
python benchmarks/two_view_pose_probability/authorize_prospective_candidate.py \
  --candidate-draft /path/to/candidate-draft.json \
  --acquisition-plan-draft /path/to/acquisition-plan.json \
  --configuration /path/to/configuration.json \
  --capture-models /path/to/capture-models.json \
  --post-model /path/to/post-model.json \
  --selective-rule /path/to/selective-rule-amendment.json \
  --readiness-report /path/to/readiness-report.json \
  --scenario-csv /path/to/acquisition-scenarios.csv \
  --expected-package-version 3.5.0 \
  --expected-source-commit 03c5b36b28225b24d3909286bf53250d7b532aa3 \
  --authorization-id AUTH-20261009-PPOST090-BALANCED60 \
  --authorized-at 2026-10-09T00:00:00-03:00 \
  --authorization-source interactive-user-approval \
  --authorization-statement "User authorized the exact p_post >= 0.90 candidate and balanced 60-group/960-panorama/3120-pair plan." \
  --output-dir /path/to/new-authorization-bundle
```

## Next observable boundary

No panorama is synthesized by authorization. Collection can begin only when
new images, explicit validity masks, capture-group identities, and capture-side
measurements exist. Those inputs are audited and frozen with
`prepare_prospective_registry.py`; model outputs are then built with
`build_prospective_predictions.py`. Reference geometry stays inaccessible until
the prediction seal has been written.
