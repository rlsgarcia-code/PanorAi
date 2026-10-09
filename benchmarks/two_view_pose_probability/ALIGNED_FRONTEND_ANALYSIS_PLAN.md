# Frozen aligned-frontend analysis plan

Status: frozen before population replay completion and before any aligned
evaluation-split aggregate was computed. This is a retrospective validation
plan, not a prospective preregistration.

## Prior exposure

The analyst had already inspected the historical VAL-018 outcomes, five
editorially selected mechanism pairs, and one random v2 pilot pair per dataset.
Those observations motivated explicit translation-orientation diagnostics.
They prevent this analysis from being described as confirmatory. No aggregate
outcome from the full aligned population had been computed when this plan was
frozen.

## Population and artifact

- Exactly 2,385 frozen two-image pairs: 1,890 Matterport360, 450 Stanford2D3D,
  and 45 P74.
- Existing development, calibration, and evaluation components remain
  unchanged; pairs sharing an image remain in one component.
- PanorAi `3.5.0`, wheel SHA-256
  `e861dafbaa5991aef77dd512b3ef1bf6fdc7967d10fbd236d850bab5c1a5f8a7`.
- Frozen PanorAi `v3.5.0` tag commit `03c5b36` (the fetched `origin/main` at
  protocol freeze) and source tree
  `c0a7d8bbf7ed1f29ff449e77d7e4b12afda40043`.
- Native spherical DoG detection in one batch of two, 4,096-keypoint capacity,
  explicit validity masks, four tangent-patch workers, 48 by 48 upright tangent
  patches, calibrated one-scale RootSIFT, the frozen matcher, and the frozen
  spherical R,t estimator.

Every pair remains in the denominator. The aligned table is invalid until all
2,385 schema-v2 result objects pass route, identity, and outcome-consistency
checks.

The post-replay implementation is frozen by
`ALIGNED_ANALYSIS_CODE_LOCK.json`. The sealed runner validates every listed
SHA-256 before creating its output directory; a code mismatch is a protocol
failure, not permission to update the lock after seeing aggregate outcomes.

Before any accuracy aggregate was opened, an outcome-blind inspection of six
recent timing records found shared-host contention: detection ranged from
6.18 to 35.17 seconds per pair and one pose stage reached 73.35 seconds. The
lock was amended only to label replay wall times as diagnostic and require the
separate `CONTROLLED_TIMING_PROTOCOL.md`; probability features, outcomes,
splits, thresholds, model fitting, and accuracy figures were unchanged.

## Outcomes

- `accepted`: the public frozen quality policy accepts the returned pose.
- `precise`: rotation error at most 1 degree and oriented translation-direction
  error at most 5 degrees.
- `usable`: `accepted AND precise`.
- `catastrophic_accepted`: accepted but rotation exceeds 15 degrees or oriented
  translation-direction error exceeds 30 degrees.

Translation magnitude is excluded because calibrated central two-view geometry
recovers only translation direction up to scale.

## Pre-capture models

The primary capture model is unchanged:

```text
P(accepted | minimum registered-cloud overlap, baseline)
P(precise | accepted, minimum registered-cloud overlap, baseline)
p_usable_capture = their product
```

These predictors are available to the capture system or operator when depth is
registered. Reference pose errors and algorithm diagnostics are prohibited.
The public-only sensitivity model may additionally use baseline/depth ratio and
RGB similarity because P74 does not provide those fields consistently.

## Post-processing models

Historical raw-score, support, common, and public-full models remain as
ablations. The aligned primary post-processing candidate is
`post-precise-aligned-orientation`, fitted only to returned poses. Its frozen
predictors are:

1. raw quality score;
2. match count, inlier count, and inlier ratio;
3. median parallax and cheirality ratio;
4. coverage entropy in each panorama;
5. translation stability P90;
6. Essential-model score margin;
7. raw and parallax-weighted translation-orientation cheirality margins;
8. median triangulation angle used for translation orientation;
9. the estimator's translation-orientation ambiguity flag.

No runtime, memory, dataset identity, reference error, ground-truth overlap, or
post-hoc failure label enters this model. Runtime and memory are reported only
as engineering outcomes.

The aligned selective-rule candidate uses
`post-precise-aligned-orientation` as its post-processing probability. The
historical rule remains reproducible with `post-precise-raw-score`; the two
must not be mixed when a frozen rule is evaluated. Capture and post thresholds
are selected from the already declared grid using calibration components only.

## Fitting and evaluation

1. Development components select logistic L2 regularization by deterministic
   group-fold validation.
2. Calibration components fit the Platt calibration map.
3. Evaluation outcomes are unavailable to both operations.
4. Evaluation reports Brier score, log loss, ECE, calibration intercept/slope,
   reliability bins, component-bootstrap intervals, coverage, selective
   precision, and catastrophic accepted poses per dataset.
5. Leave-one-dataset-out fits open no outcome from the held-out dataset. With
   fewer than five source groups, the declared fixed L2 is used rather than a
   misleading tuned estimate.
6. Rows missing any required predictor receive no probability; missingness and
   coverage remain explicit.

## Interpretation gate

The aligned analysis may show that the optimized frontend improves acceptance,
precision, or calibration. It cannot establish release reliability by itself.
The retrospective rule remains `NO-GO` unless a replacement rule is frozen on
development/calibration components and then passes the independent E8 protocol
with 40 new groups, 120 selected pairs, at least 95% observed precision, a
one-sided exact 95% lower bound of at least 90%, and zero catastrophic accepts.
`GO_FOR_PROSPECTIVE_CONFIRMATION` authorizes only that E8 collection; it is not
a release verdict and does not relax any E8 threshold.
