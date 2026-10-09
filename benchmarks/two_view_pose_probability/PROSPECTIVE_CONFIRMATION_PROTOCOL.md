# E8 prospective confirmation protocol

Status: frozen planning protocol; data collection has not started.

## Scope

Confirm a single, frozen PanorAi two-view spherical R,t pipeline and selective
rule on new independent capture groups. Dense stereo, bundle adjustment,
multiview estimation, and post-hoc threshold changes are excluded.

The current retrospective rule is NO-GO. E8 must not start as a release trial
until the frontend configuration is unified across domains and a replacement
rule is frozen. The acquisition and power requirements below can nevertheless
guide collection now.

The candidate unified frontend is the optimized public route verified in the
exact-`v3.5.0` mechanism replay: native spherical DoG detection in a
two-image batch, explicit masks, 4,096-keypoint capacity, four tangent-patch
workers, 48 by 48 upright tangent patches, and the calibrated one-scale
RootSIFT profile. The exact wheel and complete serialized configuration must be
frozen again at E8 start; this description alone is not an artifact identity.
The current aligned retrospective replay is frozen to PanorAi `3.5.0` and the
source tree `c0a7d8bbf7ed1f29ff449e77d7e4b12afda40043`; E8 must either reuse that
exact artifact or explicitly restart calibration for a newer artifact.

## Unit and independence

- One estimator input contains exactly two panoramas.
- An independent group is a new building, industrial acquisition family, or
  physically separated sequence with no shared image or station.
- At most three selected pairs from one group contribute to the primary gate.
- Extra correlated pairs may diagnose failures but do not increase the primary
  sample size.
- Groups used in E0–E7 cannot be relabelled as prospective.

## Frozen acquisition fields

Before reference geometry is opened, record:

- image IDs, group ID, timestamp, sensor and resolution;
- explicit validity masks;
- planned or tracked baseline;
- registered-cloud overlap when shared-frame depth is available;
- representative scene distance and predicted parallax when depth is available;
- independent per-image blur, clipping, valid support, and spherical texture
  occupancy;
- all frontend, matching, estimator, and quality diagnostics;
- competing translation hypotheses, translation-direction stability, parallax
  support, and spatial evidence concentration needed to detect the
  repetitive-scene ambiguity observed in the exact-main replay;
- the exact PanorAi wheel filename, version, source commit, SHA-256, Python,
  OpenCV, CPU, thread count, and serialized configuration.

Dataset or site identity is evaluation metadata, never an input to the deployed
probability models.

## Execution order

1. Build and audit the wheel from the reviewed commit; install it outside the
   repository.
2. Freeze frontend, descriptor, matcher, R,t estimator, probability model,
   calibration parameters, capture envelope, and selective thresholds.
3. Register the pair/group manifest without reference outcomes.
4. Process every pair and write immutable probabilities and selection decisions.
5. Hash and seal the prediction table.
6. Only then open reference poses and compute errors.
7. Join by dataset/pair ID, audit duplicates and split leakage, and publish every
   excluded or missing pair with a reason.

## Executable pause and resume boundary

`run_prospective_confirmation.py` implements the irreversible boundary between
outcome-blind prediction and reference-pose evaluation. It deliberately refuses
the current `NO_GO` retrospective candidate. Before collection can be treated
as confirmatory, the coordinator must authorize one exact replacement candidate
with `authorization: "GO_FOR_PROSPECTIVE_CONFIRMATION"`, zero deviations, the
gate below, and SHA-256 identities for the wheel, serialized configuration,
capture model, post-processing model, and selective rule.

The seal command requires four immutable inputs:

- `candidate.json`: exact PanorAi version and source commit, artifact hashes,
  authorization, frozen gate, and an empty deviations list;
- `registry.jsonl`: one outcome-blind row per two-image pair, with schema,
  `pair_id`, `group_id`, `domain_id`, two distinct `image_ids`, and capture
  fields under `capture`;
- `predictions.jsonl`: one row per registered pair, with the two model
  probabilities, their declared product, frontend/estimator decision flags,
  and the frozen primary-sample decision;
- `retrospective-groups.jsonl`: every E0--E7 independence component, used to
  reject reuse in E8.

Reference errors, derived accuracy labels, and ground-truth poses are forbidden
recursively in registry and prediction records. A panorama cannot cross
prospective groups, reversed pairs are rejected, and no group may contribute
more than three primary pairs. The declared future reference file must not
exist when the following command runs:

```bash
python benchmarks/two_view_pose_probability/run_prospective_confirmation.py \
  seal \
  --candidate /path/to/candidate.json \
  --registry /path/to/registry.jsonl \
  --predictions /path/to/predictions.jsonl \
  --retrospective-groups /path/to/retrospective-groups.jsonl \
  --future-references /path/to/references-after-seal.jsonl \
  --expected-package-version 3.5.0 \
  --expected-source-commit 03c5b36b28225b24d3909286bf53250d7b532aa3 \
  --output /path/to/prediction-seal.json
```

The resulting seal is the resumption checkpoint. It binds all four input files,
the predeclared future reference path, candidate identity, pair/group counts,
and primary decisions by SHA-256. Keep those files read-only. After independent
reference geometry has been produced, resume only with:

```bash
python benchmarks/two_view_pose_probability/run_prospective_confirmation.py \
  evaluate \
  --seal /path/to/prediction-seal.json \
  --references /path/to/references-after-seal.jsonl \
  --output-dir /path/to/new-e8-evaluation
```

Evaluation fails closed if a sealed byte changed or the reference path differs.
It independently derives primary, precise, and catastrophic outcomes from R/t
errors, computes the exact one-sided bound and 10,000-repeat group bootstrap,
reports reliability by capture domain and capture-variable ranges, and emits a
joined audit table plus the gate verdict. The output directory must be new.
Creating this executor does not authorize or start E8 data collection.

Before predictions, image and pair inputs must pass the separate
[`PROSPECTIVE_REGISTRY_CONTRACT.md`](PROSPECTIVE_REGISTRY_CONTRACT.md). Its
audit mode counts real panoramas, pairs and groups during acquisition without
accessing outcomes. Its freeze mode refuses any candidate or acquisition plan
that lacks explicit authorization and binds explicit validity masks and capture
variables into the registry consumed by this seal.

For consistency with the frozen retrospective evaluator, the zero-catastrophe
release gate applies to the primary selected output: a frontend-accepted pose
that the selective rule withholds is not a released pose. The report still
counts every frontend-accepted catastrophe separately so abstention cannot hide
an estimator failure mode.

## Primary gate

The confirmatory result passes only if all conditions hold:

- selected-pose precision is at least 95%;
- the exact one-sided 95% lower bound is at least 90%;
- no catastrophic pose is present in the primary selected output;
- at least 40 new independent groups contribute;
- at least 120 selected pairs contribute, with at most three per group;
- calibration and reliability are reported per capture domain;
- no supported capture condition or diagnostic was changed after unsealing.

The target gives approximately 80% simulated power under 97% true precision,
ICC 0.20, and a 0.1% marginal catastrophic rate. The simulation is a planning
model, not evidence of actual performance; the complete sensitivity grid is
generated by `plan_prospective_confirmation.py`.

## Failure and stopping policy

- Any catastrophic pose in the primary selected output makes the release gate
  fail; frontend-accepted poses withheld by the selective rule remain in the
  diagnostic denominator.
- Missing masks, unresolved provenance, use of a different frontend, or opening
  reference geometry before predictions are sealed invalidates confirmatory
  status for the affected group.
- Data collection is sequential by selected pairs. It may stop for futility,
  safety, or resource constraints, but an early underpowered sample cannot be
  reported as confirmation.
- Failed pairs remain in coverage denominators.

## Required report

Report candidate, returned, accepted, and selected counts; unique images;
groups; per-group pair counts; capture-variable ranges; precision and coverage;
catastrophic count; exact and component-bootstrap intervals; reliability bins;
all failure reasons; wall time and memory; artifact hashes; and deviations from
this protocol.
