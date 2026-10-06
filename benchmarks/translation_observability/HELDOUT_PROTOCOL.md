# GEO-017 fixed-rotation translation held-out protocol

Status: frozen before prediction and before reference access.

## Question

Does robustly refitting only the unit translation direction from epipolar-plane
normals, while preserving the MSAC+all-inlier-refit rotation exactly, improve
translation accuracy without reducing selective reliability?

## Frozen source

- Starting commit: `2503319d5024dfad6d2b9da441700dbea1a5092b`
- Relative-pose source SHA-256:
  `2176b4c5f70dcff42654228a916d98d9c62833d22dd8bf2629174370186660c0`
- Benchmark runner SHA-256:
  `b0221bec13dffb7848701b21225d134bcd417e2daa8b90ca8fd29fa11cfc22cb`
- Prediction seed: `20261205`

## Data separation

Use the already downloaded Hilti-Trimble-Oxford
`floor_2_2025-10-28_run_2` cam0 bag and Kalibr calibration. The `estimate`
command has no ground-truth argument. It must freeze a read-only prediction
before `evaluate` may open the LiDAR-derived cam0 trajectory.

The held-out offsets are `9,16,24,31,38,52` seconds, sampled at 5 Hz with
three frames per window, producing 12 adjacent pairs. They do not overlap any
frame used by GEO-016 development offsets `6,20,34,48,62,76` or prior held-out
offsets `12,27,42,57,72,86`.

## Frozen variants

Baseline:

```python
RelativePoseOptions(
    random_seed=20261205 + pair_index,
    hypothesis_ranking="msac-first",
    nonminimal_refit_max_steps=100,
)
```

Candidate: the same configuration plus:

```python
RelativePoseOptions(
    pose_refinement_method="fixed-rotation-translation",
    fixed_rotation_refit_max_steps=20,
    translation_observability_trials=32,
    translation_observability_fraction=0.8,
    translation_observability_min_effective_support=5.0,
    translation_observability_min_spectral_gap=0.01,
    translation_observability_max_p90_deg=10.0,
    fixed_rotation_require_observability=True,
)
```

All other benchmark options retain their runner defaults, including 512
RANSAC trials and the same SIFT/FLANN correspondences for both variants.

## Metrics

All rates use all 12 ordered pairs; no-return counts as failure. Angular means,
medians and higher-method P95 use returned poses and state their denominator.

- R and oriented-t mean, median and P95;
- strict success: `R < 5 deg` and oriented `t < 10 deg`;
- high precision: `R < 1 deg` and oriented `t < 5 deg`;
- return rate, accepted coverage, accepted strict precision, false accepts;
- paired t wins/losses on common returns;
- observability reasons, effective support, spectral gap and bootstrap P90;
- runtime median/P95 and peak RSS.

## Acceptance criteria

The candidate is a positive held-out result only if every condition holds:

1. paired candidate rotations equal baseline rotations within `1e-9` degrees;
2. median oriented-t error is at least 5% lower;
3. oriented-t P95 is lower and paired t wins exceed losses;
4. strict and high-precision counts do not decrease;
5. accepted strict precision does not decrease and false accepts do not
   increase;
6. candidate median runtime is no more than 1.5 times baseline.

Failure of any condition is reported; thresholds will not be changed after
prediction. Even a passing result remains Experimental because this phase uses
one camera and one sequence.
