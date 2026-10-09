# Relative-pose operating-envelope results

Run date: 2026-10-08

Source commit: `67e87591a7d52039f570083155033283a6255b63` plus the uncommitted,
benchmark-only VAL-017 harness

Environment: macOS 26.6.2 arm64, Python 3.12.4, NumPy 1.26.4

Raw result: `/private/tmp/panorai-val017-relative-pose-envelope/results.json`

## Scope

This run evaluates only the public spherical two-view estimator of rotation and
oriented translation direction. Detection, description, matching and dense
stereo are excluded. The profile is MSAC-first, bounded non-minimal refit,
1-degree inlier threshold, six stability trials, 128 model-competition trials,
the spatially weighted five-point sampler and the unchanged public conservative
acceptance policy.

There were 71 frozen conditions and three deterministic seeds per condition,
for 213 estimator calls. Strict correctness means rotation error at most 5
degrees and oriented translation error at most 10 degrees. Precise correctness
means 1 and 5 degrees respectively.

## Main results

### Parallax and inlier ratio

| Median parallax | Observed behavior with 80 full-sphere matches, 0.1 deg noise |
| ---: | --- |
| 0.10 deg | 0/18 accepted; translation generally unobservable even at 100% inliers |
| 0.25 deg | 0/18 accepted; accuracy improves at high inlier ratio but policy correctly abstains |
| 0.50 deg | 0/18 accepted; returned translation remains unstable |
| 1.00 deg | Transition region; 12/18 strict, 10/18 accepted, including one false acceptance at 25% inliers |
| 2.00 deg | 17/18 strict and 16/18 accepted; all accepted poses strict; 3/3 accepted at every inlier ratio >=55% |
| 5.00 deg | 18/18 strict and accepted, including the 25% inlier condition |

The only false acceptance in this family had 1-degree target parallax and 25%
true inliers. Its rotation error was 0.367 degrees but oriented translation
error was 14.547 degrees. The estimator reported 23/80 inliers, 1.074 degrees
of median parallax and 9.430 degrees translation P90 stability, narrowly inside
the current 10-degree policy bound.

### Noise and parallax

With 80 full-sphere matches and 70% true inliers:

- 2-degree parallax was strict and accepted in 12/12 trials across 0.02, 0.10,
  0.25 and 0.50-degree independent tangent noise;
- 1-degree parallax was strict in 7/12 and accepted in 6/12;
- 0.5 degree or less produced no accepted pose across all 24 trials.

This clean synthetic noise model does not include correlated descriptor errors,
blur, rolling shutter, calibration bias or non-central capture.

### Match count and angular support

This grid used 1-degree parallax, 70% true inliers and 0.1-degree noise.

| Geometry | 12 matches | 20 matches | 40 matches | 80 matches | 160 matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full sphere: accepted / strict | 0/3 / 0/3 | 0/3 / 1/3 | 2/3 / 2/3 | 2/3 / 3/3 | 3/3 / 3/3 |
| Equatorial band +/-10 deg: accepted / strict | 0/3 / 2/3 | 0/3 / 0/3 | 2/3 / 1/3 | 2/3 / 2/3 | 3/3 / 3/3 |
| Compact 15-deg cap: accepted / strict | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 | 0/3 / 0/3 |

A compact cap remains unusable even with 160 correspondences: zero were
accepted and the median returned translation error was 56.010 degrees. The
40-match equatorial-band cell contained a policy-accepted pose with 10.342
degrees translation error. Count and scalar entropy therefore cannot replace a
two-dimensional angular-spread check.

### Negative controls

- Pure rotation: 0/3 accepted.
- All-outlier correspondences: 0/3 returned and 0/3 accepted.
- Five-degree compact cap: 0/3 accepted.
- One-degree great-circle band: 1/3 accepted; that accepted pose was strict,
  while the other two were rejected by stability/cheirality evidence.

The controls show safe abstention for pure rotation and random matches, but
also confirm that near-one-dimensional angular arrangements need an explicit
capture rule rather than relying only on the present quality policy.

## Provisional boundary

The present development evidence supports the following candidate nominal
region:

- median inlier parallax at least 2 degrees;
- true inlier fraction at least 55%;
- sufficient texture for roughly 160 verified candidates and about 88 or more
  robust inliers;
- broad support in longitude and latitude, not a compact cap or one dominant
  scene band;
- scene overlap target at least 70%, with 50% only as an eligibility floor;
- unchanged estimator quality acceptance plus neighboring-view consistency.

The 2-degree/55% boundary was 3/3 accepted and strict. This is enough to choose
a confirmatory boundary but far too small to estimate production reliability.
It must be frozen and tested on more synthetic seeds and held-out real captures
before the word “reliable” is used in release documentation.

## Performance observation

Most 80-correspondence calls took approximately 2.2–3.0 seconds. Concentrated
and some 160-correspondence cases were much slower, with observed individual
calls above 40 seconds. This is estimator-only source-checkout timing on one
machine and reveals a tail-latency condition that must be profiled separately;
it is not comparable to the previously measured full frontend pair time.
