# GEO-016 held-out protocol

Frozen before estimating or opening reference poses for these timestamps.

## Candidate and baseline

- Baseline: `hypothesis_ranking="count-first"`, no non-minimal Essential
  refit, joint pose refinement.
- Sole candidate: `hypothesis_ranking="msac-first"`, all-inlier Essential
  refit until the inlier set stabilizes, cycles, or reaches 100 steps, joint
  pose refinement.
- Diagnostic only: the GEO-015 guarded decoupled method. It cannot determine
  whether the sole candidate meets the held-out criterion.

Every other estimator, feature, matcher, stability, competition, and random
seed setting remains identical within a pair.

## Frozen real-pair sample

- Dataset: Hilti-Trimble-Oxford `floor_2_2025-10-28_run_2`.
- Camera: calibrated central `cam0` only; no `cam1`, IMU, or metric scale enters
  estimation.
- Sampling: 5 Hz, three frames at each offset
  `12,27,42,57,72,86` seconds.
- Pairs: the two adjacent pairs inside each window, 12 total.
- These frames do not overlap the descriptive offsets
  `6,20,34,48,62,76` seconds.
- Prediction is frozen read-only before the evaluator is given the trajectory.

## Metrics and success criterion

The independent oracle linearly interpolates camera centers, SLERPs TUM
quaternions, composes camera-2-from-camera-1, and changes basis from
OpenCV/Kalibr to PanorAi coordinates.

Compared with count-first over all returned pairs, the candidate is considered
a material held-out improvement only if all conditions hold:

1. median rotation error is at least 25% lower;
2. median oriented translation error is at least 15% lower;
3. P95 oriented translation error is lower;
4. strict accuracy (`R < 5 deg`, oriented `t < 10 deg`) is not lower;
5. high-precision count (`R < 1 deg`, oriented `t < 5 deg`) is not lower.

Acceptance precision, paired win/loss counts, runtime, match density, baseline,
and reference motion are secondary diagnostics. This one-sequence result cannot
promote an Experimental method to the default regardless of outcome.
