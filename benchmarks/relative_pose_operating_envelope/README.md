# Relative-pose operating envelope

This benchmark maps when the public spherical two-view estimator returns a
correct and policy-accepted rotation and oriented translation direction. It is
deliberately independent of feature extraction, matching and dense stereo.

The frozen development grid varies:

- median translation parallax and true inlier ratio;
- correspondence count and angular coverage;
- independent tangent-plane bearing noise and parallax;
- pure rotation, all-outlier, great-circle and clustered-cap controls.

Run the preregistered three-seed development grid with:

```bash
python benchmarks/relative_pose_operating_envelope/run_benchmark.py \
  --output-dir /private/tmp/panorai-val017-relative-pose-envelope \
  --seeds 3
```

The script uses the public estimator, the current best MSAC-first profile with
bounded non-minimal refit, and the unchanged conservative public acceptance
policy. It writes raw per-case JSON, aggregated results and a technical report.

This is source-checkout synthetic evidence. It does not certify a wheel and it
does not turn scene-overlap percentage into an estimator guarantee. Overlap is
an upstream application variable whose effect must be related empirically to
match count, inlier ratio, angular coverage and parallax on a frozen real-pair
corpus.
