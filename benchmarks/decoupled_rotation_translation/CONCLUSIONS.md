# Conclusions from the decoupled pose benchmark

## Decision

Retain the compatibility estimator as the default and keep
`pose_refinement_method="decoupled"` Experimental and opt-in. The frozen
synthetic evidence demonstrates a large improvement in the intended
far-background/weak-parallax regime, including a calibrated abstention
mechanism, but does not yet cover real feature correspondences.

The held-out run contains 20 cases in each of five conditions and four paired
variants, for 400 observations. Its seed base is `1400000`; every development
seed was below `1300000`. Exact source fingerprints and the raw CSV hash are in
`results/summary.json`.

## Rotation and translation accuracy

In clean weak parallax, the compatibility estimator achieved 10% strict
accuracy, with median rotation error 0.103 degrees and median oriented
translation error 36.110 degrees. The guarded decoupled estimator achieved
100% strict accuracy, 95% high-precision accuracy, median rotation error 0.015
degrees and median translation error 0.954 degrees. There were 18 paired gains
and no losses.

With 5% outliers, count-first and MSAC-first produced zero strict successes.
The unguarded two-stage estimator produced 18/20 strict successes but falsely
accepted the two failures. The 0.15 cluster-margin gate accepted 17/20 cases;
all 17 satisfied both the strict criterion (`R < 5`, `t < 10` degrees) and the
high-precision criterion (`R < 0.1`, `t < 5` degrees). Among accepted poses,
rotation median/P95 was 0.016/0.030 degrees and translation median/P95 was
0.794/2.580 degrees. Three cases were explicitly classified
`translation-ambiguous`; the false-accept rate was zero.

In the sub-meter mixed-depth condition, both baseline and decoupled variants
retained 100% strict and high-precision accuracy. Translation median/P95
improved from 0.245/1.793 degrees to 0.160/0.489 degrees. Rotation P95 improved
from 0.098 to 0.031 degrees, although the median rose from 0.014 to 0.023
degrees; both remain far below the declared high-precision threshold.

## Fallback and abstention

All 20 well-conditioned cases classified the rotation-only model as
inapplicable and retained the exact compatibility result: 100% strict accuracy
and no numerical regression. All 20 deliberately unobservable cases had fewer
than five translation-pool rays and were rejected as
`translation-unobservable`. Under the deliberately permissive benchmark
policy, count-first and MSAC falsely accepted every unobservable pose.

## Cost

Median guarded runtime was approximately 0.97 seconds with outliers, 0.88
seconds in clean weak parallax and 1.00 seconds in the sub-meter condition on
the recorded macOS arm64 environment. This run measures the complete estimator
and not an isolated kernel. The bounded two-stage work uses 512 rotation
proposals and at most 100 consensus-refit steps; translation proposals are
exhaustive only inside a pool capped at 64 rays.

## Interpretation and next evidence

The result validates the causal diagnosis from GEO-014: the distant majority
is an excellent rotation measurement but a poor translation measurement.
Using the rotation-only residual to partition those regimes turns a 5-of-400
translation signal into a small robust-consensus problem. A score-margin gate
is necessary; the solver alone still produces occasional plausible but wrong
translation axes.

The next required evidence is a frozen corpus of real spherical feature
matches with independently measured or externally optimized relative poses.
The method should not become the default until it preserves its conditional
precision across match-density, texture, dynamic-object and calibration-error
domains.
