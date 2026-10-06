# Conclusions from the scoring/refit ablation

## Decision

Keep `count-first` with non-minimal refit disabled as the default estimator.
The held-out source-checkout ablation does not justify promoting MSAC-first,
scale-marginal-first, or all-inlier refit. The new controls remain useful for
reproducible experiments and preserve current behavior at their defaults.

The experiment contains 20 frozen cases in each of five conditions and six
variants. It uses seed base `1100000`; development observations used the
disjoint `810000` series. The raw CSV hash and source fingerprints are in
`results/summary.json`.

## Effect of hypothesis ranking

MSAC-first consistently selected a more accurate rotation in weak-parallax
scenes. Without refit, its median rotation error fell from 0.096 to 0.027
degrees in the clean weak condition and from 0.092 to 0.027 degrees with 5%
outliers.

That rotation gain did not transfer to translation direction. In the clean
weak condition, median translation error increased from 40.232 to 55.987
degrees and strict accuracy fell from 10% to 5%, with one paired gain and two
paired losses. With outliers, median translation error increased from 54.603
to 78.829 degrees and strict accuracy fell from 5% to 0%, with one paired
loss. MSAC therefore improves objective 1 (rotation precision) but is not a
safe complete-pose selection policy.

Scale-marginal-first also reduced median rotation error, but exchanged two
strict gains for two strict losses in the clean weak condition and increased
median translation error. Promoting the existing soft score ahead of inlier
count is likewise not justified.

## Effect of all-inlier Essential refit

Under the default count-first ordering, refit reduced median translation error
from 0.297 to 0.196 degrees in the ordinary sub-meter condition while retaining
100% strict accuracy. In the clean weak condition it reduced median rotation
error from 0.096 to 0.054 degrees and increased strict accuracy from 10% to
20%, with two gains and no losses. The paired exact McNemar p-value is 0.5 at
this sample size, so this is directional evidence rather than a confirmed
effect.

With 5% outliers, count-first refit reduced the aggregate median translation
error from 54.603 to 39.355 degrees and strict accuracy rose from 5% to 10%,
but the paired result contained two gains and one loss. It therefore violates
the zero-regression promotion criterion. The P95 translation error also
remained above 146 degrees, so the refit does not solve the long tail.

## Convergence and cost

The 100-step cap was intentionally loose. Well-conditioned cases required at
most four recorded refits and ordinary sub-meter cases at most seven. Weak and
unobservable cases commonly required tens of steps: medians ranged from 14 to
24 and maxima from 77 to 93. One count-first unobservable case reached the
100-step cap. No observable candidate reached the cap.

For count-first, median runtime rose from 1554 to 2095 ms in the clean weak
condition and from 1291 to 1917 ms with outliers. Runtime P95 is noisy because
the complete estimator also performs bounded nonlinear optimization, but the
refit overhead is material in the target regimes.

## Interpretation

The experiment separates two facts:

1. residual-first scoring can substantially improve the estimated rotation;
2. choosing the pose only by an epipolar residual objective can simultaneously
   worsen translation direction.

This is consistent with the earlier diagnosis: low-parallax correspondences
constrain rotation strongly but contain little information about translation.
A single scalar Essential score cannot reliably balance those objectives.

The next experiment should retain a separately estimated rotation candidate
and use a conditional two-stage policy: select or regularize `R` using robust
rotation evidence, then estimate `t` from correspondences with independently
supported translational information. The all-inlier refit can remain one
candidate in that pipeline, but should not become the default by itself.
