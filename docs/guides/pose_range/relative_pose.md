# Estimate relative pose

**Level:** Start · **Input:** matched unit bearings · **Output:** rotation and
translation direction · **Stability:** Experimental estimator

```python
from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator

estimator = SphericalRelativePoseEstimator(
    RelativePoseOptions(max_angular_error_deg=0.5, random_seed=7)
)
pose = estimator.estimate(matches.to_bearing_correspondences())
if pose is None or not pose.quality_report.accepted:
    raise RuntimeError("relative pose was not trustworthy")

R_b_from_a = pose.R
t_direction_b_from_a = pose.t
```

The translation has direction but no metric magnitude. Inspect inliers,
angular coverage, residual percentiles, parallax, cheirality, stability and
rejection reasons before continuing.

## Geometry and convention

For paired unit bearings $b_a$ and $b_b$, a valid central relative pose obeys

$$b_b^T [t]_{\times} R b_a = 0.$$

The returned transform maps camera-A coordinates into camera-B coordinates:
`X_b = R_b_from_a @ X_a + t_b_from_a`. The estimator normalizes `pose.t`, so
its magnitude is arbitrary. This convention must remain unchanged when the
pose is passed to triangulation, stereo or region transfer.

## Acceptance is part of the API

`pose.quality_report.accepted` is not decorative metadata. A high raw inlier
count can still be rejected for poor angular coverage, low parallax, unstable
translation, weak cheirality or a competitive rotation-only explanation.
Store the report beside `R` and `t`; otherwise downstream failures cannot be
distinguished from a bad pose.

## Common failure modes

- Repeated industrial structures create descriptor-consistent outliers.
- Pure rotation constrains $R$ but not a useful translation direction.
- Far structure produces small parallax and ill-conditioned triangulation.
- Bearings expressed in different camera-axis conventions invalidate the
  Essential relation even if ERP pixels appear aligned.

The Essential matrix, minimal solver, consensus and refinement are derived in
[spherical two-view geometry](../tutorials/04_two_view_geometry.md). Next:
[recover scale and triangulate](metric_scale_and_triangulation.md).
