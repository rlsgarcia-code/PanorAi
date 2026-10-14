# Recover metric scale and triangulate

**Level:** Compose · **Input:** accepted pose direction and bearing matches

Calibrated two-view geometry cannot recover translation magnitude by itself.
Choose and record a metric source before calling the result metric:

- a calibrated physical baseline;
- surveyed camera centres;
- a known camera/tripod height with an observable floor;
- another independently measured scene constraint.

If `t = s * t_direction`, every triangulated point and stereo range scales by
the same `s`. A unit translation therefore produces baseline-normalized, not
metric, structure.

Use only accepted matches with adequate parallax. Retain angular reprojection
residual, cheirality and conditioning with every sparse point; these refined
tracks are the useful bridge to multiview reconstruction and semantic
association.

## Scale propagation

Let the estimator return $\hat{t}$ with unit norm and let an external
measurement provide baseline $s$. Then $t=s\hat{t}$. Triangulation solves
depths $\rho_a,\rho_b$ such that

$$\rho_b b_b = R(\rho_a b_a) + t.$$

Multiplying $s$ by a factor multiplies both depths and every reconstructed
coordinate by the same factor. A later algorithm cannot infer the missing
unit from those points alone.

## Sparse-point acceptance

For every triangulated correspondence, require positive depth in both views,
sufficient parallax, a finite condition measure and a bounded angular
reprojection residual. Keep rejected tracks with their reason when auditing a
pipeline, but do not use them as precise anchors for semantic association.

A known-height floor constraint can estimate scale only when the floor model,
camera height and corresponding bearings are independently justified. It is
not a generic substitute for a measured baseline.

See the scale, floor-intersection and triangulation stages in
[two-view geometry](../tutorials/04_two_view_geometry.md). Then continue to
[dense spherical stereo](dense_stereo.md).
