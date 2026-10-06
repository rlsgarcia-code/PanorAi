# GEO-017 conclusions: fixed-R translation refit is rejected

## Decision

Do **not** integrate the tested fixed-rotation translation refit or its proposed
observability gate. Preserve the GEO-016 recommendation:

```python
RelativePoseOptions(
    hypothesis_ranking="msac-first",
    nonminimal_refit_max_steps=100,
)
```

The new method produced a small development improvement, but failed the frozen
held-out test. A subsequent post-hoc safeguard that applied a candidate only
when it improved the global MSAC ranking preserved the baseline almost
exactly and supplied no material gain; it is not a solution either.

## Frozen held-out result

The protocol selected 12 pairs from six windows that were disjoint from every
GEO-016 frame. Ten pairs returned poses under both methods; two low-support
pairs with 12 and 18 matches returned no pose under either method.

| Metric | MSAC + E refit | Fixed-R t refit + gate | Change |
| --- | ---: | ---: | ---: |
| Returned | 10/12 | 10/12 | unchanged |
| R mean | 4.114 deg | 4.114 deg | exactly preserved |
| R median | 0.860 deg | 0.860 deg | exactly preserved |
| t mean | 29.439 deg | 29.796 deg | 1.2% worse |
| t median | 5.425 deg | 5.981 deg | 10.3% worse |
| t P95 | 142.098 deg | 143.763 deg | worse |
| Strict (`R<5`, `t<10`) | 8/12 | 8/12 | unchanged |
| High precision (`R<1`, `t<5`) | 4/12 | 3/12 | one loss |
| Accepted coverage | 8/12 | 7/12 | one loss |
| Accepted strict precision | 8/8 | 7/7 | both 100% |
| False accepts | 0 | 0 | unchanged |
| Median runtime | 2866.6 ms | 2336.0 ms | order/cache confounded |

On common returns, the candidate improved `t` in five pairs and worsened it in
five. It passed only three of six preregistered criteria: rotation identity,
selective precision/false-accept preservation, and the non-regression runtime
bound. It failed median `t`, P95/paired wins, and high-precision non-regression.

The candidate rejected one otherwise accepted strict-correct pair. In the
0.068 m-baseline case, `t` worsened from 4.07 to 7.55 degrees despite a stable
bootstrap report. Conversely, a gross 142-degree translation failure was
correctly rejected by both methods. This shows that subset repeatability is
not equivalent to accuracy under systematic model bias.

## Development and post-hoc findings

With seed and search budget matched to GEO-016, the unrestricted refit had
improved `t` in eight of 11 returned development pairs and worsened three;
median and P95 improved by 9.4% and 8.9%. It did not remove the known false
accept (`t=13.39 -> 12.25 deg`). That promising but already opened result did
not generalize.

After held-out evaluation, a safeguard was tested only as development: apply
the fixed-R candidate if it also improves the estimator's global ranking.
Eight of ten candidates were suppressed, and aggregate pose metrics became
numerically the same as the baseline. The safeguard is safe but ineffective
and was not subjected to another held-out claim.

## Interpretation

The experiment refutes the assumption that a separate fixed-R nullspace fit is
a general improvement over the existing joint Essential refinement. The joint
optimizer has already used the same epipolar information; refitting `t` from
the same correspondences cannot correct systematic rolling-shutter,
calibration, scene-motion, or rotation-model bias. Bootstrap stability can be
confidently wrong when every subset shares that bias.

The next justified direction is to add independent information rather than
reweight the same two-view equations again. The leading candidates are:

1. adaptive temporal/keyframe spacing to increase measurable translation
   parallax before estimating E;
2. three-view translation-direction consistency, which can reject a pairwise
   direction that does not compose across a short track;
3. an external rotation prior (for example IMU) when the application provides
   one, exposed explicitly rather than silently coupled into the central
   two-view estimator.

## Integrity

- Protocol SHA-256:
  `b2f61a8cc2f2a87a305fbb046c3a49b6de84bd21c3906ca9ce4d9194187f97d1`
- Frozen prediction SHA-256:
  `2e68b53fe8f0946b860f3c310f41d13fdca976a41afa4585764b26278f4ee99c`
- Raw held-out CSV SHA-256:
  `4a9c167586d64874556e42da61381599f2a99a93d8c0286c8f5c99965e39888c`
- Held-out summary SHA-256:
  `307e3004736d4bae32909267ee14793d2b366e8d28b1966078238c69d74c91ca`

The prediction records the frozen implementation and benchmark hashes. It was
read-only before evaluation. Dataset images, calibration, trajectory,
descriptors and ROS records remain external and ignored.
