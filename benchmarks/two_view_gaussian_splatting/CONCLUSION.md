# Conclusion: sparse-view spherical Gaussian surfaces

## Question

Can two or three posed panoramas, a monocular metric-depth prior and sparse
bundle-adjusted landmarks produce a metrically useful and photorealistic novel
view by optimizing a Gaussian representation on Apple MPS?

The experiments support a narrower conclusion: Gaussian splatting is useful as
an auditable visible-surface representation and renderer, but the tested
Gaussian optimizations do not replace missing multiview geometry. The promoted
path keeps poses fixed, uses one depth surface per observed panorama, orders the
surfaces by distance to the requested view and composites their Gaussian
footprints front to back with alpha. It does not create unseen surfaces.

## Frozen promoted configuration

- Training surfaces: G046, G047 and G049.
- Pose: VAL-042 refined pose, with the registered P74 poses for other views.
- Metric scale: 69 bundle-adjusted landmarks.
- Render size: 1024x2048 ERP.
- Footprint: `sigma=1.3`, radius 4.
- Compositing: depth-binned front-to-back alpha.
- Implementation baseline: commit `d009f20`.

## Held-out compositing gate

The same frozen three surfaces were rendered into three excluded cameras. The
table compares only the compositing rule; camera poses, depths, colors and
Gaussian footprints are identical.

| Held-out view | Normalized coverage | Alpha coverage | Normalized RGB L1 | Alpha RGB L1 | Relative L1 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| G045 | 72.86% | 75.61% | 0.24744 | 0.22211 | -10.24% |
| G048 | 81.86% | 82.38% | 0.13711 | 0.13311 | -2.92% |
| G050 | 69.87% | 73.65% | 0.25322 | 0.23155 | -8.56% |

The paired median improvement is 8.56% RGB L1 with a median coverage gain of
2.75 percentage points. Alpha compositing therefore passes the multiview gate.
The absolute quality on G045 and G050 remains poor because those cameras are
2.26-3.93 m from the available surfaces; this is extrapolation rather than a
small-baseline interpolation.

Development artifacts:

- alpha G045: `/private/tmp/panorai-val068-three-view-alpha-g045-1024x2048-mps/results.json`
- alpha G048: `/private/tmp/panorai-val066-three-view-alpha-wide13-g048-1024x2048-mps/results.json`
- alpha G050: `/private/tmp/panorai-val068-three-view-alpha-g050-1024x2048-mps/results.json`
- normalized controls: `/private/tmp/panorai-val071-normalized-g045-1024x2048-mps`,
  `/private/tmp/panorai-val071-normalized-g048-1024x2048-mps` and
  `/private/tmp/panorai-val071-normalized-g050-1024x2048-mps`

## Geometry-feedback gate

Gaussian-to-depth feedback improved the G046 depth map on registered-depth
evaluation: full-support AbsRel changed from 0.24453 to 0.19620 and delta-1
from 0.48955 to 0.61950. The 69 landmark pixels were enforced exactly.

Replacing only the frozen G046 surface with that refined depth did not improve
novel-view appearance on G048. Coverage changed from 82.38% to 82.61%, while
RGB L1 regressed from 0.13311 to 0.13594. The feedback depth is retained as a
depth-product experiment, but it is rejected as a surface replacement for the
promoted multiview renderer.

Development artifact:
`/private/tmp/panorai-val069-feedback-g046-three-view-g048-1024x2048-mps/results.json`.

## Gaussian-factor hierarchy gate

The final controlled hierarchy used the native Metric3D G046 prior and trained
on G046/G047 before opening G048 and registered G046 depth. Radial means,
covariance, opacity and view-independent color were unlocked in separate
stages.

| Candidate | Registered-depth AbsRel | Landmark AbsRel | G048 RGB L1 |
| --- | ---: | ---: | ---: |
| Landmark-aligned prior | 0.23168 | 0.20681 | 0.14878 |
| 16x32 radial | **0.21453** | 0.11178 | **0.14313** |
| 32x64 radial | 0.22106 | 0.06297 | 0.14554 |
| 64x128 radial | 0.23003 | **0.04105** | 0.14833 |
| + covariance | 0.23003 | 0.04105 | 0.14833 |
| + opacity | 0.23003 | 0.04105 | 0.14833 |
| + color | 0.23003 | 0.04105 | 0.14697 |

The coarse radial stage is the only promoted optimization stage. Finer radial
freedom continued to improve landmark fit but regressed registered dense depth
and the held-out view. Covariance and opacity were neutral. Color improved the
over-refined state but did not recover the coarse-stage held-out result.

Development artifact:
`/private/tmp/panorai-val070-full-factor-metric3d-g046-g047/results.json`.

## Final decision

Promote:

1. one landmark-scaled depth surface per observed panorama;
2. distance-ordered multiview fallback;
3. front-to-back alpha compositing;
4. the 1.3/radius-4 footprint at 1024x2048;
5. optional coarse `16x32` radial correction when it passes both registered
   depth and held-out appearance gates;
6. dense point-cloud export and Gaussian-to-depth feedback as downstream
   products, with their own metric gates.

Do not promote for sparse-view reconstruction:

- fine radial optimization selected by training loss alone;
- covariance, opacity or color as substitutes for incorrect geometry;
- supersampling or post-render hole filling that trades support for appearance;
- synthesized geometry in unobserved regions.

For two or three panoramas, the remaining photorealism limit is the accuracy
and completeness of the observed depth surfaces. Material improvement now
requires either additional nearby observations or a stronger multiview depth
estimator. Continuing to add Gaussian degrees of freedom to the same evidence
is not supported by the held-out results.
