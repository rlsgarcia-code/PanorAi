# Tutorial: metric spherical BA at matched landmarks

This tutorial adds a deliberately narrow **Experimental** workflow for refining
sparse 3D landmarks observed in overlapping central panoramas. It consumes unit
bearings, metric camera poses, initial 3D landmarks, and optional monocular
radial-range priors. It returns refined cameras and landmarks.

It does **not** generate a dense depth map. Accuracy measured at matched
landmarks must not be reported as accuracy over the panorama or reconstructed
surface.

The interface is
`panorai-metric-spherical-landmark-ba/v1-experimental`.

**PanorAi-specific:** all reprojection errors are spherical log-map residuals
in the tangent plane of the measured bearing; depth means radial range; pose is
`x_camera = R_world_to_camera @ (X_world - C_world)`; metric scale is an
explicit baseline gauge; landmark blocks are eliminated with a Schur
complement.

## 1. What must already be known

For every accepted landmark, provide:

- observations from at least two different panoramas;
- one unit bearing per observing panorama;
- an initial finite 3D position in the shared world frame;
- camera rotations and centres in the same frame;
- a known metric baseline between two cameras;
- optionally, a valid monocular **radial range** and uncertainty.

PanorAi's feature and pose workflow can supply matched bearings and relative
orientation. Two-view monocular pose supplies only a translation direction,
not metric scale. The baseline must come from registration, a calibrated rig,
a surveyed distance, or another documented metric constraint.

Sample the monocular map only where its separate validity mask is true. Do not
use zero as a missing-depth sentinel and do not reinterpret radial range as
camera z-depth.

## 2. Run landmark-only refinement

The executable documentation runner contains a complete two-camera example:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:start-after: DOCS_METRIC_LANDMARK_BA_START = None
:end-before: DOCS_METRIC_LANDMARK_BA_END = None
```

Passing both camera IDs in `fixed_camera_ids` optimizes only the landmark
positions. To refine one uncertain camera together with the landmarks, fix the
reference camera only. In either case, the baseline gauge remains explicit and
the returned `support` is `"supplied-landmarks-only"`.

Inputs are copied. Result arrays are immutable. An observation that names an
unknown endpoint, a duplicate camera–landmark pair, a landmark seen from fewer
than two cameras, or an underconstrained gauge fails explicitly.

## 3. Objective and factorization

For camera (i), landmark (j), measured bearing (b_{ij}), rotation (R_i),
and camera centre (C_i), the predicted bearing is

\[
\hat b_{ij} =
\frac{R_i(X_j-C_i)}{\lVert R_i(X_j-C_i)\rVert}.
\]

The two-dimensional observation residual is the spherical log map in the
tangent plane of (b_{ij}):

\[
r_{ij}=\log_{b_{ij}}(\hat b_{ij}).
\]

An optional monocular prior at a supported landmark is

\[
r^{\mathrm{range}}_j =
\frac{\log(\lVert X_j-C_k\rVert/d^0_j)}{\sigma_j},
\]

where (d^0_j) is radial range from camera (k). Both angular and range terms
use robust Huber weights. The scale gauge adds the baseline residual

\[
r^{\mathrm{scale}} =
\frac{\lVert C_b-C_a\rVert-B}{\sigma_B}.
\]

The LM linearization has camera blocks (H_{cc}), independent landmark blocks
(H_{jj}), and camera–landmark blocks (H_{cj}). PanorAi eliminates each
landmark block before solving the camera step:

\[
S = H_{cc} - \sum_j H_{cj}H_{jj}^{-1}H_{jc}.
\]

With all cameras fixed, the camera system is empty and only the independent
landmark updates remain. The implementation uses analytic observation
Jacobians; tests compare them with central finite differences and include a
near-antipodal case so a (179^\circ) error cannot collapse toward zero.

## 4. What the P74 experiment demonstrated

The motivating development experiment was VAL-042 on the P74 G046–G047 pair.
The frontend ran without resizing on a `1024×2048` detector lattice and found
95 matches, 89 robust pose inliers, and 69 promoted landmarks. The sparse BA
used 138 bearing observations, 69 soft log-range priors, the first camera fixed,
and a `1.279521 m` baseline gauge.

| quantity | before | after |
| --- | ---: | ---: |
| mean angular error | 0.10774° | 0.09843° |
| registered rotation error | 0.41528° | 0.40966° |
| held-out median epipolar error | 0.16606° | 0.14584° |
| AbsRel **at the 69 landmark locations** | 0.80721 | 0.11585 |
| δ1 **at the 69 landmark locations** | — | 0.86765 |

The strong `0.11585` AbsRel is therefore sparse supported-landmark evidence.
It is not full-image AbsRel. The experiment jointly refined the second camera
and the landmarks; “landmarks only” describes where depth accuracy was
evaluated, not all optimization variables.

The subsequent dense propagation did not inherit this result: it improved
full-map AbsRel only from `0.758509` to `0.757831`, worsened mean normal error,
and worsened an independent held-out feature cost. It was rejected. PanorAi
therefore exposes the sparse result without claiming dense reconstruction.

P74 images, model weights, predictions, PLY files, and result arrays are not
distributed with PanorAi. These numbers are development source-checkout
evidence from commit `e01f9e83cf7ca27ef4fbf72e45f58bf49d95d744`, not an
installed-wheel accuracy guarantee.

## 5. Recommended use

1. Extract spherical features and estimate or load reliable camera poses.
2. Keep only geometrically verified matches with adequate parallax.
3. Triangulate finite initial landmarks in the documented world frame.
4. Sample the monocular radial map only at those matches and retain a separate
   validity mask and uncertainty.
5. Start with all poses fixed. Unlock a camera only when the baseline and
   correspondence support are independently trustworthy.
6. Inspect angular residuals and compare the sparse points with an independent
   reference before using them as anchors for another method.
7. Treat densification as a separate algorithm with its own held-out test.

## Limitations

- The API assumes calibrated central panoramas and fixed intrinsics.
- It does not detect features, estimate scale, discover tracks, handle dynamic
  objects, or infer occlusion.
- A monocular prior may be biased and is only a robust soft constraint.
- The P74 result covers one high-overlap pair and 69 landmarks.
- No three-view P74 clique with at least 80% overlap was available in that
  experiment.
- Promotion beyond Experimental requires prospective multi-scene validation,
  installed-wheel evidence, broader outlier/degeneracy coverage, and an
  independent consumer.
