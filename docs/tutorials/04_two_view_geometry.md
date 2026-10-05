# Tutorial: spherical Essential matrix and two-view triangulation

Two central panoramas observe the same 3D scene from different centers. The
previous tutorial produced matched unit bearings $(\mathbf b_1,\mathbf b_2)$.
This tutorial turns them into relative rotation, translation direction, and
arbitrary-scale 3D points.

`panorai.estimators` is Experimental. It is useful research software with
explicit diagnostics, not yet part of the Stable 3.x contract.

![Two spherical cameras, Essential geometry, and triangulation](../_static/tutorials/two-view-geometry.svg)

## 1. The spherical epipolar constraint

PanorAi defines relative pose by

$$
\mathbf x_2 = R_{21}\mathbf x_1 + \mathbf t_{21}.
$$

For a correct correspondence, the two bearings and the baseline lie in one
epipolar plane:

$$
\mathbf b_2^T E\mathbf b_1 = 0,
\qquad E=[\mathbf t_{21}]_\times R_{21}.
$$

The input is a pair of 3D unit bearings in the two panorama frames—not a pair
of ERP pixels and not a mixture of pixels from different virtual cameras. This
removes the planar camera matrix from the residual while retaining the same
calibrated Essential geometry.

## 2. Estimate relative pose

```python
from panorai.estimators import RelativePoseOptions, SphericalRelativePoseEstimator

estimator = SphericalRelativePoseEstimator(
    RelativePoseOptions(
        max_angular_error_deg=0.5,
        random_seed=7,
    )
)
pose = estimator.estimate(matches.to_bearing_correspondences())
if pose is None or not pose.quality_report.accepted:
    raise RuntimeError("relative pose was not trustworthy")

R_21 = pose.R
t_21_direction = pose.t
inliers = pose.inlier_mask
```

The estimator builds five-correspondence Essential hypotheses, uses a
spatially weighted but non-filtering RANSAC sampler, scores spherical
tangent-Sampson error, refines rotation and translation direction, and selects
the cheirally valid decomposition. A private C++ kernel may accelerate the
same solver and residuals; Python retains policy, diagnostics, and fallback.

The CI-sized contract check is:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:start-after: DOCS_RELATIVE_POSE_START = None
:end-before: DOCS_RELATIVE_POSE_END = None
:dedent: 4
```

Always inspect `quality_report.accepted`, `rejection_reasons`, angular
coverage, residual percentiles, cheirality, parallax, stability, and competing
rotation/projective models. A returned pose may intentionally carry a rejected
quality report so experiments can audit it.

## 3. What is PanorAi and what is OpenCV here?

| Stage | PanorAi route | OpenCV route |
| --- | --- | --- |
| local features and descriptor neighbors | PanorAi façade, OpenCV implementation | `SIFT`, `ORB`, `AKAZE`, `BFMatcher`, `FlannBasedMatcher` |
| coordinates entering geometry | global panorama-frame unit bearings | normalized pinhole pixels/bearings chosen by caller |
| Essential hypothesis and robust policy | PanorAi five-point + spherical scoring | `findEssentialMat` for a calibrated planar-camera model |
| pose decomposition | PanorAi cheirality and quality report | `recoverPose` |
| public result | PanorAi `RelativePoseResult` + diagnostics | OpenCV arrays and masks |

Both use calibrated Essential geometry. The PanorAi estimator is not a wrapper
around `cv2.findEssentialMat`, and OpenCV feature extraction does not make the
geometry itself OpenCV-owned. For the established planar APIs, consult
[OpenCV Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
The five-point foundation is [Nistér, 2004](https://doi.org/10.1109/TPAMI.2004.17).

## 4. Triangulate one inlier pair at arbitrary scale

Monocular relative pose observes only the direction of translation. PanorAi
normalizes `pose.t`; choosing that unit baseline fixes an arbitrary scale.
Camera B's center in frame A is

$$
\mathbf C_2^{(1)}=-R_{21}^T\mathbf t_{21},
$$

and B's bearing expressed in frame A is
$\mathbf d_2^{(1)}=R_{21}^T\mathbf b_2$. Solve the closest points on

$$
\lambda\mathbf b_1
\quad\text{and}\quad
\mathbf C_2^{(1)}+\mu\mathbf d_2^{(1)},
$$

then use their midpoint. Positive $\lambda$ and $\mu$ enforce cheirality;
the angle between rays measures triangulation strength.

The following educational NumPy helper is executable, but is not a new public
PanorAi API:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:start-after: DOCS_TRIANGULATION_START = None
:end-before: DOCS_TRIANGULATION_END = None
:dedent: 4
```

For many correspondences, reject non-inliers first, then reject non-positive
depth, very small ray angle, large closest-ray gap, or excessive angular
reprojection error. Any recovered distance is expressed in the arbitrary unit
baseline unless an external metric observation supplies scale.

## 5. Add the known tripod-height premise

Suppose the user records, for each photo, the distance from the **camera's
optical center** to one common, locally planar floor. Call those positive
heights $h_A$ and $h_B$, in metres. A tripod-leg or tripod-head measurement
must first be corrected by the fixed offset to the optical center.

We also need the floor-up unit vector $\mathbf g_A$ in camera A's frame. It is
`[0, 1, 0]` only when the ERP has been levelled to PanorAi's `+Y`-up frame.
Otherwise obtain it from an IMU, a trusted levelling step, or a separately
estimated floor plane. The floor in camera A coordinates is

$$
\mathbf g_A^T\mathbf X_A+h_A=0.
$$

This premise creates several different observability cases:

| Available evidence | What becomes observable |
| --- | --- |
| spherical matches only | $R_{BA}$ and translation direction; structure has arbitrary scale |
| equal known heights, known up, no identified floor point | horizontal-baseline constraint and a useful consistency check; **scale is still unknown** |
| different known heights and known up | scale from the vertical baseline component, if that component is safely non-zero |
| at least one verified floor match plus a known height | metric baseline scale, including the common equal-height case |
| many floor matches or a floor mask | robust metric scale, metric floor points, and a plane-consistency diagnostic |

The equal-height distinction is essential. If both centers are 1.60 m above the
same floor, then their relative displacement has zero vertical component, but
its horizontal length could still be 0.5 m, 2 m, or 20 m. Height alone does not
choose between them.

![Known-height two-view geometry](../_static/tutorials/known-height-two-view.svg)

## 6. Intersect a floor ray in metres

For a bearing $\mathbf b_A$ known to hit the floor,

$$
\rho_A=-\frac{h_A}{\mathbf g_A^T\mathbf b_A},
\qquad
\mathbf X_A=\rho_A\mathbf b_A.
$$

The denominator must be negative. It approaches zero at the horizon, where a
small angular error creates a very large range error; practical code therefore
rejects a configurable horizon band. This operation already gives a metric 3D
floor point from one levelled central panorama.

The uncertainty in $h_A$ transfers linearly into $\rho_A$. Uncertainty in the
up vector and bearing grows nonlinearly near the horizon, so record the tripod
measurement tolerance and do not present distant floor intersections as exact.

## 7. Recover the two-camera metric scale

Write PanorAi's relative pose as

$$
\mathbf x_B=R_{BA}\mathbf x_A+s\widehat{\mathbf t}_{BA},
$$

where $\widehat{\mathbf t}_{BA}$ is the returned unit translation and $s>0$
is the unknown baseline length. For a verified floor correspondence, first
compute its metric $\mathbf X_A$ from camera A's height. Its prediction in B is

$$
\mathbf q_B+s\widehat{\mathbf t}_{BA},
\qquad \mathbf q_B=R_{BA}\mathbf X_A,
$$

and must be parallel to the measured bearing $\mathbf b_B$. With
$P_B=I-\mathbf b_B\mathbf b_B^T$, the one-dimensional least-squares solution
is

$$
s=-\frac{(P_B\widehat{\mathbf t}_{BA})^T(P_B\mathbf q_B)}
          {\lVert P_B\widehat{\mathbf t}_{BA}\rVert^2}.
$$

Use many spatially distributed floor matches, fit one shared positive $s$ with
a robust loss, and report its dispersion. A match whose denominator is small
does not constrain scale. Floor matches should establish scale **after** a
relative pose supported by non-coplanar scene evidence; using only one plane to
estimate the Essential geometry is a known degeneracy.

The executable educational implementation is intentionally not advertised as
a public PanorAi API:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:start-after: DOCS_METRIC_FLOOR_START = None
:end-before: DOCS_METRIC_FLOOR_END = None
:dedent: 4
```

There is a second scale cue when $h_A\ne h_B$. Let
$\widehat{\mathbf C}_B^{(A)}=-R_{BA}^T\widehat{\mathbf t}_{BA}$ be camera B's
center direction in A. Then

$$
s=\frac{h_B-h_A}
        {\mathbf g_A^T\widehat{\mathbf C}_B^{(A)}}.
$$

This is ill-conditioned when the cameras are nearly at the same height, so the
floor-correspondence estimate should remain the primary route for normal
tripod captures. The height-difference equation is better used as an
independent check or as evidence in a joint robust fit.

## 8. Recover scene structure in metres

After scale is accepted, replace the unit translation by
$\mathbf t_{BA}^{metric}=s\widehat{\mathbf t}_{BA}$ and triangulate every
non-floor inlier with the same closest-rays or angular-reprojection method.
The resulting points and camera baseline are now in metres.

Use the geometry differently by point type:

- **floor points:** prefer direct ray-plane intersection; it uses the known
  metric plane and avoids unnecessary two-ray noise;
- **general static points:** triangulate with the scaled pose, then check
  positive depth, parallax, ray gap and angular reprojection in both cameras;
- **object contact points:** localize the contact on the floor, then use
  triangulated upper points to estimate metric object height;
- **camera trajectory:** camera B's center in A is
  $-R_{BA}^T\mathbf t_{BA}^{metric}$; validate its vertical coordinate against
  $h_B-h_A$.

Do not silently convert all matches into floor constraints. Shadows, rugs,
stairs, ramps, reflections, moving objects and incorrect semantic masks can
produce a plausible but false scale.

## 9. Acceptance diagnostics for a metric result

A future metric helper should return a result only with diagnostics, including:

- the height, tolerance and optical-center offset used for each image;
- the up vector source and levelling residual;
- number and angular spread of candidate/accepted floor matches;
- per-match scale estimates, robust median, MAD and rejected outliers;
- minimum horizon margin and scale denominator;
- agreement between floor-derived scale and height-difference scale when both
  are conditioned;
- positive-depth rate, triangulation-angle distribution and angular
  reprojection percentiles for non-floor points;
- an explicit failure when the floor is not common/planar or metric scale is
  underconstrained.

This is the recommended route for evolving the library: first keep the
closed-form example and synthetic oracle executable; then add an Experimental
`FloorPlanePrior`/metric-scale result; only promote it after real captures with
measured baselines and varied equal/different tripod heights.

## 10. Relation to standard two-view geometry

The derivation uses the calibrated Essential matrix, noisy-ray triangulation,
and plane-induced homography results developed in Hartley and Zisserman,
*Multiple View Geometry in Computer Vision*, second edition, sections 9.6,
12.1--12.3 and 13.1. For spherical cameras, ERP pixels are first converted to
unit bearings, so the geometry already operates in calibrated normalized
coordinates. The plane-induced bearing map is the same
$H=R-\mathbf t\mathbf n^T/d$ relation, followed by direction normalization.

## 11. Two images are not yet a reconstruction system

Two-view triangulation cannot resolve weak parallax, repeated texture, dynamic
objects, a misidentified floor, or metric scale without a valid external
anchor. With three or more panoramas, tracks can corroborate points across
views and global optimization can distribute error. Known heights can then be
added as metric plane/center residuals instead of rescaling each pair
independently.
Continue to {doc}`05_multiview_reconstruction`.
