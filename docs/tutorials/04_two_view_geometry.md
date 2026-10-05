# Tutorial: spherical Essential matrix and two-view triangulation

Two central panoramas observe the same 3D scene from different centers. The
previous tutorial produced matched unit bearings $(\mathbf b_1,\mathbf b_2)$.
This tutorial turns them into relative rotation, translation direction, and
arbitrary-scale 3D points.

See the {ref}`capability-map-two-view` theme in the spherical computer-vision
guide. Once features have become panorama-frame unit bearings, this tutorial
uses no image convolution, gnomonic raster, or backprojection to ERP.

**PanorAi-specific:** five-bearing hypotheses, spherical tangent-Sampson
scoring, panorama-frame pose conventions, cheirality selection and explicit
quality/degeneracy diagnostics.

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

## 2. The five-point kernel used by PanorAi

Five calibrated bearing correspondences give five homogeneous linear
equations in the nine entries of $E$. For correspondence $i$, PanorAi places

$$
\operatorname{vec}(\mathbf b_{2i}\mathbf b_{1i}^{T})^T
$$

in row $i$ of a $5\times9$ design matrix $A$. In the generic case,
$\operatorname{null}(A)$ has dimension four. If
$E_1,\ldots,E_4$ are its basis matrices, every linear solution is

$$
E=x_1E_1+x_2E_2+x_3E_3+x_4E_4.
$$

Most matrices in this nullspace are not Essential matrices. A calibrated
Essential matrix must also satisfy the Demazure cubic constraints

$$
2EE^TE-\operatorname{tr}(EE^T)E=0,
\qquad \det(E)=0.
$$

The implementation substitutes the four-dimensional nullspace into these nine
matrix equations plus the determinant equation, giving ten cubic polynomials.
For each of the four projective charts, one coefficient $x_k$ is fixed to one.
The remaining three variables use the 20-monomial ordering of all terms up to
degree three. Gaussian elimination expresses the ten cubic monomials through
the ten-monomial quotient basis

$$
(x^2,xy,y^2,xz,yz,z^2,x,y,z,1).
$$

Multiplication by $z$ in that quotient ring produces a $10\times10$ action
matrix. Its real eigensolutions recover $(x,y,z)$ and therefore candidate
Essential matrices. PanorAi tries all four charts so a valid solution whose
chosen constant coefficient is zero is not silently discarded. Roots are
checked against the original epipolar and calibrated-E constraints, normalized
to unit Frobenius norm, and deduplicated up to the unavoidable $E\sim-E$
projective sign. If every polynomial chart is singular, a separately named
deterministic nonlinear root search is used as a fallback.

```text
FIVE_POINT(b1[1:5], b2[1:5]):
    A[i] <- vec(b2[i] b1[i]^T)^T
    N <- four-dimensional right nullspace of A
    candidates <- empty
    for constant coefficient k in {1,2,3,4}:
        substitute E = sum_j x[j] N[j], with x[k] = 1
        C <- coefficients of 9 Demazure cubics and det(E)
        if the cubic elimination block of C is well-conditioned:
            Mz <- 10x10 action matrix for multiplication by z
            for each real eigensolution of Mz:
                reconstruct E
                retain E only if the original constraints pass
                deduplicate E and -E
    if candidates is empty:
        run the declared numerical chart fallback
    return candidates
```

This is the calibrated five-point formulation introduced by
[Nistér (2004)](https://doi.org/10.1109/TPAMI.2004.17), implemented here with
an explicit nullspace/action-matrix construction. See also
[Hartley and Zisserman, *Multiple View Geometry*, second edition](https://www.robots.ox.ac.uk/~vgg/hzbook/)
and [Szeliski, *Computer Vision: Algorithms and Applications*, second
edition](https://szeliski.org/Book/). The source implementation is
`panorai/estimators/_five_point.py`.

Hartley image-point normalization is not applied at this stage. Its purpose is
to condition homogeneous pixel coordinates in planar eight-point estimation.
Here the inputs are already calibrated, unit-length 3D bearings. PanorAi
instead validates bearing norms, distributes minimal samples over the sphere,
checks the condition of $A$, tries four coefficient charts, and validates every
root in the original equations.

## 3. Robust estimation around the minimal solver

The five-point kernel can return several algebraically valid roots, and any
five matches can include an outlier. The complete estimator therefore wraps it
in a spatially sampled, locally optimized robust loop:

```text
SPHERICAL_RELATIVE_POSE(all bearing pairs):
    normalize valid bearings to unit length
    repeat the conservative RANSAC trial budget:
        draw five spatially diverse pairs without removing any scoring pair
        for E in FIVE_POINT(the five pairs):
            score tangent-Sampson residuals on every valid pair
            decompose E into (R1,+t), (R1,-t), (R2,+t), (R2,-t)
            choose the translation orientation using parallax-weighted cheirality
            form the provisional inlier set
            locally refine R and unit t with scale-marginal robust weights
            retain the best full-set hypothesis
    refine the winning consensus nonlinearly
    report residual, coverage, parallax, cheirality, stability,
        competing-model, and translation-orientation evidence
    return R and unit t; never claim translation scale
```

The nonlinear variables are three local rotation coordinates and two tangent
coordinates on the unit sphere of translation directions. The objective is a
robustly weighted signed spherical tangent-Sampson residual. It is a
first-order epipolar objective rather than full two-view bundle adjustment;
the quality report therefore remains part of the result contract.

### 3.1 Residual, robust score, and inlier update

For $q_i=\mathbf b_{2i}^{T}E\mathbf b_{1i}$ and the tangent projector
$P_{\mathbf b}=I-\mathbf b\mathbf b^T$, the signed residual used by the
optimizer is

$$
r_i(E)=
\frac{q_i}
{\sqrt{
\lVert P_{\mathbf b_{1i}}E^T\mathbf b_{2i}\rVert^2+
\lVert P_{\mathbf b_{2i}}E\mathbf b_{1i}\rVert^2
}}.
$$

It is approximately an angular error in radians. A pair is provisionally
consistent when $|r_i|\leq\tau$, where `max_angular_error_deg` supplies
$\tau$. Cheirality is tested separately; the public inlier mask is the
intersection of the residual test, explicit input validity, and positive
depth under the selected pose.

PanorAi does not optimize a single hard-threshold count. For scales
$s_k$ linearly distributed from
`scale_marginal_min_fraction * tau` through $\tau$, it computes

$$
\omega_i=
\frac{1}{K}\sum_{k=1}^{K}
\left[\max\!\left(1-\left(\frac{|r_i|}{s_k}\right)^2,0\right)\right]^2.
$$

The robust hypothesis score is $\sum_i\omega_i$ over valid cheiral pairs.
Hypotheses are ordered lexicographically by:

1. larger inlier count;
2. larger scale-marginal robust score;
3. smaller sum of inlier residuals.

Consequently, RANSAC selects a candidate using all available correspondences,
not only the five rays that generated it. The dynamic trial bound is used only
when the injected sampler satisfies the assumptions of uniform sampling.

### 3.2 Nonlinear refinement of rotation and translation direction

Starting from $(R_0,\mathbf t_0)$, the optimizer uses a five-dimensional local
parameter $\delta=(\delta\boldsymbol\omega,\delta\mathbf u)$:

$$
R(\delta)=\exp([\delta\boldsymbol\omega]_\times)R_0,
\qquad
\mathbf t(\delta)=
\frac{\mathbf t_0+B_{\mathbf t_0}\delta\mathbf u}
{\lVert\mathbf t_0+B_{\mathbf t_0}\delta\mathbf u\rVert},
$$

where the two columns of $B_{\mathbf t_0}$ span the tangent plane of the unit
sphere at $\mathbf t_0$. Translation therefore retains unit norm and no metric
scale variable is introduced. With robust weights frozen during one inner
least-squares solve, the minimized objective is

$$
\min_{\delta\in\mathbb R^5}
\sum_{i\in\mathcal I}\omega_i\,
r_i\!\left([\mathbf t(\delta)]_\times R(\delta)\right)^2.
$$

After an inner solve, the pose is rescored on the full valid set and accepted
only if it improves the lexicographic rule above. Robust weights are then
recomputed for the configured number of IRLS steps. Local optimization stops
when a step does not improve the hypothesis or when
`local_optimization_steps` is exhausted; one final refinement is attempted on
the winning consensus. This is a bounded LO-RANSAC/IRLS procedure, not an
unbounded loop waiting for the inlier count to stabilize.

The optimizer's internal SciPy scalar `cost` is not exposed as a public field.
The inspectable result instead reports residual median/P90 and the normalized
robust evidence under `quality_report.model_competition.essential`, together
with inlier count, parallax, cheirality, stability, and rejection reasons.

## 4. Estimate relative pose

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
the cheirally valid decomposition. The default orientation policy discounts
depth signs supported only by nearly parallel rays; the historical raw-count
policy remains available for reproducible A/B studies:

```python
baseline_options = RelativePoseOptions(
    translation_orientation_method="positive-depth-count"
)
weighted_options = RelativePoseOptions(
    translation_orientation_method="parallax-weighted",
    translation_orientation_parallax_scale_deg=1.0,
)
```

A private C++ kernel may accelerate the
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

## 5. What is PanorAi and what is OpenCV here?

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
The feature backend does not alter the calibrated bearing equations above.

## 6. Resolve the orientation of translation

SVD of an Essential matrix produces two rotations and one translation axis,
which means four pose hypotheses. Epipolar residuals cannot distinguish $t$
from $-t$ because $E$ and $-E$ describe the same epipolar planes. The missing
orientation comes from triangulation and positive radial depth.

Concretely, after projecting $E$ to singular values $(s,s,0)$, let
$E=U\operatorname{diag}(s,s,0)V^T$ and

$$
W=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}.
$$

The candidates are $R_1=UWV^T$, $R_2=UW^TV^T$, and
$\mathbf t=\pm U_{:,3}$, with determinant corrections ensuring
$R\in SO(3)$. The epipolar equation alone cannot choose among these four
poses; the following cheirality evidence performs that selection.

The historical policy gave every correspondence one binary vote. That is
fragile when $\theta_i$, the angle between the two triangulation rays, is close
to zero: depth uncertainty grows approximately as $1/\sin\theta_i$, so tiny
angular noise can reverse the depth sign of a distant point. The default policy
now uses the bounded weight

$$
w_i=\frac{\sin^2\theta_i}
          {\sin^2\theta_i+\sin^2\theta_0},
\qquad \theta_0=1^\circ,
$$

and selects the decomposition with the largest
$\sum_i w_i\,\mathbf1[\lambda_i>0\land\mu_i>0]$. Each reliable point is still
capped at one vote, while an almost parallel pair contributes almost zero.
If fewer than the five correspondences of a minimal calibrated solution reach
$w_i\geq0.5$, PanorAi keeps the historical axis representative for
compatibility but reports `positive-depth-count-fallback`, forces the
orientation margin to zero, and therefore makes the quality policy abstain.
`translation_orientation` records both weighted support and the historical raw
counts, their separate margins, the scale $\theta_0$, and the selected method.

Weighting cannot create information. Pure rotation, uniformly distant scenes,
or a baseline much smaller than scene depth must still be rejected through
low parallax, competing rotation-only evidence, an ambiguous orientation
margin, or unstable re-estimation. In those cases the axis $\{t,-t\}$ may be
meaningful even when its orientation is not.

The reproducible comparison and raw samples live under
`benchmarks/translation_orientation/`. It includes an isolated decomposition
experiment and an end-to-end synthetic run on identical seeds. It is
development evidence, not a replacement for a new outcome-blind VAL-002
real-image replay.

For an article-style, self-contained description of the estimator, see
`benchmarks/translation_orientation/ALGORITHM_PAPER.md`. The numerical tables,
environment, hashes, and measured comparison remain in
`results/TECHNICAL_REPORT.md`.

## 7. Triangulate one inlier pair

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

## 8. Two images are not yet a reconstruction system

Two-view triangulation cannot resolve weak parallax, repeated texture, dynamic
objects, or metric scale by itself. With three or more panoramas, tracks can
corroborate points across views and global optimization can distribute error.
Continue to {doc}`05_multiview_reconstruction`.
