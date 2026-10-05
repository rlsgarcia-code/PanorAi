# Robust spherical relative pose with parallax-aware translation orientation

## Abstract

This paper specifies the Experimental two-view relative-pose estimator used by
PanorAi for central spherical images. The method consumes calibrated unit
bearings, generates Essential matrices with a five-correspondence polynomial
solver, selects hypotheses with a spherical tangent-Sampson LO-RANSAC, and
refines three rotation and two unit-translation coordinates by robust
iteratively reweighted least squares. The four-fold Essential decomposition is
resolved with bounded parallax-weighted cheirality. This weighting improves the
orientation of translation when many distant, noise-dominated rays would
otherwise outvote a small informative subset. Translation magnitude remains
unobservable. The companion benchmark reports a 3.7 percentage-point gain in
the isolated weak-parallax condition and no end-to-end gain when the upstream
Essential estimate is already inaccurate.

## 1. Scope and coordinate convention

For matched bearings $\mathbf b_{1i},\mathbf b_{2i}\in S^2$, PanorAi estimates
the panorama-2-from-panorama-1 transform

$$
\mathbf x_2=R_{21}\mathbf x_1+\mathbf t_{21},
\qquad
E=[\mathbf t_{21}]_\times R_{21}.
$$

The output contains $R_{21}\in SO(3)$ and the unit direction
$\hat{\mathbf t}_{21}$. Neither the algorithm nor its API claims the magnitude
of translation from two monocular central views. Inputs are global panorama
bearings; ERP pixels are converted before entering this estimator.

## 2. Minimal Essential hypotheses

Each correspondence supplies
$\mathbf b_{2i}^{T}E\mathbf b_{1i}=0$. Five pairs form a $5\times9$ design
matrix whose generic nullspace is four-dimensional. Writing
$E=\sum_{j=1}^{4}x_jE_j$, the solver substitutes this basis into

$$
2EE^TE-\operatorname{tr}(EE^T)E=0,
\qquad \det(E)=0.
$$

For each of four projective charts, one $x_j$ is fixed to one. Elimination of
the cubic monomials produces a $10\times10$ action matrix; its real
eigensolutions reconstruct candidate Essential matrices. Candidates are
validated in the original equations, normalized to unit Frobenius norm, and
deduplicated up to $E\sim-E$. A deterministic numerical chart search is used
only when every polynomial chart is singular.

Unit bearings replace the planar normalized-image coordinates used in the
classical pinhole derivation. Hartley pixel normalization is therefore not
applied. Numerical protection comes from unit normalization, spatially diverse
minimal samples, design-matrix conditioning checks, four charts, and direct
root validation.

## 3. Spherical residual and robust consensus

Define $P_{\mathbf b}=I-\mathbf b\mathbf b^T$. The signed first-order angular
residual is

$$
r_i(E)=\frac{\mathbf b_{2i}^{T}E\mathbf b_{1i}}
{\sqrt{\lVert P_{\mathbf b_{1i}}E^T\mathbf b_{2i}\rVert^2+
       \lVert P_{\mathbf b_{2i}}E\mathbf b_{1i}\rVert^2}}.
$$

It is thresholded in radians. For marginal scales $s_k$ between a declared
fraction of the threshold $\tau$ and $\tau$, PanorAi assigns

$$
\omega_i=\frac1K\sum_k
\left[\max\left(1-(|r_i|/s_k)^2,0\right)\right]^2.
$$

A spatial sampler proposes five-pair sets but never removes pairs from full
hypothesis scoring. Each Essential root is decomposed, oriented, intersected
with positive-depth support, and ranked by inlier count, then robust score
$\sum_i\omega_i$, then residual sum. A newly leading hypothesis undergoes a
bounded local optimization. Adaptive trial reduction is enabled only for
samplers compatible with the uniform RANSAC probability model.

## 4. Recovering and refining (R,t)

For projected $E=U\operatorname{diag}(s,s,0)V^T$ and

$$
W=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix},
$$

the four candidates are $(UWV^T,\pm U_{:,3})$ and
$(UW^TV^T,\pm U_{:,3})$, after determinant corrections. For each candidate,
linear two-ray triangulation yields depths $(\lambda_i,\mu_i)$.

The nonlinear state has five local coordinates. From an initial
$(R_0,\mathbf t_0)$,

$$
R(\delta)=\exp([\delta\boldsymbol\omega]_\times)R_0,
\qquad
\mathbf t(\delta)=
\frac{\mathbf t_0+B_{\mathbf t_0}\delta\mathbf u}
{\lVert\mathbf t_0+B_{\mathbf t_0}\delta\mathbf u\rVert},
$$

where $B_{\mathbf t_0}\in\mathbb R^{3\times2}$ spans the tangent plane at
$\mathbf t_0$. With IRLS weights held fixed during an inner solve, the method
minimizes

$$
\min_{\delta\in\mathbb R^5}
\sum_{i\in\mathcal I}\omega_i
r_i([\mathbf t(\delta)]_\times R(\delta))^2.
$$

The refined pose is rescored against every valid correspondence. It replaces
the current model only under the same lexicographic rule used by RANSAC.
Weights are recomputed for a bounded number of robust steps; local optimization
terminates immediately when no improvement occurs. One final refinement is
attempted after RANSAC. This is an epipolar refinement, not bundle adjustment:
3D points are not optimization variables.

## 5. Parallax-weighted orientation of translation

Raw positive-depth counts are unreliable when a ray pair is almost parallel.
For triangulation angle $\theta_i$, the current policy uses

$$
w_i=\frac{\sin^2\theta_i}
{\sin^2\theta_i+\sin^2\theta_0},
\qquad \theta_0=1^\circ,
$$

and maximizes
$\sum_iw_i\mathbf1[\lambda_i>0\land\mu_i>0]$ over the four decompositions.
The weight approaches zero with vanishing parallax, reaches one half at
$\theta_0$, and is capped by one. At least five rays must reach $w_i\geq0.5$;
otherwise PanorAi retains the historical representative only for inspection,
sets the orientation margin to zero, and makes the quality policy abstain.

The historical unweighted count remains selectable for paired experiments.
This policy changes the orientation between $+t$ and $-t$ for a fixed
Essential matrix; it cannot repair an incorrect translation axis already
encoded by $E$.

## 6. Complete algorithm

```text
ROBUST_SPHERICAL_RELATIVE_POSE(paired bearings):
    normalize explicitly valid bearings
    best <- none
    while within the conservative RANSAC budget:
        sample five spatially diverse pairs
        roots <- FIVE_POINT(sample)
        for E in roots:
            compute tangent-Sampson residuals on every valid pair
            form the provisional threshold consensus
            decompose E into four (R,t) candidates
            select orientation by parallax-weighted cheirality
            intersect residual consensus with positive-depth support
            compute inlier count, scale-marginal score, residual sum
            if the model leads:
                run bounded five-parameter LO/IRLS refinement
                rescore on every valid pair
                update best only if the lexicographic score improves
    attempt one final refinement of best
    estimate residual, coverage, parallax, orientation, stability,
        rotation-only, and spherical-homography evidence
    apply the acceptance policy
    return R, unit t, mask, residuals, and quality evidence
```

## 7. Reported cost and acceptance evidence

The internal nonlinear solver's scalar cost is intentionally not a public API.
Users can inspect:

- angular residual median and P90;
- inlier count and ratio;
- normalized scale-marginal Essential score;
- spherical coverage in both views;
- median parallax and cheirality ratio;
- weighted and raw translation-orientation margins;
- subset re-estimation stability;
- competition with rotation-only and spherical-homography models;
- the final acceptance decision and explicit rejection reasons.

A returned pose is not necessarily accepted. Low parallax, ambiguous sign,
instability, weak spatial coverage, or a competitive simpler model can retain a
diagnostic estimate while preventing it from being presented as trustworthy.

## 8. Evaluation and limitations

The companion
[`results/TECHNICAL_REPORT.md`](results/TECHNICAL_REPORT.md) compares the
historical and parallax-weighted orientation policies on identical seeded
synthetic scenes. In the isolated weak-parallax stress condition, orientation
accuracy increased from 96.0% to 99.7%, with 37 paired gains and no losses.
The end-to-end experiment showed no paired gain because the upstream Essential
axis was already inaccurate in that regime.

The evaluation is not evidence for real-world prevalence or calibrated
acceptance probabilities. It excludes feature-domain shift, dynamic scenes,
rolling shutter, non-central capture, and metric-scale recovery. A frozen,
outcome-blind real-pair replay is still required before broader accuracy claims.

## References

1. D. Nistér, “An Efficient Solution to the Five-Point Relative Pose Problem,”
   *IEEE TPAMI*, 2004. <https://doi.org/10.1109/TPAMI.2004.17>
2. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*,
   second edition, Cambridge University Press, 2004.
   <https://www.robots.ox.ac.uk/~vgg/hzbook/>
3. R. Szeliski, *Computer Vision: Algorithms and Applications*, second
   edition, 2022. <https://szeliski.org/Book/>

## Reproducibility statement

The numerical corpus, raw paired observations, environment, source hashes,
runtime protocol, exact comparison, and threats to validity are recorded under
`benchmarks/translation_orientation/results/`. This document describes the
algorithm implemented in the source checkout; it makes no wheel, sdist, or
published-package claim.
