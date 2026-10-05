# Decoupled spherical relative pose

## Abstract

Weak-parallax spherical pairs often contain many distant correspondences that
constrain rotation accurately and only a few nearby correspondences that
constrain translation. Ranking one Essential matrix by a single scalar score
can therefore improve rotation while degrading translation. This Experimental
method separates the two observability regimes and abstains when translation
is not distinguishable from a competing direction.

## Coordinate model

PanorAi estimates panorama-2-from-panorama-1 pose

$$
\mathbf x_2=R\mathbf x_1+\mathbf t,
\qquad R\in SO(3),\quad \|\mathbf t\|=1.
$$

Metric translation magnitude is not observable. Bearings are calibrated unit
vectors and satisfy

$$
\mathbf b_2^T[\mathbf t]_\times R\mathbf b_1=0.
$$

## Stage 1: rotation-only consensus

Three-correspondence samples generate proper Wahba rotations. Candidates are
ranked by truncated quadratic angular cost rather than maximum consensus:

$$
C_R(R)=\sum_i\min\left[
\left(\frac{\arccos(\mathbf b_{2i}^TR\mathbf b_{1i})}{2.5\tau}\right)^2,
1\right].
$$

The best inlier set is refit by SVD until the mask stabilizes, cycles, or
reaches the configured cap. At least half the valid rays must support the
rotation-only explanation. Otherwise the method is inapplicable and the
existing joint estimator is retained.

## Stage 2: translation consensus

Rays outside the rotation-only consensus contain either measurable
translation or gross correspondence errors. For each such ray,

$$
\mathbf n_i=(R\mathbf b_{1i})\times\mathbf b_{2i},
\qquad \mathbf n_i^T\mathbf t=0.
$$

Every non-degenerate pair proposes
$\mathbf t_{ij}\propto\mathbf n_i\times\mathbf n_j$. The implementation
evaluates both orientations. At residual scales
$s\in\{0.8,1.0,1.2,1.4,1.6,2.0,2.5\}\tau$, its score is

$$
S(\mathbf t)=\sum_s\sum_i
w_i\,\mathbb 1[\text{positive depth}]\,
\max\left(1-\frac{r_i(\mathbf t)^2}{s^2},0\right),
$$

where $w_i$ is the bounded parallax evidence used by PanorAi's orientation
policy. Consensus refits solve the null vector of normalized plane normals and
continue to a stable or repeated mask.

## Ambiguity and abstention

Let $S_1$ be the selected score and $S_2$ the best score whose oriented
translation differs by more than 10 degrees. The reported relative margin is

$$
m_t=\frac{S_1-S_2}{\max(S_1,\epsilon)}.
$$

The default Experimental gate requires $m_t\ge0.15$. A pool with fewer than
five rays is unobservable. Either failure is reported and rejected rather than
silently converted into a confident pose.

## Compatibility and limitations

The default estimator remains joint LO-RANSAC/IRLS. The method assumes a
dominant far-background rotation consensus and deliberately falls back for
ordinary close-range scenes. Its current evidence is synthetic source-checkout
evidence; real feature matches and calibrated multi-domain pairs remain
required before default promotion.

## Frozen synthetic result

On 20 held-out cases per condition, the guarded estimator raised clean
weak-parallax strict accuracy from 10% to 100%, reducing median `R` error from
0.103 to 0.015 degrees and median oriented `t` error from 36.110 to 0.954
degrees. With 5% outliers it accepted 17/20 cases; every accepted pose had
`R < 0.1` and `t < 5` degrees, with translation P95 2.580 degrees and zero
false accepts. It retained the compatibility result for all 20
well-conditioned cases and rejected all 20 unobservable controls. Full raw
data, environment and fingerprints are in `results/summary.json` and
`results/raw.csv`.
