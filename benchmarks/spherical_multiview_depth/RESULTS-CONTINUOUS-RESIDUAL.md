# Continuous spherical grid-residual result

## Frozen protocol

The experiment reused the 1,320 strict W119/W124 consensus proposals frozen by
VAL-031. Matching, pose, proposal selection, and the native 4128x8256 CNN prior
were unchanged. No RGB or depth observation was resized. A 129x258 correction
lattice was solved and only this modeled log-depth field was evaluated on the
native pixel centers.

The quadratic objective combined confidence-weighted proposal agreement,
edge-aware spherical-neighbor smoothness, and a zero-residual CNN anchor. The
default weights were fixed before ground truth was opened: seed 64, smoothness
12, anchor 1, RGB sigma 0.12, prior log-depth sigma 0.25, and minimum edge
coupling 0.02. The prediction and hashes were written to
`FROZEN-BEFORE-GT.json` before evaluation.

## Native result

| Full native map | CNN seed | Bounded consensus splat | Continuous consensus |
| --- | ---: | ---: | ---: |
| AbsRel | 0.310326 | 0.304179 | **0.285678** |
| delta-1 | 0.509802 | 0.522725 | **0.576143** |
| RMSE (m) | 1.490122 | 1.482841 | **1.433542** |
| Scale-aligned relative 3D RMSE | 0.400933 | 0.398947 | **0.387227** |
| Scale-invariant log RMSE | 0.374323 | 0.372321 | **0.359349** |
| Log-depth correlation | 0.718195 | 0.720610 | **0.734619** |
| Mean normal error | **48.2036 deg** | 50.8777 deg | 51.1819 deg |
| Normal P90 | **100.5478 deg** | 104.6017 deg | 103.2526 deg |
| ERP seam MAE | **0.089395 m** | 0.092376 m | 0.089575 m |

At an absolute log-residual threshold of 0.005, the continuous field changed
17,570,101 evaluation pixels. It improved 76.12% of them and reduced local
AbsRel from 0.31858 to 0.28931. The solver converged in 158 conjugate-gradient
iterations to a relative linear residual of 9.88e-9; solve plus native field
evaluation took 0.53 seconds on the recorded CPU environment.

The continuous output visually removes the isolated circular splat boundaries.
On an 8-pixel audit lattice, mean absolute residual curvature fell from 0.01553
for the splat to 0.00149 for the continuous field. The seam also returned close
to the CNN baseline.

## Interpretation

The simple continuous field succeeds at its primary densification purpose:
it converts reliable sparse proposals into a coherent native map and improves
both metric depth and scale-invariant 3D structure by substantially more than
the bounded splat.

It does not restore local surface orientation. L2 neighbor smoothness creates
broad residual ramps; these are visually cleaner than splat islands but still
alter depth gradients over large regions. The mean normal error is 2.98 degrees
worse than the CNN seed and 0.30 degrees worse than the bounded consensus
splat. This rejects a claim that smooth interpolation alone repairs geometry.

The next justified model is an edge-aligned piecewise-constant residual field,
for example robust total variation or a small region graph. That objective can
apply near-constant scale corrections inside surfaces while concentrating
transitions at supported boundaries. The existing W121 result must not be used
to retune the current weights; further selection requires a predeclared
ablation and a different frozen target.

## Artifacts and scope

Outputs are external under
`/private/tmp/panorai-val032-continuous-residual`, including the native NPY/PNG,
native residual, compact grid residual, three verified PLY clouds, comparison
panel, `results.json`, freeze manifest, and self-contained `viewer.html`.

P74 and model-derived bytes remain external. This is development-only
source-checkout evidence, not an installed-wheel, stable API, redistribution,
or publication claim.
