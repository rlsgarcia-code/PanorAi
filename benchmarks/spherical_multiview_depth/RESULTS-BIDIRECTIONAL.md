# Bidirectional spherical cost-volume result

Status: **implemented development control; globally neutral-to-negative**. This
is not publication evidence.

The experiment implements the intended depth-seeded stereo search rather than
the earlier free per-pixel residual optimization. For every W121 pixel it
samples 17 inverse-range hypotheses over `seed / 1.5` to `seed * 1.5`, projects
them through fixed P74 `R,t` into W119 or W124, selects a differentiable soft
proposal, and searches back from the projected source ray to W121. RGB,
features, poses, CNN weights and the seed are fixed. Only radial range varies.

All predictions were frozen at native 4128×8256 before W122 or ground truth was
opened. The common metric mask contains 23,122,418 pixels and caps radial range
at 15 m.

| Variant | AbsRel ↓ | scale-aligned relative 3D RMSE ↓ | mean normal error ↓ | W122 feature cost ↓ |
|---|---:|---:|---:|---:|
| CNN seed | **0.31033** | **0.40093** | **48.20°** | 0.170597 |
| forward cost volume | 0.31855 | 0.40454 | 70.40° | 0.171104 |
| bidirectional reciprocal | 0.31083 | 0.40120 | 50.47° | **0.170570** |

The forward volume reproduces the high-frequency failure of unconstrained
photometric fitting. Reciprocity is highly selective: W119 accepts 318,650
pixels (2.23% of textured candidates) and W124 accepts 332,693 (2.33%), with
mean return errors 0.80 and 0.82 pixels. The union contains 633,644 pixels;
488,363 both lie on the common evaluation mask and actually change depth.

Reciprocity nearly restores the seed globally and slightly improves the W122
photometric score, but the accepted changes are not geometrically reliable:
only 47.0% reduce per-pixel GT relative error. On changed pixels, AbsRel rises
from 0.29622 for the seed to 0.31668 after reciprocal matching. Confidence from
cost-volume entropy/margin is not predictive of correctness in this scene.

This is a materially different result from the first control: the two-way
algorithm works as specified and suppresses almost all false forward matches,
but point descriptors remain ambiguous in repeated industrial structure. A
naive axis-aligned 3×3 ERP patch descriptor was rejected in the analytic test
because it ignores the local spherical/perspective warp. The next valid change
is a geodesically warped patch or learned frozen feature descriptor evaluated
at every depth hypothesis, plus visibility/occlusion reasoning—not weaker
cycle thresholds or additional iterations.

The external evidence bundle is
`/private/tmp/panorai-val029-bidirectional-depth`. It contains all native maps,
confidence and accepted-count arrays, verified PLYs, the pre-W122 manifest,
visual panels and `results.json`.
