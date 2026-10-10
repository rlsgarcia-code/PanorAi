# Tutorial: two-view Gaussian depth feedback

This tutorial closes a deliberately bounded reconstruction loop:

```text
monocular radial prior + two posed ERPs + BA landmarks
        ↓
visible-surface Gaussian representation
        ↓
fused Gaussian-centre cloud
        ↓
spherical visibility splat + edge-aware correction
        ↓
native-resolution refined radial range + confidence
```

The workflow is a **development-only benchmark**, not a Stable PanorAi API or
a general two-image reconstruction guarantee. It reconstructs only surfaces
supported by the two input panoramas; it does not synthesize hidden geometry.

The integration candidate is deliberately narrower than the complete tutorial:
distance-ordered observed surfaces, front-to-back alpha compositing, the frozen
1024x2048 footprint, and optionally the coarse `16x32` radial correction after
metric and held-out gates. The finer radial, covariance, opacity, and color
stages below are retained as reproducible ablations; they are not recommended
product settings. The benchmark tree is not installed as part of the public
`panorai` package.

See the {ref}`capability-map-dense-stereo` theme for the relationship between
monocular priors, known-pose spherical stereo, and visible-surface recovery.

**PanorAi-specific:** ERP pixel-centre rays, canonical `+X` right, `+Y` up,
`+Z` forward coordinates, radial range in metres, explicit validity,
longitude wrapping, non-wrapping latitude, spherical nearest-depth visibility,
and separate confidence/provenance arrays.

## 1. What this stage adds

A monocular model provides dense spatial structure but may have scale drift or
locally biased depth. Sparse bundle adjustment provides accurate geometry at
only a few pixels. A second posed panorama supplies cross-view constraints.
The Gaussian representation connects those signals without forcing a mesh or
inventing surfaces outside observed support.

The feedback stage does **not** treat every Gaussian centre as equally true:

| Evidence | Default role |
| --- | --- |
| target-only centre | weak; substantially circular with the target monocular prior |
| source-only centre | medium; retained only after visibility and RGB consistency |
| target+source visible surface | strong; depth agreement controls confidence |
| BA landmark | hard metric anchor |

This distinction matters. A cloud derived partly from Metric3D cannot become
independent ground truth merely by being written as PLY; this is not an
independent Gaussian-depth oracle.

## 2. Required inputs

The frozen runner expects:

- a native `H×2H` Metric3D radial-range array in metres;
- a binary Gaussian-centre PLY in the target-camera frame;
- BA landmark points with shape `(N, 3)` in the same frame and units;
- the target ERP RGB image;
- evaluation-only registered radial depth; and
- a 2:1 solve lattice, normally matching the Gaussian surface resolution.

The PLY schema is explicit:

```text
float x, y, z
uchar red, green, blue
ushort observations
uchar view_mask       # 1 target, 2 source, 3 both
```

The `observations` field records how many Gaussian centres entered a fused
voxel. It is not a calibrated uncertainty estimate. `view_mask` records camera
provenance.

## 3. Refine Gaussian means step by step

The differentiable renderer exposes a conservative coarse-to-fine experiment
before cloud export. The important distinction is that `16x32`, `32x64`, and
`64x128` are **parameter lattices**, not the ERP render size. At a `512x1024`
render resolution, the finest radial field still has only 8,192 parameters
shared smoothly over 524,288 target Gaussians.

Run the recommended schedule with:

```bash
PYENV_VERSION=panorai python \
  benchmarks/two_view_gaussian_splatting/run_p74_experiment.py \
  --p74-root /path/to/panorama-dataset \
  --prior /path/to/metric3d-radial-m.npy \
  --landmarks /path/to/ba-landmarks-m.npy \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --ground-truth /path/to/evaluation-only-radial-m.npy \
  --height 256 --width 512 \
  --optimization-antialias-samples 4 \
  --render-height 1024 --render-width 2048 \
  --render-antialias-samples 2 --render-point-chunk-size 65536 \
  --hierarchical --hierarchy-iterations 20 20 20 20 \
  --gaussian-radius-px 3 \
  --render-gaussian-sigma-px 1.15 --render-gaussian-radius-px 3 \
  --device mps \
  --output /path/to/hierarchical-run
```

The two resolutions serve different purposes. `--height/--width` select the
photometric optimization lattice. `--render-height/--render-width` select a
separate output lattice and cause the runner to reload the native target,
source, and held-out images directly at that size. It never upsamples the
low-resolution RGB. Stratified subpixel integration suppresses aliasing while
mapping the native polar images to ERP.

The optimizer's bounded field is transferred as

$$
c_{\mathrm{opt}}=\log D_{\mathrm{optimized}}-
\log D_{\mathrm{aligned}},\qquad
D_{\mathrm{render}}=D_{\mathrm{aligned,render}}\exp
(\operatorname{resize}_{\mathrm{periodic}} c_{\mathrm{opt}}).
$$

Longitude interpolation is periodic and latitude is clamped. Covariance
scales are transferred by the same periodic bilinear rule. The renderer then
processes projected points in bounded chunks, so 1024×2048 output is practical
on MPS without materializing every Gaussian/pixel contribution at once.
Inspect `high-resolution-render-comparison.png` for appearance. The original
`heldout-render-comparison.png` is intentionally written at the optimization
size and is useful only for fast numerical diagnosis; enlarging it will show
aliasing by construction.

The stages unlock parameters in this order:

| Stage | Trainable factors | Still frozen |
| --- | --- | --- |
| `16x32` | coarse radial mean correction | covariance, RGB/SH, alpha, pose, count |
| `32x64` | inherited correction + medium residual | same |
| `64x128` | inherited correction + fine residual with analytic tangent covariance | covariance scale, RGB/SH, alpha, pose, tangential mean, count |
| `64x128 covariance` | two bounded tangent-axis covariance scales; radial field frozen | radial/tangential mean, RGB/SH, alpha, pose, count |
| `64x128 opacity` (opt-in) | bounded opacity-logit residual; all geometry frozen | means, covariance, color/SH, pose, count |
| `64x128 color` (opt-in) | bounded view-independent DC color residual | all geometry, opacity, higher-order SH, pose, count |

Every new radial residual begins at zero and is regularized toward the
upsampled preceding stage. The final covariance starts from an analytic
surface-tangent ellipse. Optimization changes only its major/minor scale by at
most 25%; orientation stays tied to the projected surface. A fixed training
mask prevents the optimizer from winning merely by changing coverage.

Evaluate the hierarchy causally:

1. run `--radial-only-hierarchy` with the analytic covariance fixed as the
   three-level mean ablation;
2. run the recommended hierarchy with covariance;
3. compare source loss, landmark error, held-out reprojection, and registered
   depth after the prediction freeze;
4. retain covariance only if metric geometry is non-inferior, not merely if
   source RGB loss decreases.

Each stage is frozen as `optimized-stage-*.npy` before registered depth is
opened, so this comparison does not require rerunning or selecting a stage
after inspecting its ground-truth score.

If and only if the geometry/covariance gates pass, append the two implemented
appearance stages:

```bash
--unlock-appearance-factors --opacity-iterations 12 --color-iterations 8
```

Opacity and DC color have separate learning rates, magnitude bounds, L2 priors,
and spherical smoothness terms. The outputs are `optimized-opacity.npy` and
`optimized-color-rgb.npy`. Tangential mean motion, split/prune, and
higher-order SH remain blocked. Opening factors together would make it
impossible to know which improved appearance and which damaged geometry.

### 3.1 Require the registered-depth renderer control to pass

Before interpreting a photometric optimization, freeze pose and render with
registered target depth. On P74, that control still failed visually across the
1.278 m G046-G047 baseline: a single G046 surface cannot contain G047
disocclusions, and the simplified soft-depth rasterizer fragments strongly
slanted surfaces. A larger visibility tolerance improved numerical coverage
but did not turn the single surface into a valid novel-view model. Therefore
the hierarchy above is a geometry diagnostic, not the recommended visual
renderer.

For visual interpolation, retain one frozen surface per observed panorama and
compose them according to camera proximity. Never symmetrically average the
two layers at an observed endpoint: the endpoint's own surface is primary and
the other is only a hole fallback.

```bash
PYENV_VERSION=panorai python \
  benchmarks/two_view_gaussian_splatting/rerender_two_surface_run.py \
  --p74-root /path/to/panorama-dataset \
  --surface-run /path/to/frozen-two-surface-run \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --interpolation-alpha 0.10 \
  --gaussian-sigma-px 0.85 --gaussian-radius-px 3 \
  --occlusion-tau-m 0.20 --point-chunk-size 65536 \
  --antialias-samples 2 --device mps \
  --baseline-results /path/to/accepted-results.json \
  --output /path/to/view-aware-render
```

This route reloads native RGB at the frozen surface resolution. On the P74
all-seen 1024×2048 representation it reduced source-endpoint RGB L1 from
`0.06172` to `0.02554`; the alpha-0.10 view retained 94.86% coverage and honest
holes. G048 is outside the G046-G047 interpolation segment and remains a
diagnostic extrapolation, not an acceptance view.

The rerenderer exposes three independent quality ablations:

```bash
--projected-covariance-mode jacobian \
--compositing-mode alpha \
--surface-cleaning \
--composition-mode continuous
```

The Jacobian mode estimates centered projected-surface tangents with one-sided
fallbacks at depth edges and adapts the ellipse footprint. Alpha composition
is exact front-to-back for the materialized renderer and uses bounded
front-to-back depth slabs at high resolution. Surface cleaning removes only
isolated floaters and downweights unreliable photometric evidence; it never
adds geometry to a hole. A baseline JSON turns source reprojection and
interpolation coverage into an explicit non-regression gate.

On the frozen P74 1024×2048 surfaces, the combined stack was rejected: G047
RGB L1 changed from `0.02554` to `0.03022`. Alpha with unit opacity was also
rejected at `0.03311`. These capabilities are implemented and tested, but the
default remains `legacy` projected covariance plus `normalized` accumulation
and `primary` composition until another predeclared evaluation passes. The
continuous P74 midpoint showed visible doubled edges/ghosted pipes and is kept
only as an ablation. The baseline gate therefore also limits midpoint RGB drift
from the accepted render, in addition to endpoint error and support coverage.

## 4. Export the visible Gaussian centres

Starting from a completed two-view surface run:

```bash
python benchmarks/two_view_gaussian_splatting/export_gaussian_point_cloud.py \
  --p74-root /path/to/panorama-dataset \
  --surface-run /path/to/two-view-surface-run \
  --refined-pose /path/to/refined-pose.json \
  --voxel-size-m 0.01 \
  --output /path/to/gaussian-centres-1cm.ply \
  --preview /path/to/gaussian-centres-preview.png
```

The exporter transforms source-camera centres into the target frame and
averages centres and RGB values within each one-centimetre voxel. It exports
Gaussian means only; it does not create artificial density by sampling inside
the covariance ellipsoids.

## 5. Run Gaussian-to-depth feedback

```bash
python benchmarks/two_view_gaussian_splatting/run_gaussian_depth_feedback.py \
  --p74-root /path/to/panorama-dataset \
  --prior /path/to/metric3d-radial-m.npy \
  --gaussian-cloud /path/to/gaussian-centres-1cm.ply \
  --landmarks /path/to/ba-landmarks-m.npy \
  --ground-truth /path/to/evaluation-only-radial-m.npy \
  --solve-height 1024 --solve-width 2048 \
  --output /path/to/gaussian-depth-feedback
```

The feedback implementation itself is NumPy/OpenCV and does not require CUDA
or MPS. An accelerator may still be useful in the preceding monocular-model or
Gaussian-rendering stages.

The runner writes every prediction and its SHA-256 before the evaluation-only
ground truth is opened. This ordering prevents the registered depth from
entering the correction solve.

## 6. How the solve works

### 6.1 Spherical Gaussian rasterization

For a target-frame point $\mathbf X$, radial range and direction are

$$
r=\lVert\mathbf X\rVert_2,
\qquad
\mathbf b=\frac{\mathbf X}{r}.
$$

The direction is projected to ERP pixel-centre coordinates. Each centre
contributes through a bounded Gaussian image-space kernel. Longitude wraps at
the ERP seam, while latitude never wraps.

The accepted normalized path first stores the nearest radial surface and then
attenuates farther contributions using the configured occlusion tolerance.
The opt-in alpha path instead composites front-to-back; its high-resolution
chunked form uses bounded depth slabs to avoid materializing every splat.
RGB disagreement, voxel observation count, and camera provenance then modulate
the evidence weight.

Target and source evidence are accumulated separately. Where both are present,
their log-range disagreement reduces confidence. This produces four
solve-lattice arrays: radial anchors, confidence, provenance mask, and point
count.

### 6.2 Edge-aware log-range correction

For accepted anchor pixels, the measured residual is

$$
q_p=\log D_{\text{anchor}}(p)-\log D_{\text{prior}}(p).
$$

Large non-landmark residuals are rejected before optimization. The solver then
estimates a bounded correction field $c$ using anchor data, a zero-correction
prior, and RGB-weighted four-neighbour regularization:

$$
E(c)=
\sum_p w_p(c_p-q_p)^2
+\lambda_0\sum_p c_p^2
+\lambda_s\sum_{(p,q)}g_{pq}(c_p-c_q)^2.
$$

The edge weight $g_{pq}$ falls exponentially with RGB difference. The solve
therefore spreads a correction across a coherent surface but resists crossing
a strong visual boundary. The final native depth is

$$
D_{\text{refined}}=D_{\text{prior}}\exp(c).
$$

The correction is solved at the Gaussian lattice and bilinearly evaluated on
the original native lattice. The output depth is therefore native-resolution;
it is not a nearest-neighbour enlargement of a low-resolution render.

## 7. Inspect the outputs

The principal files are:

| File | Shape and meaning |
| --- | --- |
| `refined-radial-m.npy` | native `H×2H` radial range in metres |
| `refined-confidence.npy` | native propagated confidence in `[0,1]` |
| `gaussian-anchor-radial-m.npy` | solve-lattice visible Gaussian depth |
| `gaussian-anchor-confidence.npy` | solve-lattice evidence confidence |
| `gaussian-anchor-view-mask.npy` | target/source provenance bits |
| `log-correction-solve.npy` | solved additive log-depth correction |
| `accepted-anchor.npy` | anchors retained by the predeclared residual gate |
| `prediction-freeze.json` | hashes written before evaluation depth is opened |
| `results.json` | evaluation metrics and artifact records |

Load depth and validity separately:

```python
import numpy as np

depth = np.load("/path/to/gaussian-depth-feedback/refined-radial-m.npy")
confidence = np.load("/path/to/gaussian-depth-feedback/refined-confidence.npy")

valid = np.isfinite(depth)
trusted = valid & (confidence >= 0.10)

assert depth.ndim == 2
assert confidence.shape == depth.shape
```

Do not infer validity from `depth != 0`. Invalid floating-point range remains
`NaN`, while confidence is a separate numeric field.

## 8. Frozen development result

The first G046/G047 run used 824,040 fused centres and a 1024×2048 solve
lattice, then applied the correction to the original 4128×8256 prior. On
25,162,111 common evaluation pixels:

| Metric | Metric3D prior | Gaussian feedback |
| --- | ---: | ---: |
| AbsRel ↓ | 0.244529 | **0.196196** |
| RMSE ↓ | 1.472432 m | **1.284763 m** |
| $\delta_1$ ↑ | 0.489551 | **0.619498** |

Of 17,613,016 pixels changed by at least one percent, 86.34% moved closer to
registered depth. A clean repeat wrote all nine prediction arrays byte for
byte identically, and an independent metric calculation reproduced the three
directions of improvement.

These numbers describe one external industrial pair and the source-checkout
benchmark. They are not installed-wheel evidence and must not be generalized
to arbitrary scenes.

## 9. Failure modes and boundaries

- Two panoramas cannot constrain surfaces invisible in both views.
- Incorrect metric baseline or pose moves every source-derived surface.
- Repeated texture and occlusion can produce plausible but wrong source-only
  anchors.
- Target-only centres largely repeat the monocular prior and therefore receive
  low confidence.
- RGB edges are imperfect geometry boundaries; textureless depth
  discontinuities remain difficult.
- Confidence is an evidence-ranking signal, not a calibrated probability.
- BA landmark residuals are construction checks when those same landmarks are
  enforced as hard anchors.
- This implementation represents observed surface Gaussians and their
  feedback loop. It supports bounded learned opacity and DC color experiments,
  but does not claim adaptive split/prune, higher-order SH, or full renderer
  behavior of a canonical CUDA 3D Gaussian Splatting system.

The next valid promotion step is a frozen, no-retuning evaluation over several
independent eligible pairs, including pose perturbations, thin structures,
textureless regions, seam-adjacent surfaces, and depth discontinuities.
