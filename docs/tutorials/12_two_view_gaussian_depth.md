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

## 3. Export the visible Gaussian centres

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

## 4. Run Gaussian-to-depth feedback

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

## 5. How the solve works

### 5.1 Spherical Gaussian rasterization

For a target-frame point $\mathbf X$, radial range and direction are

$$
r=\lVert\mathbf X\rVert_2,
\qquad
\mathbf b=\frac{\mathbf X}{r}.
$$

The direction is projected to ERP pixel-centre coordinates. Each centre
contributes through a bounded Gaussian image-space kernel. Longitude wraps at
the ERP seam, while latitude never wraps.

A first pass stores the nearest radial surface in every touched pixel. The
second pass rejects contributions farther than the configured occlusion
tolerance. RGB disagreement, voxel observation count, and camera provenance
then modulate the splat weight.

Target and source evidence are accumulated separately. Where both are present,
their log-range disagreement reduces confidence. This produces four
solve-lattice arrays: radial anchors, confidence, provenance mask, and point
count.

### 5.2 Edge-aware log-range correction

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

## 6. Inspect the outputs

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

## 7. Frozen development result

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

## 8. Failure modes and boundaries

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
  feedback loop; it does not claim the adaptive split/prune, learned opacity,
  or full renderer behavior of a canonical CUDA 3D Gaussian Splatting system.

The next valid promotion step is a frozen, no-retuning evaluation over several
independent eligible pairs, including pose perturbations, thin structures,
textureless regions, seam-adjacent surfaces, and depth discontinuities.
