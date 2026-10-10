# Two-view Gaussian depth-prior feasibility experiment

The completed decision record and multiview gates are in
[`CONCLUSION.md`](CONCLUSION.md).

## Integration status

Only the validated rendering path is a promotion candidate:

- one landmark-scaled visible surface per observed panorama;
- camera-distance ordering for multiview fallback;
- depth-binned front-to-back alpha compositing;
- a `sigma=1.3`, radius-4 footprint at 1024x2048; and
- an optional coarse `16x32` radial correction, provided that it passes both
  registered-depth and held-out-appearance gates on the target dataset.

The fine radial hierarchy, learned covariance, learned opacity, learned color,
supersampling and hole filling are retained below as reproducible research
ablations. They are not the product path. This entire implementation remains
under `benchmarks/`, is excluded from the installed `panorai` package, and must
not be treated as a Stable API until it passes a broader multi-scene gate.

This development-only benchmark asks a narrow question: can a dense monocular
radial-range prior, robustly scaled by sparse bundle-adjusted landmarks, be
represented and locally refined as visible-surface Gaussians using one second
posed panorama?

The frozen P74 experiment trains on G046 and G047. G048 and the registered
G046 depth are not opened until the optimized range map and its SHA-256 have
been written. One Gaussian is initialized per supported target pixel. The
default experiment optimizes only a smooth log-range correction. An explicit
full-factor schedule can later unlock covariance, opacity, and bounded DC color
one at a time; camera pose, sparse landmarks, Gaussian count, and tangential
means remain fixed. No Gaussian is created outside declared observed support.

The plain-PyTorch renderer is intentionally small and auditable. It runs on
CPU or Apple MPS, selecting MPS automatically when the process can access it.
Its legacy compatibility path uses Gaussian image-space accumulation and a
nearest-depth visibility gate. The promoted multiview candidate uses
front-to-back alpha compositing: exact when materialized and depth-binned in
the high-resolution chunked path. A centered projected-surface Jacobian is
also implemented, but remains an ablation because it did not pass the frozen
P74 gate.
It is a feasibility control for the geometry/densification hypothesis, not a
claim of feature parity with a canonical 3DGS rasterizer or its adaptive
split/prune policy.

```bash
python benchmarks/two_view_gaussian_splatting/run_p74_experiment.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/G046-monocular-radial-m.npy \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --ground-truth /path/to/G046-registered-radial-m.npy \
  --output /path/to/output \
  --device auto
```

Run the command from the repository root, or use the script's absolute path.
Do not judge visual quality from the default low-resolution panel. Reducing the
native P74 RGB to the optimization lattice permanently removes detail, and
enlarging that RGB afterward cannot restore it. Keep optimization compact and
request an independent render instead:

```bash
--height 256 --width 512 \
--optimization-antialias-samples 4 \
--render-height 1024 --render-width 2048 \
--render-antialias-samples 2 \
--render-point-chunk-size 65536 \
--render-gaussian-sigma-px 1.15 --render-gaussian-radius-px 3
```

The native polar RGBs are regridded directly at each requested size with
stratified subpixel integration. The learned bounded log-range correction is
transferred onto the landmark-aligned prior at render resolution, rather than
enlarging the low-resolution optimized depth. A two-pass chunked rasterizer
bounds contribution memory on Apple MPS. `high-resolution-render-comparison.png`
is the visual artifact; `heldout-render-comparison.png` remains a compact
numerical diagnostic. Neither path invents geometry in disoccluded regions.

The primary comparisons are unaligned monocular depth, landmark-scaled depth,
and two-view Gaussian-optimized depth. An oracle-depth render is reported only
to expose the renderer/photometric floor; it is never used for optimization.

## Archived ablation: hierarchical factors

The original `16x32` correction grid is intentionally coarse: its 512 values
describe a smooth radial displacement field shared by all supported target
Gaussians. It is not the render resolution and it is not a `16x32` point
cloud. Use the opt-in hierarchy to add degrees of freedom in a controlled
order:

1. `16x32`: fit only large-scale radial shape;
2. `32x64`: optimize a zero-initialized residual around the upsampled coarse
   result;
3. `64x128`: add finer radial detail with a fixed surface-tangent EWA basis;
4. `64x128 covariance`: freeze radial geometry and optimize only the ellipse's
   two principal log-standard-deviation scales;
5. optional `64x128 opacity`: freeze means/covariance and optimize a bounded
   opacity-logit residual;
6. optional `64x128 color`: freeze geometry/opacity and optimize a bounded
   view-independent DC color residual.

The fine residual is penalized toward its inherited coarse solution. The two
covariance scales are bounded to `log(1.25)` in magnitude and have strong L2
and spherical smoothness priors. Their principal axes remain determined by
projected surface tangents, so the ellipse cannot rotate freely to explain the
training image. Pose, RGB/spherical harmonics, opacity, tangential mean motion,
Gaussian count, split/prune, pose, landmarks, tangential means, and
higher-order spherical harmonics remain frozen. Two views do not constrain the
angular degrees of freedom of higher-order SH reliably.

```bash
PYENV_VERSION=panorai python \
  benchmarks/two_view_gaussian_splatting/run_p74_experiment.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/G046-monocular-radial-m.npy \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --ground-truth /path/to/G046-registered-radial-m.npy \
  --output /path/to/hierarchical-output \
  --height 256 --width 512 \
  --optimization-antialias-samples 4 \
  --render-height 1024 --render-width 2048 \
  --render-antialias-samples 2 --render-point-chunk-size 65536 \
  --hierarchical --hierarchy-iterations 20 20 20 20 \
  --gaussian-radius-px 3 \
  --render-gaussian-sigma-px 1.15 --render-gaussian-radius-px 3 \
  --device mps
```

Only after the four-stage geometry/covariance run passes its metric gate, add
`--unlock-appearance-factors --opacity-iterations 12 --color-iterations 8`.
The runner writes `optimized-opacity.npy` and `optimized-color-rgb.npy`. Lower
training RGB error alone never promotes these factors; registered depth and
landmark agreement are evaluated after the prediction freeze.

For the required causal ablation, rerun with `--radial-only-hierarchy`; this
keeps the same three radial levels and analytic surface-tangent ellipse, but
does not optimize its covariance scales. Compare registered depth only after
each prediction has been frozen. Accepting a lower training photometric loss alone is not enough: the
stage must preserve landmark agreement and improve held-out or registered
metric geometry. The run writes per-stage parameter counts/losses and, when
enabled, `optimized-covariance-log-scales.npy`. It also freezes one
`optimized-stage-*.npy` radial map per level so the metric effect of each
unlock can be evaluated separately.

The fourth, fifth, and sixth phases have zero trainable radial parameters, so
appearance cannot move geometry while each factor is assessed. The factors are
implemented as separate gates and are never unlocked together. Tangential mean
updates, split/prune, and higher-order SH remain intentionally unsupported.

A six-stage `64x128` MPS smoke run with one iteration per stage confirmed the
integration contract: stages 1–3 contained 512, 2,048, and 8,192 radial
parameters; stage 4 contained 16,384 covariance parameters; stage 5 contained
8,192 opacity parameters; and stage 6 contained 24,576 DC-color parameters.
The registered depth metrics were identical from stage 3 through stage 6,
while the final appearance stage slightly reduced source reprojection error.
This is a wiring check, not evidence to promote appearance optimization.

### Current causal P74 diagnostic

A `256x512` MPS run with `10 10 20 20` iterations validates the mechanism but
also supplies an early stopping warning. Against registered G046 depth, the
landmark-aligned prior had AbsRel `0.69603`; the `16x32` and `32x64` radial
stages reached approximately `0.6411` and `0.6389`. The `64x128` radial stage
regressed to approximately `0.6441` even though its training objective
continued to fall. MPS scatter accumulation produced small last-digit changes
between separate executions, so these rounded values—not cross-run bit
identity—are the appropriate summary. This pair does
not justify choosing the finest grid merely because it has more parameters.

Inside that same frozen run, the covariance-only phase preserved the radial
NPY byte for byte: stage 3, stage 4, and the final file had the same SHA-256 in
the frozen run. It reduced source weighted RGB L1 from `0.16098` to `0.16035`,
but worsened the held-out value from `0.17727` to `0.17780`. Covariance therefore improved the
training appearance slightly, did not and could not improve depth while means
were frozen, and did not pass the held-out appearance gate on this run. These
are development-only diagnostics, not a general recommendation to stop at
`32x64` on other scenes.

### Registered-depth renderer gate and two-surface composition

High-resolution input alone did not make the single-surface render valid. A
causal control froze pose and supplied registered G046 depth as the Gaussian
means. Even this oracle produced conspicuous holes and splat fragmentation
when moved across the 1.278 m G046-G047 baseline. Increasing the visibility
depth scale improved coverage but did not make a single observed surface a
complete model of the second panorama. Do not unlock color, opacity, or more
mean parameters to compensate for that renderer/control failure.

The supported visual route is therefore the already-frozen **two-surface**
representation: G046 supplies one observed surface and G047 supplies another.
At an endpoint, that camera's own surface is primary; at an interpolated pose,
the nearer camera is primary. The other surface fills only pixels where the
primary has insufficient coverage. Symmetric blending is intentionally
rejected because it let an independently estimated farther surface overwrite
valid observed appearance.

Rerender an existing two-surface run without repeating stereo:

```bash
PYENV_VERSION=panorai python \
  benchmarks/two_view_gaussian_splatting/rerender_two_surface_run.py \
  --p74-root /path/to/P74/eq \
  --surface-run /path/to/frozen-two-surface-run \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --interpolation-alpha 0.10 \
  --gaussian-sigma-px 0.85 --gaussian-radius-px 3 \
  --occlusion-tau-m 0.20 --point-chunk-size 65536 \
  --antialias-samples 2 --device mps \
  --baseline-results /path/to/accepted-results.json \
  --output /path/to/view-aware-render
```

On the frozen 1024x2048 all-seen P74 surfaces, view-aware composition reduced
G047 endpoint RGB L1 from `0.06172` to `0.02554` over 82.85% coverage. The
alpha-0.10 interpolation covered 94.86% and was visually much more coherent
than the single-surface render, while retaining honest holes. G048 remained a
poor extrapolation and is not an acceptance image for this two-view method.

In this earlier two-surface control, the renderer-quality stack is opt-in and
independently testable:
`--projected-covariance-mode jacobian`, `--compositing-mode alpha`, and
`--surface-cleaning`. Continuous camera-distance blending is separately exposed
as `--composition-mode continuous`. Surface cleaning removes only isolated observed points
and reduces confidence near depth edges, textureless areas, and clipped pixels;
it never fills invalid support. At 1024×2048 on P74, however, the combined
alpha/Jacobian/confidence run increased G047 RGB L1 to `0.03022` versus the
accepted `0.02554`, so the non-regression gate rejected it. Alpha with unit
opacity also failed (`0.03311`). These modes remain research ablations; the
default stays on the accepted normalized/legacy footprint path for this
two-surface rerendering command. A later frozen three-surface multiview gate,
documented below and in `CONCLUSION.md`, did accept alpha compositing; the two
results use different protocols and are not interchangeable.
The continuous midpoint preserved support but showed obvious double edges and
ghosted pipes against the primary/fallback reference, so the visual-stability
gate also rejected it. `primary` remains the default composition mode.

## Native-resolution two-surface path

The useful two-image result requires running the perspective depth model at
its trained spatial scale. Inferring a 1024x2048 ERP through 196x336 tangent
windows retained the angular field of view but collapsed Metric3Dv2's depth
variation: landmark depth correlation was only 0.14. Inferring the original
4128x8256 ERP through 812x1400 windows raised that correlation to 0.69 and the
registered log-depth correlation to 0.91. Downsampling happens only after
native inference.

Supply one native radial-range NPY for each observed panorama. Model inference
is deliberately outside this benchmark: external Metric3D source, checkpoints,
and their license terms remain caller-managed and are never redistributed by
PanorAi. The arrays used in the frozen run were generated at 4128x8256 through
812x1400 tangent windows and were not resized before model inference.

Then build both observed surfaces. `consistent` keeps only samples corroborated
by both depth and RGB in the other view. `all-seen` retains every sample seen
by either input and gives much denser bounded novel views; it still creates no
surface outside the two source panoramas' support.

```bash
python benchmarks/two_view_gaussian_splatting/run_stereo_surface_experiment.py \
  --p74-root /path/to/P74/eq \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --ground-truth /path/to/G046-registered-radial-m.npy \
  --target-prior /path/to/G046-prior/metric3d-radial-m.npy \
  --source-prior /path/to/G047-prior/metric3d-radial-m.npy \
  --prior-mode all-seen \
  --interpolation-alpha 0.10 \
  --height 1024 --width 2048 \
  --device mps \
  --output /path/to/output
```

For G046/G047, the visually stable range is deliberately bounded near an
observed camera (approximately alpha 0.00-0.20 or 0.80-1.00). G048 is not an
interpolation test: its camera centre is 1.59 m from G046 and 2.78 m from G047,
on the opposite side of G046 from G047. Rendering it is therefore severe
extrapolation, and its holes or floaters must not be presented as a failure of
two-view interpolation. The primary/fallback compositor preserves the nearer
training surface and uses the other only to fill disocclusions.

## Registered multiview gate and learned third surface

Additional panoramas are admitted as independent observed surfaces, not as
extra color observations forced onto the original G046 geometry. The first
gate is a leave-one-out registered-surface oracle: exclude the target panorama,
render every remaining registered XYZ surface into it, order layers by camera
distance, and let farther surfaces fill only unsupported pixels.

```bash
PYENV_VERSION=panorai python \
  benchmarks/two_view_gaussian_splatting/run_multiview_registered_surface_experiment.py \
  --p74-root /path/to/P74/eq \
  --target-id G048 \
  --source-ids G044 G045 G046 G047 G049 G050 \
  --baseline-source-ids G046 G047 \
  --height 512 --width 1024 --antialias-samples 2 \
  --device mps --output /path/to/registered-oracle
```

On P74/G048 the six-view oracle raised coverage from 72.52% to 81.72% and
reduced RGB L1 from 0.13626 to 0.12975. The best compact subset was the two
nearest sources, G049 and G046, at 77.52% coverage and 0.12772 RGB L1. More
distant views filled small additional holes but did not improve error
monotonically, so source count alone is not an acceptance criterion.

The learned-depth confirmation appends a landmark-calibrated native Metric3D
surface from G049 to the already-frozen G046/G047 run. G048 remains held out
until all three source surfaces are fixed:

```bash
PYENV_VERSION=panorai python \
  benchmarks/two_view_gaussian_splatting/run_multiview_prior_surface_experiment.py \
  --p74-root /path/to/P74/eq \
  --frozen-two-view-run /path/to/frozen-G046-G047-run \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --additional-id G049 \
  --additional-prior /path/to/G049-prior/metric3d-radial-m.npy \
  --target-id G048 --height 1024 --width 2048 \
  --antialias-samples 2 --device mps \
  --output /path/to/three-view-result
```

At 1024x2048, the learned third surface with the original compact footprint
raised held-out coverage from 70.20% to 79.43% and reduced RGB L1 from 0.15205
to 0.13725 when the closest camera was primary. A wider `sigma=1.15`, radius-3
footprint reduced visible point sampling without changing geometry; it reached
81.54% coverage and 0.13667 RGB L1. Replacing normalized overlap averaging with
depth-binned front-to-back alpha compositing then reached 81.95% coverage and
0.13330 RGB L1. A final footprint sweep selected `sigma=1.3`, radius 4, reaching
82.38% coverage and 0.13311 RGB L1; `sigma=1.4` added marginal support but
slightly increased L1. The selected footprint and alpha compositing are
therefore the multiview defaults. Pass `--compositing-mode normalized` only to
reproduce the legacy ablation. Using G049 only as a final hole filler remained
photometrically worse.

Two seemingly plausible post-hoc fixes were explicitly rejected. Hard
three-layer depth consensus introduced seams and raised RGB L1 to 0.15113.
Confidence/edge-based cleaning removed 604 isolated points but reduced coverage
and raised RGB L1 to 0.13749. A bounded `32x64` radial update against G046 and
G047 also increased both training-view losses; its stable checkpoint therefore
selected the unchanged input. A 2x supersampled raster improved RGB L1 but
reduced coverage, while Jacobian covariance also exposed larger holes. Closing
radius-2 microholes raised coverage to 82.37% but did not improve RGB L1, so it
was not promoted. These failures indicate that the remaining fragmentation is
dominated by inaccurate depth boundaries and pose/depth inconsistency, not
merely rasterizer aliasing. This validates adding a nearby observed surface and
proper alpha occlusion, but it does not justify synthesizing unseen geometry or
indiscriminately adding distant views.

## Dense RGB point-cloud export

Export the Gaussian centres from both observed surfaces into the G046 frame:

```bash
python benchmarks/two_view_gaussian_splatting/export_gaussian_point_cloud.py \
  --p74-root /path/to/P74/eq \
  --surface-run /path/to/two-view-surface-run \
  --refined-pose /path/to/VAL-042-refined-pose.json \
  --voxel-size-m 0.01 \
  --output /path/to/gaussian-centres-1cm.ply \
  --preview /path/to/gaussian-centres-preview.png
```

The binary PLY contains XYZ, RGB, the number of centres fused into each voxel,
and a view bitmask (`1` for G046, `2` for G047, `3` for both). The exporter
uses the Gaussian centres themselves; it does not fabricate density by drawing
extra samples inside each covariance ellipsoid.

## Gaussian-to-depth feedback

The exported visible Gaussian surface can be projected back into G046 to
refine the native Metric3D radial-range map. This closes the experimental loop
without treating the cloud as independent ground truth: target-only centres
receive very low weight, source-only centres require RGB agreement, and the
strongest anchors are visible-surface splats supported by both views or the BA
landmarks. A spherical nearest-depth gate removes occluded contributions.

The correction is solved in log radial range on the Gaussian lattice with
longitude wrapping and RGB edge-aware regularization, then bilinearly applied
to the original native map. Invalid prior support remains invalid; the method
does not create unseen surfaces.

```bash
python benchmarks/two_view_gaussian_splatting/run_gaussian_depth_feedback.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/G046-prior/metric3d-radial-m.npy \
  --gaussian-cloud /path/to/gaussian-centres-consistent-1cm.ply \
  --landmarks /path/to/G046-G047-ba-landmarks-m.npy \
  --ground-truth /path/to/G046-registered-radial-m.npy \
  --solve-height 1024 --solve-width 2048 \
  --output /path/to/depth-feedback
```

Prediction arrays and checksums are frozen before the evaluation-only ground
truth is opened. The principal outputs are native-resolution
`refined-radial-m.npy` and `refined-confidence.npy`; the solve-lattice Gaussian
depth, confidence, provenance, correction and acceptance maps remain available
for diagnosis. On the frozen G046 run, full-support AbsRel changed from
`0.24453` to `0.19620`, delta-1 from `0.48955` to `0.61950`, and 86.34% of the
17.61 million pixels changed by at least one percent moved closer to registered
depth. These are development-only P74 results, not a general accuracy claim.
