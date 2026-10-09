# Experimental spherical multiview depth refinement

This benchmark evaluates whether registered neighboring P74 panoramas can
improve a frozen native-resolution CNN radial-range prior. It is deliberately
outside the `panorai` package: P74 data, model checkpoints, predictions, and
research optimization are not projection-library assets.

The target is W121. W119 and W124 constrain a differentiable spherical warp;
W122 is held out until both refined maps and their hashes have been written.
Registered P74 poses are a **pose-oracle control**, not an RGB-only system. The
ground truth is loaded only after refinement and held-out scoring.

The refiner optimizes a per-native-pixel log-range residual. Horizontal
sampling wraps at the ERP seam, latitude uses pixel centers, loss aggregation
uses spherical solid-angle weights, and the optional pairwise term is an
edge-aware CRF-like regularizer. A pixel is photometrically optimized only when
both registered source views support it; their costs are averaged rather than
letting a pixel select whichever one is easiest. It does not build an image
pyramid or resize the model output. The P74 RGB adapter only changes coordinates from its native
0--150 degree polar grid to the audited equal-angular ERP at non-minifying
angular density.

The `DepthPrior` interface contains radial range, validity, optional confidence,
and provenance. It is intentionally independent of the CNN implementation so
that a future ViT/transformer backend can provide the same contract without
changing the multiview geometry.

Run the frozen experiment:

```bash
python benchmarks/spherical_multiview_depth/run_p74_experiment.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/W121-spherical-radial.npy \
  --ground-truth /path/to/W121-gt-radial.npy \
  --evaluation-validity /path/to/W121-depth15-evaluation-validity.npy \
  --output /path/to/output
```

The runner produces native float32 NPY maps in metres, native uint16 PNG maps
encoded as `depth / 15 m`, independently checked binary PLY clouds, a visual
audit panel, the pre-W122 freeze manifest, and `results.json`. Numerical metrics
never use the decimated display panel.

Primary comparisons are metric depth, scale-aligned relative 3D RMSE,
scale-invariant log RMSE, local surface-normal agreement, latitude bands, seam
continuity, and W122 held-out reprojection cost. Compare:

1. frozen CNN prior;
2. multiview optimization without the pairwise term;
3. multiview optimization with the CRF-like pairwise term.

This is experimental benchmark code, not a stable public API or publication
claim.

The first complete W121 control is documented in [`RESULTS.md`](RESULTS.md).
Its multiview residual overfits the fixed optimization views and is a negative
result. To inspect generated PLYs in browsers that do not handle `.ply`, build
a self-contained WebGL viewer:

```bash
python benchmarks/spherical_multiview_depth/make_point_cloud_viewer.py \
  "CNN=/path/to/cnn-prior.ply" \
  "multiview=/path/to/multiview.ply" \
  --output /path/to/viewer.html
```

## Bidirectional local cost volume

`run_p74_bidirectional_experiment.py` implements the stricter stereo procedure
that followed the first control. For every target pixel and its frozen CNN
radial-range seed, it samples a local inverse-range distribution, projects all
hypotheses through the fixed registered `R,t` into a source ERP, and forms a
native spherical cost volume. The soft forward proposal is then used to anchor
a reverse source-to-target search. A proposal is fused only when confidence,
angular pixel-cycle error and metric range-cycle error pass explicitly recorded
thresholds. The ERP epipolar locus is obtained by 3D projection; it is never
approximated as a horizontal image line.

RGB, photometric features, `R,t`, the CNN and the seed remain constants. Only
radial-range hypotheses differ. W119 and W124 produce independent reciprocal
proposals; agreement is checked in log range before fusion. The computation is
striped at native 4128×8256 and creates no resized image or depth pyramid.

```bash
python benchmarks/spherical_multiview_depth/run_p74_bidirectional_experiment.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/W121-spherical-radial.npy \
  --ground-truth /path/to/W121-gt-radial.npy \
  --evaluation-validity /path/to/W121-depth15-evaluation-validity.npy \
  --output /path/to/bidirectional-output
```

The preliminary native result is recorded in
[`RESULTS-BIDIRECTIONAL.md`](RESULTS-BIDIRECTIONAL.md).

## Native DoG + tangent-descriptor depth seeds

`run_tangent_seed_experiment.py` tests the sparse alternative to a dense
photometric cost volume. It runs the existing PanorAi spherical DoG detector
directly on the native ERP, materializes 48x48 gnomonic tangent patches, and
uses the v2 descriptor adapter with locally standardized RootSIFT
(`r6/d1.5`, fixed orientation). Lowe matching is executed in both directions;
only mutual matches survive bilateral spherical NMS.

For every accepted correspondence, the benchmark samples a log-inverse-range
distribution around the frozen monocular prior. The discrete distribution
selects a local basin; continuous two-ray triangulation supplies the final
metric radial range only when it remains inside that basin and passes explicit
reprojection, parallax, cheirality, and ray-miss gates. Corrections are
propagated only inside the keypoint's angular descriptor support, with exact
spherical distance and an RGB boundary weight. Pixels without support remain
bit-identical to the prior.

The input panorama and depth map are never resized. The DoG detector still has
its standard spherical Gaussian octave pyramid; this is an anti-aliased
detector scale space, not a resized inference image. The runner also executes
PanorAi's five-point LO-RANSAC and nonminimal pose refit as a diagnostic. By
default, depth uses the registered metric pose so descriptor, depth, and pose
errors are not conflated. A separate CLI mode can use the refined rotation and
translation direction with the registered baseline scale.

```bash
python benchmarks/spherical_multiview_depth/run_tangent_seed_experiment.py \
  --p74-root /path/to/P74/eq \
  --prior /path/to/W121-spherical-radial.npy \
  --ground-truth /path/to/W121-gt-radial.npy \
  --evaluation-validity /path/to/W121-depth15-evaluation-validity.npy \
  --output /path/to/tangent-seed-output \
  --source-id P-74+MD-04_concluido_408+W_119
```

The first native pilot and its coverage limitation are documented in
[`RESULTS-TANGENT-SEEDS.md`](RESULTS-TANGENT-SEEDS.md).
