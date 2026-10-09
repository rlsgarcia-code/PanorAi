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
