# P74 spherical monocular depth: CNN stage

This development-only benchmark compares one frozen external monocular-depth
CNN through two execution routes:

1. **ported spherical CNN** — all learned `Conv2d` and `ConvTranspose2d`
   operators in Metric3D-v1 ConvNeXt-Tiny/Hourglass are replaced by tangent-
   sampled ERP counterparts while retaining the exact `Parameter` objects;
2. **six-face cubemap** — the unchanged planar model runs independently on the
   six canonical 90-degree PanorAi faces, its axial predictions are converted
   to radial range, and the faces are reconstructed to ERP.

The organized P74 XYZ raster is the independent radial-range reference.  Data
validity (finite XYZ with positive range) and the audited 0–150 degree native
polar support are explicit masks.  Scores use pixel-cell solid-angle weights.
Evaluation intersects that mask with Metric3D-v1's declared 0.3–150 m depth
domain, and predictions are clamped to the same domain before log metrics.
No tangent-view oracle, training, fine-tuning, dataset, checkpoint, or upstream
model source is included in PanorAi.

The frozen preliminary sample contains one pre-outcome panorama from each P74
family: `W_121`, `G046`, and `M-014`.  The native-angular protocol infers a
full 2:1 ERP from both scanner sampling axes and rounds the denser estimate
*upward* to the 32-pixel model lattice: 4128×8256 for W/G and 5184×10368 for
M.  It therefore never minifies either source angular axis.  The cubemap face
size is likewise rounded upward from `ERP_width/pi` (2656 or 3328).  The
unchanged planar model runs fully convolutionally on that face and receives
only mean-valued horizontal padding preserving Metric3D's documented
544:1216 canvas ratio.  There is no input or prediction resize in either
inference route.  Coordinate-defined P74→ERP and ERP↔cube projection sampling
remain necessary; panel scaling and deterministic PLY decimation are outside
inference and evaluation.

In addition to solid-angle-weighted metric-depth scores, the native-angular
run reports three scale-invariant structural views:

- a dimensionless point-to-point 3D RMSE after the optimal single global scale
  is fitted to the prediction;
- centred log-depth RMSE and weighted log-depth correlation;
- local surface-normal angular error on a fixed 0.35-degree stencil, with an
  explicit target-continuity mask around depth discontinuities.

The dense radial GT and predictions remain `.npy` files at the complete
evaluation resolution.  Binary little-endian PLYs use canonical PanorAi
`+X right, +Y up, +Z forward` coordinates and metres, but are deterministically
grid-decimated to a caller-selected point limit for interactive inspection.

## Native-angular outcome

The frozen three-panorama run is a **negative baseline for both routes**. The
ported spherical route and native-density cube route respectively produced
macro delta-1 of 2.05% and 2.42%, mean surface-normal errors of 54.9 and 56.7
degrees, log-depth correlations of 0.257 and 0.223, and floor-clamped area of
39.4% and 38.2%. Their optimal-scale relative 3D RMSE was 0.663 and 0.683.
Those values do not represent usable scene depth. A small mean advantage for
one bad route over another is not evidence of semantic equivalence, spherical
equivariance, successful portability, or suitability for P74. G046 also
reversed the aggregate ordering, which makes the scene dependence explicit.

The likely dominant limitation is domain mismatch: the evaluated checkpoint
is documented by Metric3D as outdoor-only, whereas P74 contains indoor
industrial scenes. The experiment also isolates inference portability; it does
not train spherical filters, calibrate the decoder on ERP geometry, or repair
the checkpoint's semantic/domain mismatch.

Metric3D's own test script describes the v1 ConvNeXt-Tiny checkpoint as
outdoor-only, so P74 industrial interiors are deliberately out of its training
domain. The checkpoint repository has no model card or separate checkpoint
license. Its use here is external, development-only evidence; PanorAi neither
redistributes it nor asserts that the BSD-2-Clause source-code license governs
the weights.

## Reproduction

Run from this directory with a separately obtained Metric3D checkout,
`convtiny_hourglass_v1.pth`, and P74 root:

```bash
KMP_USE_SHM=0 OMP_NUM_THREADS=1 \
PYTHONPATH=/path/to/Metric3D:/path/to/PanorAi \
python run_experiment.py \
  --p74-root /path/to/eq \
  --metric3d-source /path/to/Metric3D \
  --checkpoint /path/to/convtiny_hourglass_v1.pth \
  --output /private/tmp/panorai-val020-native-noresize \
  --native-angular \
  --spherical-chunk-elements 50000000 \
  --normal-step-deg 0.35 \
  --ply-max-points 1500000
```

`results.json`, explicit evaluation-validity `.npy` masks, radial `.npy`
predictions, and visual panels are written only to the requested external
output directory. To inspect one panorama in
CloudCompare, open its `-gt-view.ply`, `-spherical-view.ply`, and
`-cubemap-view.ply` files together and toggle their visibility in the DB tree.

## Two-stage research plan

The present stage is CNN-only.  A later ViT stage will keep the same P74 sample,
radial contract, cubemap control, metrics, and masks, then compare (a) cubemap
tokens, (b) ERP tokens with spherical relative-position bias, and (c) equal-area
tokens.  That stage must treat positional embeddings, attention neighborhoods,
and angular scale explicitly; simply replacing patch embedding convolution is
not considered a spherical ViT port.

## Related primary references

- [Metric3D source and checkpoints](https://github.com/YvanYin/Metric3D)
- [BiFuse (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_BiFuse_Monocular_360_Depth_Estimation_via_Bi-Projection_Fusion_CVPR_2020_paper.html)
- [UniFuse](https://arxiv.org/abs/2102.03550)
- [360MonoDepth](https://arxiv.org/abs/2111.15669)
- [OmniFusion](https://arxiv.org/abs/2203.00838)
