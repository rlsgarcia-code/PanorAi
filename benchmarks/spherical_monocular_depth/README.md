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

## Indoor CNN intervention

VAL-023 replaced the outdoor checkpoint rather than tuning projections or
post-processing its failed output.  The external `CNNDepth ResNet-101 indoor`
checkpoint published with Depth Any Camera was trained on 670k indoor images
from HM3D, Taskonomy, and Hypersim.  Its 51.7M-parameter state dictionary loads
strictly with `torch.load(..., weights_only=True)`.  PanorAi retains each learned
tensor exactly, preserves the upstream convolution subclass's normalization and
activation post-operations, and does not package source or weights.

The native cubemap route improved coarse depth substantially over the outdoor
baseline on the frozen three scenes: macro delta-1 rose from 2.42% to 32.75%,
log-depth correlation from 0.223 to 0.392, and optimal-scale relative 3D RMSE
fell from 0.683 to 0.513.  This is a real domain benefit, but not a usable depth
map: mean surface-normal error worsened from 56.7 to **73.9 degrees**.  The
`W_121` spherical port produced delta-1 34.97%, relative 3D RMSE 0.518,
log-depth correlation 0.320, and normal error 67.9 degrees.  Visual inspection
shows warped, locally inconsistent surfaces in both routes.

As a stronger control, the official Matterport3D UniFuse ResNet-18 checkpoint
was executed at 4160x8320.  This is the next 64-pixel network lattice above the
non-minifying 4128x8256 P74 target; the P74 raster is projected directly to that
grid, rather than resizing an intermediate ERP.  All learned tensors were
loaded unchanged, while only UniFuse's non-learned ERP/cube sampling grids were
rebuilt for the native shape.  The model was strongly subscaled on `W_121`
(prediction median 0.75 m versus 2.51 m and optimal scale 3.56x).  Removing the
single scale still left relative 3D RMSE 0.471, log-depth correlation 0.277,
and normal error 67.8 degrees.  A panorama-trained residential CNN therefore
did not provide a usable P74 industrial reconstruction either.

These results reject the hypothesis that checkpoint domain alone explains the
failure.  They also expose a resolution issue that “fully convolutional” does
not solve: a model trained at 512x1024 and evaluated at roughly eight times the
linear sampling density sees an eight-times smaller angular receptive field.
The next porting experiment should preserve each learned layer's reference
*angular* support with fractional spherical sampling; it should not smooth or
rescale these failed predictions and call them improved.

Depth Any Camera's top-level repository and model card state MIT, but several
source files carry CC-BY-NC notices.  UniFuse has a top-level MIT license, while
its downloadable checkpoint has no separate weight license statement.  Both
sources and checkpoints therefore remain external to PanorAi, and this
benchmark makes no redistribution claim.

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

The indoor controls use the same native-angular protocol:

```bash
python run_indoor_cnn_experiment.py \
  --p74-root /path/to/eq \
  --source /path/to/depth_any_camera \
  --config /path/to/cnndepth_resnet101_indoor.json \
  --checkpoint /path/to/cnndepth_resnet101_indoor.pt \
  --output /private/tmp/panorai-val023-indoor-cnn-native \
  --route cubemap

python run_panoramic_cnn_experiment.py \
  --p74-root /path/to/eq \
  --source /path/to/UniFuse-Unidirectional-Fusion \
  --checkpoint /path/to/Matterport3D/model.pth \
  --output /private/tmp/panorai-val023-unifuse-native \
  --only P-74+MD-04_concluido_408+W_121
```

## Two-stage research plan

The present stage is CNN-only.  A later ViT stage will keep the same P74 sample,
radial contract, cubemap control, metrics, and masks, then compare (a) cubemap
tokens, (b) ERP tokens with spherical relative-position bias, and (c) equal-area
tokens.  That stage must treat positional embeddings, attention neighborhoods,
and angular scale explicitly; simply replacing patch embedding convolution is
not considered a spherical ViT port.

## Related primary references

- [Metric3D source and checkpoints](https://github.com/YvanYin/Metric3D)
- [Depth Any Camera source and checkpoints](https://github.com/yuliangguo/depth_any_camera)
- [UniFuse source and Matterport3D checkpoint](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)
- [BiFuse (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_BiFuse_Monocular_360_Depth_Estimation_via_Bi-Projection_Fusion_CVPR_2020_paper.html)
- [UniFuse](https://arxiv.org/abs/2102.03550)
- [360MonoDepth](https://arxiv.org/abs/2111.15669)
- [OmniFusion](https://arxiv.org/abs/2203.00838)
