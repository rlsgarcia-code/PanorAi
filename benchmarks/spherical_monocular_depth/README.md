# P74 spherical monocular depth: CNN and ViT controls

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
VAL-024 tested the corresponding intervention directly. Rather than using the
512x1024 raster ratio, it used CNNDepth's canonical perspective focal length
of 519 px: one training pixel is locally about `1/519` rad, equivalent to a
reference ERP near 1630x3261. On the native W lattice this gives a fractional
north/east tap factor of 2.53176 at every spatial layer. The input and output
remained 4128x8256, the source weights/stride/padding/dilation were unchanged,
and no smoothing or spatial output rescaling was applied.

The intervention was a **mixed negative result**. Relative to the one-cell
spherical port, log-depth correlation improved from 0.320 to 0.387 and mean
normal error from 67.9 to 62.8 degrees. Yet scale-aligned relative 3D RMSE
worsened from 0.518 to 0.542. The raw metric prediction became strongly
overscaled: median 10.63 m against a 2.53 m target, delta-1 only 2.9%. This
isolates angular support as a real contributor to local structure, but rejects
it as a sufficient explanation or repair for the inconsistent map.

## ConvNeXt-Large capacity control

VAL-026 replaced only the official Metric3D-v1 ConvNeXt-Tiny backbone and
checkpoint with ConvNeXt-Large/Hourglass. The 813,150,733-byte checkpoint has
203.2M tensor parameters and SHA-256
`0eaaa2501557ac627ada0070e257c6bc74e3e60b45477b29c2efceb70440cfe8`.
The W sentinel remained at 4128x8256, cube faces at 2656, and all learned
parameters stayed frozen and identity-preserved across 66 ported spatial
layers. No resize, prefilter or output smoothing was introduced.

Capacity materially improved image-domain depth structure. The one-cell
spherical Large route reached delta-1 50.9%, log-depth correlation 0.708 and
48.2-degree mean normal error, versus 0.3%, 0.068 and 48.4 degrees for the
Tiny W run. The cubemap Large route reached 36.3%, 0.522 and 49.4 degrees,
versus 4.9%, 0.291 and 53.7 degrees for Tiny. However, global 3D structure
remained poor: optimal-scale relative 3D RMSE was 0.800 for spherical Large
and 0.759 for cubemap Large, both worse than their Tiny counterparts.

The focal-derived Large spherical factor was only 1.31398 because Metric3D's
canonical focal is 1000 px. It reduced optimal-scale 3D RMSE to 0.664 but
degraded delta-1 to 10.1%, log-depth correlation to 0.689, normal error to
55.1 degrees and seam MAE to 1.34 m. Its 6.44 m median prediction was strongly
overscaled against the 2.53 m target. This variant is rejected.

The Large native run peaked near 31.5 GB RSS. Spherical inference took about
471 seconds and cubemap inference 198--213 seconds, excluding model loading,
evaluation and artifact generation. The duplicated cubemap predictions from
the one-cell and fractional runs are byte-identical. Since none of the three
Large routes produced consistent scale-invariant 3D structure, the experiment
stopped after W rather than expanding to G/M.

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

python run_indoor_cnn_experiment.py \
  --p74-root /path/to/eq \
  --source /path/to/depth_any_camera \
  --config /path/to/cnndepth_resnet101_indoor.json \
  --checkpoint /path/to/cnndepth_resnet101_indoor.pt \
  --output /private/tmp/panorai-val024-angular-support-native \
  --route spherical \
  --preserve-angular-support \
  --only P-74+MD-04_concluido_408+W_121

python run_experiment.py \
  --p74-root /path/to/eq \
  --metric3d-source /path/to/Metric3D \
  --checkpoint /path/to/convlarge_hourglass_0.3_150_step750k_v1.1.pth \
  --model-size large \
  --output /private/tmp/panorai-val026-large-native-one-cell \
  --native-angular \
  --only P-74+MD-04_concluido_408+W_121 \
  --spherical-chunk-elements 20000000

# Repeat into a separate output directory with:
# --preserve-angular-support

python run_panoramic_cnn_experiment.py \
  --p74-root /path/to/eq \
  --source /path/to/UniFuse-Unidirectional-Fusion \
  --checkpoint /path/to/Matterport3D/model.pth \
  --output /private/tmp/panorai-val023-unifuse-native \
  --only P-74+MD-04_concluido_408+W_121
```

## ViT native-density tangent control

VAL-034 begins the ViT stage with the required multiface control. Metric3Dv2
ViT-Small and ViT-Large run unchanged on 42 overlapping 812x1400 tangent
windows sampled directly from W121's 4128x8256 ERP. The window focal length is
the native ERP pixels/radian, so the inputs preserve both native angular
density and the model's learned 56.09 by 34.34 degree field of view. There is
no source resize, image pyramid, prediction resize, or fitted output scale.

ViT-Large materially outperformed the spherical ConvNeXt-Large control:
AbsRel 0.1904 versus 0.3096, scale-aligned relative 3D RMSE 0.2638 versus
0.4002, log-depth correlation 0.8705 versus 0.7196, and mean normal error
36.84 versus 48.07 degrees. ViT-Small reached AbsRel 0.2410 and the best
delta-1, 0.6237. A narrow-FOV native-density ablation failed, establishing that
preserving sampling density without preserving learned angular context is not
sufficient. Full tables and limitations are in [RESULTS-VIT.md](RESULTS-VIT.md).

Reproduce the Large control with separately obtained, checksum-verified source
and checkpoint:

```bash
python run_vit_experiment.py \
  --p74-root /path/to/eq \
  --source /path/to/Metric3D-pinned-source \
  --checkpoint /path/to/metric_depth_vit_large_800k.pth \
  --variant large \
  --cnn-prior /path/to/W121-convnext-large-radial.npy \
  --ground-truth /path/to/W121-gt-radial.npy \
  --evaluation-validity /path/to/W121-depth15-validity.npy \
  --output /private/tmp/panorai-val034-vit-large-native-trained-fov
```

The next ViT experiment will compare this tangent-atlas result with ERP tokens
using spherical relative-position bias and then equal-area tokens. Merely
replacing the patch-embedding convolution is not considered a spherical ViT
port; attention neighborhoods and positional geometry must change explicitly.

## Depth Anything 3 Metric Large control

VAL-035 repeats the same 42-view, 812x1400, native-density tangent protocol
with the current Depth Anything 3 Metric Large checkpoint. The benchmark calls
the network directly because DA3's convenient high-level API resizes images by
default. The source ERP, tangent inputs, and predictions are not resized. DA3's
canonical axial depth is multiplied by the native tangent focal length divided
by 300, following the official metric rule, and is then converted to radial
range before PanorAi reconstruction.

On the identical 22,991,801-pixel support, DA3 reached AbsRel 0.1966,
delta-1 0.7105, scale-aligned relative 3D RMSE 0.3385, log-depth correlation
0.8538, and 34.29-degree mean normal error. Relative to Metric3Dv2 ViT-Large,
DA3 has substantially better delta-1, local normals, seam continuity, and raw
metric scale, but slightly worse AbsRel and materially worse global
scale-invariant 3D structure. This is a complementary result, not an overall
win. The full comparison and limitations are in [RESULTS-VIT.md](RESULTS-VIT.md).

Reproduce it with the official DA3 source and Apache-2.0 metric checkpoint,
plus the frozen baseline arrays:

```bash
python run_da3_experiment.py \
  --p74-root /path/to/eq \
  --source /path/to/depth-anything-3 \
  --source-archive /path/to/depth-anything-3-source.tar.gz \
  --checkpoint /path/to/DA3METRIC-LARGE/model.safetensors \
  --cnn-prior /path/to/W121-convnext-large-radial.npy \
  --vit-prior /path/to/W121-metric3dv2-vit-large-radial.npy \
  --vit-validity /path/to/W121-metric3dv2-vit-large-validity.npy \
  --ground-truth /path/to/W121-gt-radial.npy \
  --evaluation-validity /path/to/W121-depth15-validity.npy \
  --output /private/tmp/panorai-val035-da3metric-large-native-trained-fov
```

## DA3 spherical feature-sharing diagnostic

VAL-036 tests the first bounded transformer adaptation without training. The
four frozen DINO feature levels (layers 4, 11, 17, and 23) are extracted for
the same 42 native-density tangent views. Tokens that observe the same
panorama-frame ray are bilinearly gathered and combined with a deterministic
Gaussian center weight before the unchanged official DPT decodes each view.
The source ERP remains 4128x8256, every network input remains 812x1400, and no
source, prediction, or final map resize is performed.

On the same 22,991,801 pixels, direct feature consensus was a limited result.
It reduced seam MAE from 0.008430 to 0.006461 m and scale-aligned relative 3D
RMSE from 0.338527 to 0.336350, but worsened AbsRel from 0.196638 to 0.199603,
delta-1 from 0.710510 to 0.689916, and mean normal error from 34.29 to 36.06
degrees. The result shows that ray correspondence is useful for continuity,
but averaging already position- and view-conditioned ViT features is not a
sufficient spherical transformer adaptation.

This experiment is stage A only. It is not a spherical DPT: the official DPT
convolutions still operate independently in every tangent view. A dense
256-channel tensor at the native ERP size would require about 32.5 GiB in
float32 before decoder intermediates, so the intended stage C requires a
sparse or streaming spherical atlas. Detailed metrics and limitations are in
[RESULTS-VIT.md](RESULTS-VIT.md).

```bash
python run_da3_sphere_experiment.py \
  --p74-root /path/to/eq \
  --source /path/to/depth-anything-3 \
  --source-archive /path/to/depth-anything-3-source.tar.gz \
  --checkpoint /path/to/DA3METRIC-LARGE/model.safetensors \
  --cnn-prior /path/to/W121-convnext-large-radial.npy \
  --vit-prior /path/to/W121-metric3dv2-vit-large-radial.npy \
  --vit-validity /path/to/W121-metric3dv2-vit-large-validity.npy \
  --da3-prior /path/to/W121-da3metric-large-radial.npy \
  --da3-validity /path/to/W121-da3metric-large-validity.npy \
  --ground-truth /path/to/W121-gt-radial.npy \
  --evaluation-validity /path/to/W121-depth15-validity.npy \
  --output /private/tmp/panorai-val036-da3-shared-features
```

## Sparse spherical attention inside DA3

VAL-037 moves communication inside the frozen ViT instead of averaging its
four exported levels. The backbone is split after block index 11. In block
index 12 (the thirteenth transformer block), each target query keeps the
official local attention and additionally attends only to tokens from
overlapping views that observe the same panorama-frame ray. Q/K/V, projection,
normalization, LayerScale, MLP, and all later layers are the official frozen
weights; the adapter adds no learned parameter. A self-only neighborhood is a
neutral operation, and `alpha=0` reproduces all four official feature levels
bit for bit with the real checkpoint.

The selected variant also applies a first-order transport of the learned 2D
absolute position: the sampled source position is removed and the target local
position is added before K/V. This modestly improved every tracked metric over
the otherwise identical no-transport attention ablation, but did not restore
the original metric calibration.

Against DA3 Metric Large on the same 22,991,801 pixels, sparse attention with
position transport improved seam MAE from 0.008430 to 0.004193 m and
scale-aligned relative 3D RMSE from 0.338527 to 0.321668. It simultaneously
worsened AbsRel from 0.196638 to 0.220580, delta-1 from 0.710510 to 0.560978,
scale-invariant log RMSE from 0.264366 to 0.283803, and mean normal error from
34.29 to 36.73 degrees. The result is mixed: sparse attention coordinates
cross-view structure and continuity, but a training-free residual does not
preserve the monocular metric representation expected by the frozen DPT.

The experiment retains the 4128x8256 ERP and 812x1400 view tensors without
resize. It remains an exploratory one-panorama result designed after VAL-036,
not independent publication evidence. Reproduce it by adding the adapter
selection to the VAL-036 command:

```bash
python run_da3_sphere_experiment.py \
  --adapter sparse-attention \
  --position-transport \
  --p74-root /path/to/eq \
  --source /path/to/depth-anything-3 \
  --source-archive /path/to/depth-anything-3-source.tar.gz \
  --checkpoint /path/to/DA3METRIC-LARGE/model.safetensors \
  --cnn-prior /path/to/W121-convnext-large-radial.npy \
  --vit-prior /path/to/W121-metric3dv2-vit-large-radial.npy \
  --vit-validity /path/to/W121-metric3dv2-vit-large-validity.npy \
  --da3-prior /path/to/W121-da3metric-large-radial.npy \
  --da3-validity /path/to/W121-da3metric-large-validity.npy \
  --ground-truth /path/to/W121-gt-radial.npy \
  --evaluation-validity /path/to/W121-depth15-validity.npy \
  --output /private/tmp/panorai-val037b-da3-sparse-attention-pos
```

## Learned token gate and intelligent ERP fusion

VAL-038 adds two development-only adapters around the frozen DA3 model. The
first is a 257-parameter signed token gain at block 12; it also exposes an
eight-dimensional latent per patch. The second is a 449-parameter shared
per-contribution scorer. For every output ERP ray, depth, gain and latent are
sampled at the same continuous gnomonic coordinate. The scorer is applied to
each valid face independently, its local hidden state is FiLM-conditioned by
the corresponding first-MLP latent, and a masked softmax is taken over the
one-to-six valid faces. This supports variable overlap cardinality without a
fixed concatenation or face ordering. The final depth remains a convex
combination and cannot leave the range of valid face predictions.

Training uses RGB and frozen model predictions only: W050/W100/W150 for
fitting, W120 for checkpoint selection, and W121 once after both checkpoints
are frozen. No P74 depth or NPZ enters either trainer. The output fusion
improves the held-out self-supervised consensus loss by 12.9%, but W121 gains
are very small and mixed. Relative to gated Gaussian fusion, RMSE improves
`1.260351 -> 1.259778` m and scale-invariant log RMSE improves
`0.264254 -> 0.264161`, while mean normal error worsens
`34.29 -> 34.73` degrees and seam MAE worsens
`0.008445 -> 0.008608` m. See [RESULTS-VIT.md](RESULTS-VIT.md) for the complete
table and limitations.

The relevant commands are:

```bash
python train_da3_spherical_gate.py \
  --p74-root /path/to/eq \
  --source /path/to/depth-anything-3 \
  --checkpoint /path/to/DA3METRIC-LARGE/model.safetensors \
  --output /private/tmp/panorai-val038-da3-gate

python train_da3_learned_fusion.py \
  --p74-root /path/to/eq \
  --source /path/to/depth-anything-3 \
  --checkpoint /path/to/DA3METRIC-LARGE/model.safetensors \
  --gate-checkpoint /private/tmp/panorai-val038-da3-gate/da3-spherical-token-gate.safetensors \
  --output /private/tmp/panorai-val038-da3-fusion

python run_da3_sphere_experiment.py \
  --adapter learned-fusion \
  --position-transport \
  --gate-checkpoint /path/to/da3-spherical-token-gate.safetensors \
  --fusion-checkpoint /path/to/da3-latent-conditioned-fusion.safetensors
```

Add the external source, checkpoint, P74, frozen-control and output arguments
shown in the preceding `run_da3_sphere_experiment.py` command.

This route does not make the DPT spherical: each face is still decoded
independently. Model source, checkpoints, learned adapters, P74 data, dense
maps and point clouds remain external and are not distributed with PanorAi.

## Canonical spherical feature atlas

VAL-039 makes the pre-DPT representation unique per panorama ray. After the
frozen signed gate, each of DA3's four `58x100x1024` per-face feature levels is
scattered into one `295x590` ERP field and sampled back into all 42 face
lattices. Faces therefore keep their decoder geometry but receive the same
continuous content for the same ray. The 0.6102-degree atlas spacing matches
the native patch density; RGB remains 4128x8256, model inputs remain 812x1400,
and the final depth remains 4128x8256. No image or prediction resize is used.

This improves global spherical coherence over gated Gaussian fusion:
scale-aligned relative 3D RMSE changes `0.338255 -> 0.319315`, scale-invariant
log RMSE `0.264254 -> 0.257358`, and seam MAE `0.008445 -> 0.005265` m. It also
smooths local geometry: AbsRel changes `0.196530 -> 0.201393`, delta-1
`0.710649 -> 0.671702`, and mean normal error `34.29 -> 36.89` degrees. Only
0.058% of broadcast tokens require fallback, so missing atlas support does not
explain the tradeoff.

Run it by replacing the learned-fusion selection above with:

```bash
python run_da3_sphere_experiment.py \
  --adapter canonical-atlas \
  --position-transport \
  --canonical-atlas-height 295 \
  --gate-checkpoint /path/to/da3-spherical-token-gate.safetensors
```

Add the same source, checkpoint, P74, frozen-control and output arguments used
by the other DA3 adapters. This is still not a spherical DPT: already
view-conditioned ViT features are averaged into the field and each official
DPT executes independently. Full metrics and artifacts are documented in
[RESULTS-VIT.md](RESULTS-VIT.md).

## Related primary references

- [Metric3D source and checkpoints](https://github.com/YvanYin/Metric3D)
- [Depth Anything 3 source](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [DA3 Metric Large model card](https://huggingface.co/depth-anything/DA3METRIC-LARGE)
- [Depth Any Camera source and checkpoints](https://github.com/yuliangguo/depth_any_camera)
- [UniFuse source and Matterport3D checkpoint](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)
- [BiFuse (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_BiFuse_Monocular_360_Depth_Estimation_via_Bi-Projection_Fusion_CVPR_2020_paper.html)
- [UniFuse](https://arxiv.org/abs/2102.03550)
- [360MonoDepth](https://arxiv.org/abs/2111.15669)
- [OmniFusion](https://arxiv.org/abs/2203.00838)
