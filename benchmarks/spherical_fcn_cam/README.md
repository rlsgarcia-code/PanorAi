# Pretrained spherical FCN/CAM experiment

This isolated experiment asks whether a conventional ImageNet classifier can
produce a dense class-evidence map directly on an equirectangular panorama
without tiling it into perspective views.

It has three deliberately separate steps:

1. reinterpret the classifier head as convolutions (FCN);
2. verify that the planar FCN reproduces the original classifier logits at its
   canonical input crop;
3. call ``panorai.experimental.deep_learning`` to port every
   spatial layer and reuse the exact learned weights with tangent-sampled,
   seam-wrapped, differentiable convolution and max-pooling on the ERP sphere.

The tested Torchvision `DEFAULT` weights are:

| model | documented inference resize | documented crop | FCN conversion |
|---|---:|---:|---|
| AlexNet | 256 | 224 × 224 | `6×6` first MLP layer → convolution; later linear layers → `1×1` |
| VGG16 | 256 | 224 × 224 | `7×7` first MLP layer → convolution; later linear layers → `1×1` |
| ResNet18 | 256 | 224 × 224 | final linear layer → `1×1`; spatial logits precede global pooling |

The adapter is not implemented in this benchmark. The extensible public
experiment surface is ``panorai.experimental.deep_learning``; it delegates its
geometric operator to the optional ``panorai.image_processing.torch`` core and
uses the PanorAi frame and ERP pixel-centre convention.
For every output ray, its planar kernel offsets are interpreted in the local
east/north tangent basis, mapped with the spherical exponential map, sampled
bilinearly with longitude wrapping, and contracted with the original learned
kernel. Gradients reach both inputs and weights through `grid_sample` and the
kernel contraction.

The port report in each ``result.json`` inventories every replacement path.
It requires ``Conv2d → SphericalConv2d`` and
``MaxPool2d → SphericalMaxPool2d`` throughout the feature extractor and dense
head, records whether each parameter identity was preserved, and fails if any
conventional spatial layer remains. Batch normalization, ReLU, dropout and
residual addition remain unchanged because they are pointwise or topological;
they do not define a planar sampling neighbourhood. Adaptive/global planar
pooling is removed from the FCN path and final scores use spherical-area
averaging instead.

Run one model per fresh process so runtime and peak RSS remain attributable:

```bash
python benchmarks/spherical_fcn_cam/run_experiment.py \
  --model resnet18 \
  --output-dir /private/tmp/panorai-spherical-fcn-cam
```

Custom inputs must also pass `--input-license`; only the tracked tutorial ERP
is recognized as CC0 automatically. This prevents a custom dataset image from
silently inheriting the tutorial asset's provenance.

Repeat with `alexnet` and `vgg16`. Weights are downloaded by Torchvision into
its external cache. They are never copied into the repository, wheel, or
sdist. The default image is the checksum-pinned CC0 Poly Haven ERP already
used by the PanorAi tutorials.

For a licensed local Matterport360/Stanford2D3D prepared manifest, run the
group-distinct development slice without copying corpus bytes into the tree:

```bash
python benchmarks/spherical_fcn_cam/run_public_datasets.py \
  --inputs /private/tmp/panorai-val010-full-public/pairs-method-inputs.jsonl \
  --output-dir /private/tmp/panorai-spherical-fcn-public-datasets-native \
  --preserve-input-resolution
```

The dataset runner selects at most one source panorama per spatial group,
executes all three models in fresh subprocesses, records solid-angle-weighted
CAM concentration, resumes verified per-view JSON outputs, and creates
per-dataset/per-model contact sheets. Never use
held-out groups for model or threshold selection. Matterport3D and Stanford
2D-3D-S retain their own terms; inputs and derived visualizations must remain
outside source and release artifacts unless redistribution is separately
authorized.

Native resolution is the primary dataset protocol. `--erp-height 224` remains
available only as an explicitly resized scale-control run. Because convolution
weights are reused geometrically, spherical fine-tuning is not required to
claim weight portability; any later training would address semantic domain or
scale adaptation, not define the spherical operator.

## What the output means

`spherical_feature_shape` is the final backbone feature lattice. The
`spherical_logit_map_shape` is the dense 1,000-class score lattice. Global
scores use a cosine-latitude solid-angle average, then softmax. Each overlay is
the positive, normalized dense score for one class, upsampled to the ERP.

These overlays are weak localization evidence, not semantic segmentation.
ImageNet provides image-level labels, not per-pixel supervision; a bright
region indicates evidence used by the converted classifier, not a calibrated
class probability for that pixel.

## Primary references

- Long, Shelhamer, and Darrell, *Fully Convolutional Networks for Semantic
  Segmentation*, CVPR 2015: classification fully connected layers can be
  reinterpreted as convolutions to obtain dense output.
- Zhou et al., *Learning Deep Features for Discriminative Localization*, CVPR
  2016: global average pooling and class weights expose class activation maps.
- Simonyan, Vedaldi, and Zisserman, *Deep Inside Convolutional Networks*, ICLR
  Workshop 2014: image-specific class saliency from score gradients.
- Selvaraju et al., *Grad-CAM*, ICCV 2017: gradient-weighted localization for
  CNN families including VGG and ResNet.
- Su and Grauman, *Learning Spherical Convolution for Fast Features from 360°
  Imagery*, NeurIPS 2017: transfer perspective-network features to ERP-aware
  spherical convolution.
- Coors, Condurache, and Geiger, *SphereNet*, ECCV 2018: adapt convolution
  sampling locations to undo ERP distortion while leveraging existing models.

## Explicit limitations

- No ImageNet or panoramic fine-tuning is performed.
- Fine-tuning is not required for the weight-portability claim. It would be an
  optional semantic/domain adaptation experiment only.
- Batch-normalization statistics remain those learned on perspective crops.
- Kernel weights remain perspective-trained; only their sampling geometry is
  changed. This is closest to SphereNet-style weight transfer, not the learned
  response distillation of Su and Grauman.
- Planar FCN equivalence is directly verified, but equivalence between the
  spherical stack and independently rendered rectilinear tangent views is not
  yet a closed oracle. The current core uses an azimuthal-equidistant
  exponential-map neighbourhood. Exact perspective-camera equivalence would
  require a gnomonic tangent-view oracle and explicit angular-scale contract.
- The MLP-derived VGG/AlexNet logit maps are coarser than ResNet CAMs at a
  224-pixel ERP height because their first classifier convolutions have large
  valid kernels.
- The benchmark is CPU/reference code and makes no throughput claim.
- The namespace is Experimental: it is opt-in, Torch-dependent, outside the
  frozen stable API, and may change before promotion. It is not a release input.
