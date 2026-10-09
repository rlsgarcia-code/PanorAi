# Tutorial: pretrained spherical FCN and class-activation maps

This tutorial ports three Torchvision ImageNet classifiers to differentiable
spherical inference while preserving their learned parameters. The API is
**Experimental** and opt-in under `panorai.experimental.deep_learning`; it is
not part of the stable projection contract.

See the {ref}`capability-map-image-processing` theme for the distinction
between sphere-native sampling and planar models evaluated through projected
views.

**PanorAi-specific:** full spatial-layer inventory, exact parameter reuse,
longitude-wrapped tangent sampling, spherical-area score pooling, checkpoint
provenance, and explicit separation between CAM localization and semantic
segmentation.

## 1. Install the optional stack

Install Torch and Torchvision only when this experiment is required:

```bash
pip install "panorai[deep-learning]"
```

The ordinary `panorai`, `panorai.geometry`, and
`panorai.experimental` imports remain independent of Torchvision. Importing
the explicit deep-learning submodule requires Torch, while Torchvision is
loaded lazily when a pretrained model is requested.

Supported official `DEFAULT` weights are:

| architecture | perspective crop | FCN head conversion |
| --- | ---: | --- |
| AlexNet | `224×224` | first linear layer → valid `6×6` convolution |
| VGG16 | `224×224` | first linear layer → valid `7×7` convolution |
| ResNet18 | `224×224` | final linear layer → `1×1` convolution |

## 2. Download automatically or prefetch

No checkpoint is stored in the repository or PanorAi distribution. The first
inference request downloads a missing checkpoint through Torchvision into the
user-controlled Torch cache. PanorAi verifies the URL hash prefix and records
the complete SHA-256, path, size, URL, and whether the file was already cached.

Prefetch all three models before an offline run:

```bash
python benchmarks/spherical_fcn_cam/download_models.py
```

Or prefetch a subset:

```bash
python benchmarks/spherical_fcn_cam/download_models.py \
  --model resnet18 --model vgg16
```

The command emits JSON with schema
`panorai-pretrained-imagenet-cache/v1`. A cached file whose SHA-256 no longer
matches the hash prefix in the official URL is rejected rather than silently
used or deleted. Remove or replace such a file deliberately before retrying.

The same behavior is available from Python through
`prefetch_imagenet_weights()` and `load_pretrained_imagenet_model()`. These
helpers contact the network only when their explicit model-loading operation
finds a missing checkpoint.

```python
from panorai.experimental.deep_learning import prefetch_imagenet_weights

records = prefetch_imagenet_weights(("alexnet", "vgg16", "resnet18"))
for record in records:
    print(record.model_name, record.cache_path, record.sha256)
```

Checkpoint licenses and terms remain those of their upstream publishers. The
prefetch command is acquisition by the user, not redistribution by PanorAi.

## 3. What is ported

For AlexNet and VGG16, `ImageNetFCN` reshapes each linear classifier parameter
into the corresponding convolution parameter. For ResNet18, the final linear
classifier becomes a `1×1` convolution and the global average is moved after
dense prediction.

`port_module_with_report()` then recursively performs:

```text
Conv2d    → SphericalConv2d
MaxPool2d → SphericalMaxPool2d
```

Every convolution reuses the exact source `Parameter` objects. Batch
normalization, activation, dropout, and residual addition remain unchanged
because they do not define a planar spatial neighbourhood. Final class scores
use a cosine-latitude solid-angle average instead of ordinary pixel averaging.

At each output ray, kernel offsets lie in its local tangent basis, are carried
to the unit sphere with the exponential map, and sample the ERP bilinearly with
longitude wrapping. This operation is differentiable with respect to inputs
and weights.

## 4. Run the tracked CC0 example

The repository includes a checksum-pinned, CC0 Poly Haven panorama for the
documentation examples. Run one model per process:

```bash
python benchmarks/spherical_fcn_cam/run_experiment.py \
  --model resnet18 \
  --output-dir /private/tmp/panorai-spherical-fcn-cam \
  --preserve-input-resolution
```

Repeat with `alexnet` and `vgg16`. A missing model is downloaded automatically.
Each model directory contains the input ERP, top-class CAM overlays, and a
`result.json` containing:

- checkpoint URL, cache path, size, SHA-256 and cache status;
- canonical planar FCN parity error;
- input, feature and dense-logit shapes;
- every replaced layer and parameter-identity result;
- class scores and solid-angle CAM concentration;
- runtime, peak process memory, versions and input provenance.

For a custom ERP, provenance is mandatory:

```bash
python benchmarks/spherical_fcn_cam/run_experiment.py \
  --model resnet18 \
  --input /path/to/panorama.jpg \
  --input-license "dataset name and applicable terms" \
  --output-dir /private/tmp/my-spherical-cam \
  --preserve-input-resolution
```

The input must be a `2:1` ERP. Generated outputs should remain outside the
repository unless their redistribution rights and provenance are separately
established.

## 5. Why the dense map is smaller than the panorama

Fully convolutional means that the network accepts spatial dimensions other
than its training crop. It does **not** mean that the output stride becomes one
pixel.

ResNet18 reduces the input by a total factor of 32. Therefore:

```text
1024×2048 ERP → 32×64 feature grid → 32×64 class grid
2048×4096 ERP → 64×128 feature grid → 64×128 class grid
```

VGG16 also produces a stride-32 terminal grid, followed by the valid `7×7`
convolution converted from `fc6`:

```text
1024×2048 ERP → 32×64 features → 26×58 class logits
2048×4096 ERP → 64×128 features → 58×122 class logits
```

CAM overlays are interpolated back to the ERP only for display. That
interpolation does not add spatial evidence. A denser intrinsic map requires a
smaller output stride, skip features, or repeated tangent-view evaluation.

## 6. Resolution is not angular scale

A `3×3` kernel on a `2048×4096` ERP covers a smaller angular neighbourhood
than the same raster kernel on a `224×448` ERP. Running at native resolution
therefore changes both the number of output positions and the angular context
seen by every layer.

This does not imply that spherical weight transfer requires training. It means
that a controlled experiment must declare:

- ERP resolution;
- angular spacing of kernel taps;
- effective receptive field;
- output stride;
- perspective FoV used by any tangent-view reference.

An explicit angular-scale policy or spherical scale pyramid should be tested
before attributing scale differences to a need for fine-tuning.

## 7. Interpret a CAM correctly

The dense class channel is evidence used by an ImageNet classifier. A bright
region means that it contributed strongly to a class score; it is not a
calibrated per-pixel probability and does not necessarily cover the complete
object.

Use **spherical class-activation map** rather than unqualified “saliency map.”
In 360° literature, saliency often denotes predicted human gaze or fixation,
which is a different task. ImageNet labels also do not directly match all
Matterport3D or Stanford2D3D semantic classes.

## 8. Three controlled comparison scenarios

The current implementation supplies scenario 1. The proposed study keeps the
same pretrained weights and compares:

1. **Original spherical FCN, output stride 32.** Preserve the classification
   topology and measure its native dense lattice.
2. **Dilated spherical FCN, output stride 8.** Remove late strides and replace
   them with spherical dilation so the receptive field is retained while the
   class lattice becomes denser.
3. **Independent gnomonic oracle.** Evaluate the unmodified perspective model
   on `224×224` tangent views at declared centers and FoV, then reproject its
   responses to the sphere.

A conventional planar FCN applied directly to the ERP is the
distortion-unaware control. Compare maps before visualization upsampling using
solid-angle-weighted error, correlation, top-fraction IoU, peak geodesic
distance, class-rank agreement, rotation-equivariance error, runtime and peak
memory. Repeat across multiple ERP resolutions while holding angular support
fixed.

The current exponential-map neighbourhood is not yet proven numerically
equivalent to a perspective gnomonic view. Scenario 3 is the independent
oracle required to make or reject that claim.

## 9. Run the licensed public-dataset protocol

Given a locally prepared and licensed manifest, the batch runner selects at
most one development panorama per spatial group, runs every model in a fresh
process, and generates contact sheets:

```bash
python benchmarks/spherical_fcn_cam/run_public_datasets.py \
  --inputs /path/to/prepared-inputs.jsonl \
  --output-dir /private/tmp/panorai-spherical-fcn-public \
  --preserve-input-resolution
```

Matterport3D and Stanford2D3D are not distributed with PanorAi. Their bytes,
derived visualizations and applicable terms remain outside the source and
release artifacts. Current real-data results demonstrate executable
localization, not semantic-segmentation accuracy.

## 10. Current limits

- No ImageNet or panoramic fine-tuning is performed.
- Batch-normalization statistics remain perspective-trained.
- Only AlexNet, VGG16, and ResNet18 have explicit FCN adapters.
- Native inference is CPU/reference code, not an optimized throughput claim.
- Unsupported image regions require explicit masks; black pixels are never
  inferred to be invalid.
- Exact gnomonic equivalence and output-stride-8 dilation remain future
  experimental scenarios, not current capabilities.

Continue with the implementation-oriented
`benchmarks/spherical_fcn_cam/README.md` in the source checkout, the
{doc}`spherical_image_processing` tutorial, and the exact {doc}`../geometry-v1`
coordinate contract.
