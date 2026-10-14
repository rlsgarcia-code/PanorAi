# Generate a CAM with a pretrained spherical CNN

**Level:** Start · **Strategy:** Sphere-native learned inference ·
**Stability:** Experimental

PanorAi first converts the classifier head to a fully convolutional head, then
replaces its planar spatial layers with spherical layers. The learned parameter
objects are preserved.

```python
import torch
from panorai.core.torch import port_module_with_report, spherical_area_average
from panorai.experimental.deep_learning import (
    ImageNetFCN,
    class_activation_map,
    load_pretrained_imagenet_model,
)

loaded = load_pretrained_imagenet_model("resnet18")
model = ImageNetFCN(loaded.model, "resnet18").eval()
report = port_module_with_report(model)

# `rgb` is the native HWC uint8 ERP supplied by the caller.
values = torch.as_tensor(rgb).permute(2, 0, 1).float()[None] / 255.0
transform = loaded.weights.transforms()
mean = torch.tensor(transform.mean)[None, :, None, None]
std = torch.tensor(transform.std)[None, :, None, None]
normalized_erp_nchw = (values - mean) / std

with torch.inference_mode():
    dense = model.forward_dense(normalized_erp_nchw)
    scores = spherical_area_average(dense.logits)
    class_id = int(scores.argmax(dim=1)[0])
    cam = class_activation_map(dense.logits, class_id, output_shape=rgb.shape[:2])[0]
```

The ERP tensor must use the normalization declared by `loaded.weights`; no
face set is created. `Conv2d` and `MaxPool2d` become spherical operations,
while normalization, activation, residual addition and other pointwise
operations remain unchanged.

## What the conversion does

`ImageNetFCN` replaces the fixed classifier aggregation with a dense head. A
larger ERP therefore produces a larger class-score lattice according to the
network output stride. `port_module_with_report()` then changes every supported
spatial neighbourhood in place while preserving learned parameter objects.
The classifier vocabulary remains the fixed ImageNet vocabulary.

The global class score is a solid-angle-weighted average of the dense logits.
This avoids allowing high-latitude ERP rows to contribute more simply because
the rectangular raster contains the same number of columns there.

## Required checks

Fail the run if `report.remaining_planar_spatial_layers` is non-empty. Retain
the exact model name, weights identifier, transforms, input ERP shape, dense
lattice shape, output stride, class index/name and port report. Upsampling the
CAM to ERP size is useful for display, but does not create new evidence.

The example chooses the globally strongest class. For a declared proxy
experiment, choose class IDs before evaluating the images and retain negative
channels as well as visually attractive ones.

## What this does not establish

Mechanical port completeness does not prove that perspective-trained features
retain their semantics under spherical neighbourhoods. Compare global logits
and CAM structure against the unported planar FCN and an independent
gnomonic-view control before making an equivalence claim.

See the [spherical FCN and CAM tutorial](../../tutorials/09_spherical_fcn_cam.md)
for acquisition, pinned weights and the evaluation protocol.

Next: [interpret the CAM](interpreting_cam.md).
