# Compare non-learned stereo pipelines

**Level:** Compose/evaluate · **Input:** an already-rectified pinhole pair ·
**Output:** disparity, validity, confidence and optional radial range

These Experimental baselines make every major classical stereo step
inspectable. Their input is one already-rectified pinhole pair; they do not
interpret an ERP. Use them directly when studying a matcher. Use
[PanoramicDenseStereo](panoramic_dense_stereo.md) to resolve pose, plan and
project rectified view pairs, run the selected matcher and fuse radial range
back onto the sphere.

```python
from panorai.applications import StereoSGBMDepth
from panorai.stereo.classical import (
    RectifiedStereoGeometry,
    RectifiedStereoPair,
    StereoSGBMConfig,
)

stereo = StereoSGBMDepth(
    StereoSGBMConfig(min_disparity=0, max_disparity=128)
)
result = stereo(
    rectified_a,
    rectified_b,
    reference_support=support_a,
    source_support=support_b,
    geometry=RectifiedStereoGeometry(
        focal_length_x=fx_px,
        focal_length_y=fy_px,
        principal_point_x=cx_px,
        principal_point_y=cy_px,
        baseline=baseline_m,
    ),
)

disparity_px = result.disparity
radial_range_m = result.radial_range
valid = result.validity
rectified_pair = RectifiedStereoPair(
    rectified_a, rectified_b, support_a, support_b
)
```

`disparity` is $x_A-x_B$ in pixels. Geometry first gives optical-axis depth
$z=f_xB/d$ and then returns radial range from the reference camera centre.
Invalid numeric samples are `NaN`; `validity`, `occlusion` and `mismatch` are
separate masks. Zero intensity and zero-valued evidence remain valid data.

## Six baselines

| Application | Inspectable composition | Implementation boundary |
|---|---|---|
| `StereoBMDepth` | block matching → validation | OpenCV StereoBM |
| `ADCensusStereoDepth` | AD-Census → cross aggregation → SGM → selection → validation | PanorAi reference operations |
| `StereoSGBMDepth` | block evidence + full SGM → validation | OpenCV StereoSGBM `HH` |
| `CensusMGMStereoDepth` | Census → box aggregation → MGM or tMGM → selection → validation | PanorAi reference operations |
| `ELASStereoDepth` | Census → aggregation → triangulated support prior → SGM → selection → validation | ELAS-style PanorAi reference; not binary-equivalent to libELAS |
| `PatchMatchStereoDepth` | slanted-plane PatchMatch → validation | deterministic PanorAi reference |

All six wrappers expose the same call and return type. Select a bounded
disparity interval appropriate to the rectified baseline; the defaults are not
a scene-specific calibration.

```python
run = stereo.as_pipeline().run(
    rectified_pair,
    retain=("matching_and_optimization",),
)
print(run.describe())
```

For bottom-up experiments, compose the same public operations directly:

```python
from panorai.pipeline import Pipeline
from panorai.stereo.classical import (
    ADCensusCost,
    BuildStereoCost,
    CrossBasedCostAggregation,
    RefineStereoDisparity,
    SelectStereoDisparity,
    SemiGlobalMatching,
    TransformStereoCost,
)

pipeline = Pipeline(
    BuildStereoCost(ADCensusCost(), 0, 128),
    TransformStereoCost(CrossBasedCostAggregation(), "aggregation"),
    TransformStereoCost(SemiGlobalMatching(paths=8), "optimization"),
    SelectStereoDisparity(subpixel=True),
    RefineStereoDisparity(method="my-ad-census"),
)
result = pipeline(rectified_pair)
```

The internal cost volume is $H\times W\times D$ and can be large. MGM/tMGM,
the ELAS-style reference and PatchMatch are correctness-oriented reference
implementations, not current performance claims. Benchmark after fixing image
size, disparity range, support, warm-up, repetitions and hardware.

## What is independently checked

The focused regression uses a textured rectified pair with a closed-form
translation and known disparity. It verifies all six compositions, explicit
support, radial-range conversion, immutable results and safe application
configuration. This is an analytic correctness gate, not evidence that any
method is superior on real scenes. Wide-baseline exposure change, repeated
industrial texture, occlusion and rectification error remain prospective
evaluation requirements.
