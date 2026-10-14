# Choose sampling and fusion

**Level:** Compose · **Strategy:** Projection-domain

Sampling decides where pinhole views look and how much they overlap. Fusion
decides how several valid predictions contribute to one spherical direction.
These choices are independent of the face-local model.

- Use **cube** for six conventional views.
- Use **icosahedron** or **Fibonacci** for more uniform angular coverage.
- Increase overlap when the method is unreliable near face boundaries.
- Use continuous blends for RGB, logits, confidence or range.
- Use nearest sampling and closest fusion for labels and masks.

FOV controls distortion and context: a narrow view is locally more pinhole-like
but requires more faces; a wide view gives the model more context but increases
projection distortion. Record layout, FOV, face size, overlap and blend with
every result.

## Choose from the output, not from the picture

| Output | Face sampling | Reconstruction/fusion |
| --- | --- | --- |
| RGB, float response, logits, CAM | bilinear when valid | feather, Gaussian or weighted mean |
| radial range | bilinear plus explicit validity | validity-aware continuous blend |
| label image or boolean mask | nearest only | closest/discrete policy |
| keypoints | detector-native pixels | bearing lift plus angular NMS |
| boxes, polygons, masks | geometry-preserving lift | solid-angle IoU NMS or merge |
| class scores/embedding per face | one record per supported footprint | coverage-normalized aggregation |

The face size controls the planar method's sampling density. The FOV controls
angular context and distortion. The layout controls directional coverage.
Changing all three at once makes comparisons uninterpretable, so record them
independently.

## Practical selection

Start with `cube` when interoperability with a conventional cubemap matters.
Use `icosahedron` for broadly uniform overlapping views. Use Fibonacci
directions when the experiment requires an explicit number of approximately
uniform centres. Then inspect the support map: a nominal list of views is not
proof that every source direction contributed equally.

Fusion must match the quantity. Gaussian weighting can reduce continuous
border artifacts, but it must never average class IDs. `closest` can preserve
discrete values, but it does not make a low-quality boundary correct.

## Reproducibility record

Store the sampler name and parameters, face shape, horizontal/vertical FOV,
interpolation, support policy, blender and output units. These values are part
of the method, not presentation metadata.

See the [sampler and blender reference](https://rlsgarcia-code.github.io/PanorAi/reference/samplers_blenders.html)
for every supported policy and the
[projection foundations](../tutorials/02_projection_foundations.md) for the
ray geometry. Next: [choose the output contract](outputs_and_observations.md).
