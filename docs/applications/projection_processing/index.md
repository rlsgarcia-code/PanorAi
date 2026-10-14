# Process and project panoramic images

**Outcome:** reuse an image operation designed for a pinhole camera over a
complete spherical image and return its result to the ERP domain.

![Projection and reconstruction over a public spherical panorama](../../_static/tutorials/application-projection-sundowner.jpg)

## Start

Run one function through the complete face workflow:
[apply a pinhole method over the sphere](pinhole_pipeline.md).

## Compose

Choose the angular coverage, overlap and reconstruction rule:
[sampling and fusion](sampling_and_fusion.md).

## Understand and extend

Choose the correct return path for rasters, points, regions and view-level
evidence: [return the right spherical output](outputs_and_observations.md).

The complete geometry derivation remains in the
[projection foundations](../../tutorials/02_projection_foundations.md).
Modality handling and low-level sampling policies are documented in the
[sampler and blender reference](https://rlsgarcia-code.github.io/PanorAi/reference/samplers_blenders.html).
