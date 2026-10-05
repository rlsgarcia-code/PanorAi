# PanorAi

**Convention-safe spherical projection and computer vision for Python.**

PanorAi converts between equirectangular panoramas, gnomonic views, cubemaps,
and unit rays without hiding coordinate, interpolation, support, or validity
rules. NumPy is required; Torch and downstream integrations are optional.

## Choose a use case

| I want to… | Start here | Stability |
| --- | --- | --- |
| understand ERP pixels, rays, gnomonic views, cubemaps, samplers, and blenders | [Projection foundations](docs/tutorials/02_projection_foundations.md) | Stable geometry; compatibility samplers |
| run a model over 6 or arbitrary-N views and reconstruct the panorama | [Custom projection workflow](docs/tutorials/01_custom_pipeline.md) | Stable |
| extract SIFT, ORB, or AKAZE features and match panoramas | [Features and matching](docs/tutorials/03_features_and_matching.md) | Stable core |
| estimate spherical relative pose and triangulate two views | [Two-view geometry](docs/tutorials/04_two_view_geometry.md) | Experimental pose |
| estimate a radial-range map from two calibrated ERPs | [Spherical dense stereo](docs/tutorials/06_spherical_dense_stereo.md) | Experimental |
| reconstruct cameras and points from 3+ panoramas | [Multiview reconstruction](docs/tutorials/05_multiview_reconstruction.md) | Experimental |
| track a central ERP or calibrated central-fisheye sequence | [Spherical visual SLAM](docs/how_to/spherical_slam.rst) | Experimental |
| check exact coordinates, masks, interpolation, and cubemap ties | [Geometry v1 contract](docs/geometry-v1.md) | Stable 3.x |

## Install

```bash
pip install panorai
```

PanorAi supports Python 3.11–3.14 and OpenCV `>=4.9,<5`. Optional extras are
`panorai[torch]`, `panorai[pycolmap]`, `panorai[slam]`, and `panorai[pcd]`.

## Minimal projection

```python
import numpy as np
from panorai.geometry import GnomonicProjector, GnomonicSpec

erp_rgb = np.zeros((512, 1024, 3), dtype=np.float32)  # HWC, central ERP
spec = GnomonicSpec(
    center_lat_deg=10,
    center_lon_deg=30,
    hfov_deg=90,
    vfov_deg=70,
    output_shape_hw=(256, 320),
)
projector = GnomonicProjector(spec, interpolation="bilinear")
view = projector.project(erp_rgb)
restored = projector.back_project(view, output_shape_hw=erp_rgb.shape[:2])

assert view.data.shape == (256, 320, 3)
assert restored.support_mask.shape == erp_rgb.shape[:2]
```

`ProjectionResult` keeps values, geometric support, data validity, and valid
interpolation weight separate. Labels and masks use nearest sampling; floating
RGB and radial range may use bilinear sampling. Zero is data, never an implicit
invalid marker.

## What PanorAi owns

| Layer | Owner |
| --- | --- |
| spherical coordinates, projection, masks, provenance, view orchestration | PanorAi |
| SIFT, ORB, AKAZE, BFMatcher, and FLANN implementations | OpenCV |
| spherical five-point pose, dense stereo, and global mapper | PanorAi, Experimental |
| optional COLMAP database/SfM route | PyCOLMAP/COLMAP |

The stable contracts are `panorai.geometry`, the modality-aware object
workflow, the supported blenders, and the `panorai.features` spherical
extraction/matching core. Relative pose in `panorai.estimators`, PyCOLMAP
export, dense stereo in `panorai.stereo`, multiview reconstruction in
`panorai.reconstruction`, multiscale routing, and `panorai.slam` remain clearly
marked Experimental.

## Reference

- [Documentation home](docs/index.rst)
- [API stability tiers](docs/reference/stability.rst)
- [3.4.0 release checklist](docs/release-3.4.0-checklist.md)
- [Changelog](CHANGELOG.md)

PanorAi's distributed source is MIT licensed. Optional upstream projects and
models retain their own terms; see [LICENSE](LICENSE).
