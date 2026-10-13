# PanorAi

[![PyPI](https://img.shields.io/pypi/v/panorai.svg)](https://pypi.org/project/panorai/)
[![Python](https://img.shields.io/pypi/pyversions/panorai.svg)](https://pypi.org/project/panorai/)
[![CI](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rlsgarcia-code/PanorAi/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-2563eb.svg)](https://rlsgarcia-code.github.io/PanorAi/)
[![License: MIT](https://img.shields.io/badge/license-MIT-059669.svg)](LICENSE)

**Geometry-safe spherical image processing and projection for computer-vision
applications.**

PanorAi works with central spherical images, especially 2:1 equirectangular
panoramas (ERPs). Use it to project local pinhole views, process an ERP directly
on the sphere, match visual evidence, estimate geometry, or combine registered
views into spatial hypotheses.

## Install

```bash
pip install panorai==3.7.0
```

NumPy is required. Install an extra only for the application that needs it:

| Need | Install |
| --- | --- |
| spherical geometry and projection | `panorai` |
| feature extraction and matching | `panorai` (OpenCV is included) |
| Torch spherical models | `panorai[torch]` |
| pretrained CAM and segmentation adapters | `panorai[deep-learning]` |
| PyCOLMAP interoperability | `panorai[pycolmap]` |
| SLAM | `panorai[slam]` |
| point-cloud export | `panorai[pcd]` |

Python 3.11–3.14 and OpenCV `>=4.9,<5` are supported.

## Choose an application

| Application | Start with | Result | Status |
| --- | --- | --- | --- |
| Project views or run a pinhole method | `EquirectangularImage.views()` / `process_views()` | projected views, support, reconstructed ERP | Stable |
| Filter or detect edges on the sphere | `panorai.image_processing` | filtered ERP or spherical response | Experimental |
| Extract and match features | `SphericalFeaturePipeline` | ERP pixels, unit bearings, descriptors, matches | Stable core |
| Estimate relative pose or radial range | `panorai.estimators` / `panorai.stereo` | `R`, translation direction, or radial range | Experimental |
| Reconstruct three or more views | `SphericalGlobalMapper` | poses, bearing tracks, sparse 3D points | Experimental |
| Localize and associate semantic regions | `panorai.experimental.deep_learning` + `panorai.object_localization` | directional evidence and auditable hypotheses | Experimental |

The sections below show the required input, the public call, the returned
object, and the main limitation for each route.

## 1. Project views or run a pinhole method

Use this route when an algorithm expects a conventional camera image. PanorAi
projects overlapping views and maps their results back to the ERP.

```python
import cv2
import numpy as np
from panorai import EquirectangularImage

erp = np.zeros((128, 256, 3), dtype=np.uint8)  # HWC ERP
result = EquirectangularImage(erp).process_views(
    lambda view: cv2.GaussianBlur(view, (5, 5), 1.2),
    layout="cube",
    size=64,
    fov=95,
    blend="gaussian",
)
filtered_erp = result.image
observed = result.support_mask
```

- **Input:** HWC ERP plus a planar function or model.
- **Output:** reconstructed ERP, geometric support, weights, and provenance.
- **Choose explicitly:** face layout, resolution, FOV, overlap, interpolation,
  and blender.
- **Important:** labels and masks require nearest-neighbour sampling; bilinear
  sampling of integer or boolean data fails by design.

Start with [projection foundations](docs/tutorials/02_projection_foundations.md)
or build an [arbitrary-N overlapping layout](docs/tutorials/01_custom_pipeline.md).

## 2. Process directly on the sphere

Use sphere-native operators when the neighbourhood must remain angularly
consistent across the ERP seam and toward the poles.

```python
from panorai.image_processing import (
    spherical_canny,
    spherical_gaussian_blur,
)

blurred = spherical_gaussian_blur(erp, ksize=5, sigma=1.2)
gray = erp[..., 0]
edges = spherical_canny(gray, threshold_low=20, threshold_high=60)
```

- **Input:** ERP array and, where needed, an explicit validity mask.
- **Output:** an ERP-aligned filter response; no face reconstruction is needed.
- **Available routes:** Gaussian/custom convolution, gradients, median, Canny,
  pyramids, rotation, and `spherical_equalize_histogram`.
- **Important:** PanorAi does **not** currently claim a sphere-native CLAHE.
  Apply planar CLAHE through `process_views()` when that trade-off is suitable.

See [spherical image processing](docs/tutorials/spherical_image_processing.md)
and the [reproducible route benchmark](docs/benchmarks.rst).

## 3. Match features and estimate relative pose

The feature pipeline detects on projected views, removes overlap duplicates,
and returns panorama-frame pixels and unit bearings. The relative-pose
estimator consumes those bearings directly.

```python
from panorai.estimators import SphericalRelativePoseEstimator
from panorai.features import SphericalFeaturePipeline

pipeline = SphericalFeaturePipeline.for_relative_pose()
matches = pipeline.extract_and_match(
    erp,
    np.roll(erp, 16, axis=1),
    panorama_id_a="a",
    panorama_id_b="b",
)
pose = None
if len(matches) >= 5:
    pose = SphericalRelativePoseEstimator().estimate(
        matches.bearings_a,
        matches.bearings_b,
    )
```

- **Input:** two ERPs and optional validity masks.
- **Feature output:** matched ERP pixels, unit bearings, descriptors, and stable
  panorama/feature identifiers.
- **Pose output:** `R`, translation **direction**, inliers, residuals, and a
  quality report. Consume `pose.quality_report.accepted`, not success alone.
- **Important:** monocular two-view geometry does not recover metric baseline.

See [features and matching](docs/tutorials/03_features_and_matching.md) and
[two-view geometry](docs/tutorials/04_two_view_geometry.md).

## 4. Estimate radial range

With two aligned ERPs and a known metric relative pose, call
`estimate_spherical_range(erp_a, erp_b, R_b_from_a, t_b_from_a_m, options=...)`.

- **Input:** equal-size ERPs and pose convention
  `X_b = R_b_from_a @ X_a + t_b_from_a_m`.
- **Output:** `result.range`, `result.validity_mask`, confidence, matching cost,
  and consistency diagnostics.
- **Important:** the quantity is radial range, not pinhole Z-depth. If the
  translation has arbitrary scale, the recovered range has the same scale.

See [dense spherical stereo](docs/tutorials/06_spherical_dense_stereo.md) and
the [stereo API](docs/reference/stereo.rst). For optional single-view model
evidence, see [spherical monocular depth](docs/tutorials/10_spherical_monocular_depth.md).

## 5. Reconstruct three or more views

Create pairwise `SphericalFeatureMatches`, then call
`SphericalGlobalMapper().reconstruct(matches=pairwise_matches)`.

- **Input:** pairwise matches with consistent panorama IDs; optionally reuse
  accepted pairwise edges, metric baselines, or matched radial-range priors.
- **Output:** camera poses, conflict-free bearing tracks, sparse 3D points,
  scale state, diagnostics, and explicit failure reasons.
- **Important:** without a metric prior the reconstruction is arbitrary-scale.
  Check `result.success` before consuming poses or points.
- **Alternative:** use the documented PyCOLMAP/COLMAP bridge when external
  mapper interoperability matters more than first-party spherical geometry.

See [multiview reconstruction](docs/tutorials/05_multiview_reconstruction.md),
the [reconstruction API](docs/reference/reconstruction.rst), and
[SLAM](docs/tutorials/07_spherical_slam.md).

## 6. Localize and associate semantic regions

This application has two explicit stages:

1. Produce ERP-aligned directional evidence with a documented provider, such
   as a spherical pretrained CAM adapter in `panorai.experimental.deep_learning`.
2. Pass registered evidence, concepts, poses/tracks, and optional radial range
   to `SphericalObjectHypothesisMethod().run(...)`.

- **Input:** dense semantic channels with support, registered camera geometry,
  and a declared proxy vocabulary.
- **Output:** per-view regions, supporting views/tracks, spatial mode, score,
  provenance, and `proposed`, `ambiguous`, or `confirmed` state.
- **Important:** a CAM is directional evidence, not a segmentation mask or a
  calibrated probability. Evidence without metric geometry can only yield a
  bearing-only proposal.
- **Optional segmentation:** `SphericalProxySegmenter` can use a CAM as a SAM
  prompt, but its automatic proxy label is not ground truth.

See [spherical FCN/CAM](docs/tutorials/09_spherical_fcn_cam.md),
[object hypotheses](docs/tutorials/11_cam_guided_object_hypotheses.md), and the
[object-localization API](docs/reference/object_localization.rst).

## How the routes fit together

| Route | What PanorAi does | Use it for |
| --- | --- | --- |
| Projection-domain | sample gnomonic/cubemap views, preserve support, back-project, blend | planar filters, descriptors, pretrained pinhole models |
| Sphere-native | evaluate neighbourhoods, residuals, or learned operators directly on bearings/ERP | filtering, gradients, pose, stereo, spherical CNN evidence |
| Hybrid | produce local evidence in either domain and combine it with registered spherical geometry | reconstruction, SLAM, semantic association |

These are execution strategies, not separate package hierarchies. The
[capability map](docs/tutorials/spherical_capability_map.md) lists the route
used by each public component.

## Data contract

- Canonical geometry is pixel-center based, top-left origin, `+X` right,
  `+Y` up, `+Z` forward, longitude `[-π, π)`, and north-to-south latitude.
- Depth means radial range unless an API explicitly names another quantity.
- Geometric support, numeric validity, and value are separate. Zero is data;
  it is never an implicit invalid marker.
- NumPy is required. Torch, Open3D, PyCOLMAP, and learned checkpoints remain
  optional and isolated.
- Check the [stability matrix](docs/reference/stability.rst) before depending
  on an Experimental surface.

## Evidence and boundaries

PanorAi owns spherical coordinates, projections, support/validity semantics,
sphere-native kernels, bearing geometry, and its native estimators. OpenCV owns
standard SIFT/ORB/AKAZE descriptors and matchers. Optional PyCOLMAP/COLMAP and
pretrained models retain their own contracts and licenses.

Benchmarks, public-data evaluations, hardware, limitations, and reproduction
commands live on the [evidence page](docs/benchmarks.rst), not in this README.
No datasets or model checkpoints are distributed with the package.

## Reference

- [Documentation home](docs/index.rst)
- [Tutorials by application](docs/tutorials/index.rst)
- [API reference](docs/reference/index.rst)
- [Geometry v1 contract](docs/geometry-v1.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
