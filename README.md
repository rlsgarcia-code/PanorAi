# PanorAi

**Convention-safe spherical projection, vision, reconstruction, and SLAM.**

PanorAi starts with stable geometry, modality-aware view-processing workflows,
and spherical feature extraction/matching, then adds Experimental relative
pose, PyCOLMAP export, multiview reconstruction, and visual SLAM. NumPy is
required; Torch, PyCOLMAP, Open3D, and dataset adapters remain optional.

The canonical frame is explicit: pixel centers, top-left image origin, `+X`
right, `+Y` up, `+Z` forward, radial depth, horizontal seam wrapping, and
geometric support kept separate from data validity.

## Capabilities

| Goal | Public surface | Stability |
| --- | --- | --- |
| ERP, gnomonic, and cubemap projection | `panorai.geometry` | Stable 3.x |
| Panorama → views → model → panorama | `EquirectangularImage.process_views` | Stable v1 |
| Spherical keypoints and matching | `panorai.features` | Stable core v1 |
| Pairwise rotation and translation direction | `panorai.estimators` | Experimental v1 |
| Sparse reconstruction from 3+ panoramas | `panorai.reconstruction` | Experimental v1 |
| Incremental ERP or calibrated-fisheye SLAM | `panorai.slam` | Experimental v1 |

## Installation

```bash
pip install panorai
```

Install only the optional integration you need:

```bash
pip install "panorai[torch]"     # differentiable Torch geometry
pip install "panorai[features]"  # explicit OpenCV feature façade alias
pip install "panorai[pycolmap]"  # virtual-camera rig database export
pip install "panorai[slam]"      # optional ROS bag reader for adapters
pip install "panorai[pcd]"       # Open3D compatibility surface
```

PanorAi supports Python 3.11–3.14. The feature backend supports OpenCV
``>=4.9,<5``.
Names such as `model`, `panorama_a`, and `decoded_erp_frames` below are inputs
owned by the application; each section shows the complete PanorAi call path.

## 1. Project spherical imagery

The stable geometry API accepts NumPy arrays and optional Torch tensors. Shapes
are `(height, width)` and angles are degrees.

```python
import numpy as np
from panorai.geometry import GnomonicProjector, GnomonicSpec

rgb = np.zeros((512, 1024, 3), dtype=np.float32)  # HWC central ERP
spec = GnomonicSpec(
    center_lat_deg=10.0,
    center_lon_deg=30.0,
    hfov_deg=90.0,
    vfov_deg=70.0,
    output_shape_hw=(256, 320),
)
projector = GnomonicProjector(spec, interpolation="bilinear")

view = projector.project(rgb)
restored = projector.back_project(view, output_shape_hw=rgb.shape[:2])

assert view.data.shape == (256, 320, 3)
assert restored.data.shape == rgb.shape
```

`ProjectionResult` keeps support and validity beside the values they describe.

## 2. Run a model over panoramic views

The object workflow chooses modality-safe defaults and delegates all geometry
to the stable core:

```python
import panorai as pa

pano = pa.EquirectangularImage(rgb)
views = pano.views("cube", size=256)
processed = views.map(model)
result = processed.reconstruct()

# Exact shorthand for views().map().reconstruct().
same_flow = pano.process_views(model, layout="cube", size=256)
```

Use `with_depth(..., valid=..., units="m")` for radial range and
`with_labels(...)` for categorical data. Image/depth use bilinear sampling;
labels use nearest. `describe()` exposes every resolved policy.

## 3. Match two panoramas and estimate pose

Given two decoded central ERP arrays, OpenCV supplies local features while
PanorAi owns the virtual cameras, spherical bearings, robust estimation, and
public result objects:

```python
from panorai.estimators import SphericalRelativePoseEstimator
from panorai.features import SphericalFeaturePipeline

pipeline = SphericalFeaturePipeline.from_preset(
    "sift-flann",
    face_sampler="icosahedron",
    face_fov_deg=80.0,
    face_shape_hw=(512, 512),
)
matches = pipeline.extract_and_match(
    panorama_a,
    panorama_b,
    panorama_id_a="pano-a",
    panorama_id_b="pano-b",
)
pose = SphericalRelativePoseEstimator().estimate(
    matches.to_bearing_correspondences()
)

if pose is None or not pose.quality_report.accepted:
    raise RuntimeError("pairwise geometry was not trustworthy")

R_b_from_a = pose.R
t_b_from_a_direction = pose.t  # unit direction; metric scale is unobservable
```

No `cv2.KeyPoint` or `cv2.DMatch` crosses the normal PanorAi API.

## 4. Reconstruct three or more panoramas

Reuse the same feature pipeline and preserve panorama IDs in every pair:

```python
from panorai.reconstruction import SphericalGlobalMapper

panoramas = {"p0": pano0, "p1": pano1, "p2": pano2}
pairs = (("p0", "p1"), ("p1", "p2"), ("p0", "p2"))
pairwise_matches = [
    pipeline.extract_and_match(
        panoramas[a], panoramas[b], panorama_id_a=a, panorama_id_b=b
    )
    for a, b in pairs
]

reconstruction = SphericalGlobalMapper().reconstruct(matches=pairwise_matches)
if not reconstruction.success:
    raise RuntimeError(reconstruction.failure_reasons)

camera = reconstruction.pose("p1")
R_world_to_p1 = camera.R
center_p1_world = camera.center
points_world = reconstruction.points_xyz
assert reconstruction.scale == "arbitrary"
```

The mapper uses accepted pairwise poses, rotation averaging, multiview tracks,
camera/point positioning, spherical bundle adjustment, filtering, and
independent multiview corroboration. Failure returns reasons, not fabricated
geometry.

## 5. Track a central-ERP sequence

```python
from panorai.slam import SphericalIncrementalSLAM

slam = SphericalIncrementalSLAM.from_preset("sift-flann")
for frame_id, timestamp_s, panorama_rgb in decoded_erp_frames:
    tracking = slam.add_frame(
        panorama_rgb, frame_id=frame_id, timestamp_s=timestamp_s
    )
    if tracking.pose is not None:
        consume_pose(tracking.pose)
    else:
        report_loss(frame_id, tracking.reasons)

trajectory = slam.finish()
assert trajectory.scale == "arbitrary"
```

A lost frame has no pose. The Experimental SLAM surface supports automatic
keyframes, a local spherical map, local bundle adjustment, relocalization,
loop discovery, and global correction. It does not claim real-time operation,
metric scale, IMU fusion, or exact dual-fisheye geometry.

## 6. Export virtual cameras to PyCOLMAP

`pipeline.build_virtual_camera_rig(...)` exposes known gnomonic intrinsics and
face-to-panorama geometry. `pipeline.export_pycolmap(...)` writes cameras,
features, and matches to a new COLMAP database. COLMAP/PyCOLMAP can then own
its established SfM lifecycle; this route coexists with PanorAi's independent
Experimental global mapper.

## Modality and compatibility rules

- Do not stack RGB, labels, masks, or depth under one implicit interpolation
  policy; project each modality according to its semantics.
- Continuous image/depth data may use bilinear sampling; labels and masks use
  nearest sampling.
- Depth means radial range unless an API explicitly names another quantity.
- Zero is data, never an implicit invalid marker.
- Strict invalid propagation is the default. Renormalization requires an
  explicit validity mask and `min_valid_weight`.
- Existing 3.0 `attach_*`, `to_gnomonic*`, and `to_equirectangular` names
  remain available throughout 3.x.
- Experimental vision results require diagnostics and conservative acceptance;
  a returned numerical estimate is not automatically trustworthy.

## Documentation

- [Geometry v1 contract](docs/geometry-v1.md)
- [Panorama workflow](docs/explanation/workflow-evolution.md)
- [Spherical features and matching](docs/how_to/spherical_features.rst)
- [Pose and multiview reconstruction](docs/how_to/spherical_reconstruction.rst)
- [Spherical visual SLAM](docs/how_to/spherical_slam.rst)
- [API stability tiers](docs/reference/stability.rst)
- [Changelog](CHANGELOG.md)

Public documentation is built with warnings as errors. Executable examples are
also exercised against an installed wheel. Performance and accuracy statements
are bounded to their recorded fixtures and environments.

## License

PanorAi's distributed source is MIT licensed. Optional upstream projects and
models retain their own terms. See [LICENSE](LICENSE) before distribution.
