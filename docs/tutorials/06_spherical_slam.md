# Tutorial: spherical visual SLAM

This tutorial closes the sequence from panorama pixels to an incremental
camera trajectory. PanorAi exposes two Experimental central-camera routes: one
for already stitched equirectangular panoramas and one for a calibrated
equidistant fisheye lens. Both return arbitrary-scale visual trajectories.

See the {ref}`capability-map-slam` theme in the spherical computer-vision
guide. In particular, SLAM does not filter an ERP or reconstruct one: the
default ERP frontend materializes gnomonic faces for feature extraction, maps
detections to unit bearings, and performs tracking and mapping in bearing
space.

**PanorAi-specific:** central-camera ray conversion, panorama-frame tracking,
spherical 2D–3D refinement and bundle adjustment, explicit tracking states,
relocalization/loop evidence and arbitrary-scale trajectory diagnostics.

## 1. Incremental tracking from central ERP frames

```python
from panorai.slam import SphericalIncrementalSLAM

session = SphericalIncrementalSLAM.from_preset(
    "sift-flann",
    face_sampler="cube",
    face_fov_deg=100.0,
    face_shape_hw=(320, 320),
    max_features=2048,
)

for frame_id, timestamp_s, panorama_rgb in decoded_erp_frames:
    tracking = session.add_frame(
        panorama_rgb,
        timestamp_s=timestamp_s,
        frame_id=frame_id,
    )
    if tracking.pose is not None:
        R_world_to_panorama = tracking.pose.R
        center_world = tracking.pose.center
    else:
        log_tracking_failure(frame_id, tracking.reasons)

result = session.finish()
```

The frontend follows the Stable feature route from
{doc}`03_features_and_matching`: ERP → overlapping gnomonic views → OpenCV
descriptor extraction → panorama bearings. No projected image is blended back
into ERP. Accepted geometric inliers seed the local map; spherical 2D–3D
refinement and local bundle adjustment minimize angular tangent residuals.

Never invent a pose for `tracking.state == "lost"`. Inspect correspondence
count, parallax, active local points and `reasons` on every frame. The final
result can contain a useful partial map while still declaring that the full
trajectory was not completed.

## 2. Supply precomputed spherical features

Use `add_features()` when another component already owns a
`SphericalFeatureSet`. This route starts directly from unit bearings and avoids
re-projecting an ERP inside SLAM:

```python
features = feature_pipeline.extract(panorama_rgb, panorama_id=frame_id)
tracking = session.add_features(features, timestamp_s=timestamp_s)
```

The public feature objects are copied. A reset begins a new session with the
same configuration.

The CI-sized lifecycle example is executable:

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:start-after: DOCS_SLAM_START = None
:end-before: DOCS_SLAM_END = None
:dedent: 4
```

## 3. Track one calibrated fisheye lens

```python
from panorai.slam import EquidistantFisheyeCamera, SphericalVisualSLAM

camera = EquidistantFisheyeCamera.from_kalibr_yaml(
    "kalibr_imucam_chain.yaml",
    camera_id="cam0",
)
session = SphericalVisualSLAM(camera)

for frame_id, timestamp_s, image in decoded_frames:
    session.add_frame(image, timestamp_s=timestamp_s, frame_id=frame_id)

result = session.finish()
```

Here PanorAi inverts calibrated equidistant fisheye pixels directly into unit
bearings. It does not pretend that two non-coincident fisheye lenses form one
central camera. Use a single calibrated lens, or a stitched ERP only when its
central-camera approximation is justified.

The fisheye frontend builds a temporal graph as frames arrive. Rotation
averaging, tracks, positioning, triangulation and spherical bundle adjustment
run when `finish()` freezes the graph.

## 4. Understand the output boundary

| Output | Meaning |
| --- | --- |
| `pose.R` | world-to-panorama or world-to-camera rotation under the documented frame |
| `pose.center` | camera centre in the arbitrary-scale world frame |
| tracking state | explicit initializing/tracking/keyframe/relocalized/lost status |
| map observations | bearing-linked visual landmarks, never reconstructed ERP pixels |
| diagnostics | accepted/rejected edges, inliers, parallax, reprojection and optimization evidence |

Visual-only translation scale is unobservable. Metric evaluation requires an
external scale or similarity alignment. Current limits include rolling shutter,
IMU fusion, metric scale, dynamic-scene modelling, exact generalized dual-
fisheye geometry and bounded-memory marginalization.

Continue with {doc}`../how_to/spherical_slam` for operational guidance and
{doc}`../reference/slam` for the complete Experimental API and limitations.
