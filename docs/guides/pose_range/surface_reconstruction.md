# Reconstruct a surface from registered spherical views

Use this route when camera poses and physical scale are already known. It does
not estimate registration. Start with at least three central 2:1 ERPs for the
candidate reviewed profile.

## 1. Build the registered scene

```python
import numpy as np

from panorai import EquirectangularImage
from panorai.reconstruction import (
    RegisteredSphericalScene,
    RegisteredSphericalView,
    SphericalCameraPose,
)

views = []
for panorama_id, path, R_world_to_camera, center_world in camera_records:
    views.append(
        RegisteredSphericalView(
            image=EquirectangularImage.open(path),
            pose=SphericalCameraPose(
                panorama_id=panorama_id,
                rotation_world_to_panorama=np.asarray(R_world_to_camera),
                center_world=np.asarray(center_world),
            ),
        )
    )

scene = RegisteredSphericalScene(views=tuple(views), units="m")
```

`validity=` can add a boolean HW mask to a view. It is intersected with the
image container's own support and validity. Do not encode missing range or
image samples as zero.

## 2. Run the candidate reviewed profile

```python
from panorai.applications import SphericalDenseReconstruction

method = SphericalDenseReconstruction.reviewed_profile(
    voxel_size=0.05,
    bounds=None,
)
result = method(scene)

if result.accepted:
    mesh = result.require_success().surface
    print(mesh.vertices_xyz.shape, mesh.faces.shape)
else:
    print(result.reasons)
```

With `bounds=None`, bounds come from accepted range points and expand by the
TSDF truncation distance. The method fails rather than silently increasing the
voxel size when the derived volume exceeds `max_voxels`.

The name `reviewed_profile` identifies frozen parameters for future promotion.
The application still has Experimental maturity: the 2026-10-19 gate found no
eligible, prospectively frozen Matterport3D/Stanford2D3D scene manifest after
excluding prior Stereo development cohorts. Analytic and installed-wheel
evidence cannot substitute for that missing scientific evidence.

## 3. Inspect or alter the composition

```python
pipeline = method.as_pipeline()
print(pipeline.describe())

run = pipeline.run(
    scene,
    retain=("estimate_multi_view_ranges", "fuse_tsdf"),
)
surface = run.output.require_success().surface
```

The stages are public:

```python
from panorai.reconstruction import (
    BuildRegisteredReferenceSets,
    EstimateMultiViewRanges,
    FuseTSDF,
    ExtractSurface,
    ColorizeSurface,
    ValidateDenseReconstruction,
)
```

`FuseTSDF` also accepts an external `MultiViewRangeSet`. That is the extension
point for another range estimator; it does not change the TSDF, frame, support
or validity contracts.

## 4. Evaluate against a reference surface

```python
from panorai.evaluation import evaluate_surface_reconstruction

metrics = evaluate_surface_reconstruction(
    result.require_success().surface,
    reference_mesh,
    distance_threshold=2 * method.voxel_size,
)
print(metrics.accuracy_median, metrics.recall, metrics.fscore)
```

The Stable evaluator uses bidirectional vertex-to-triangle distances. Freeze the
distance threshold and held-out scene list before inference. Dataset download,
split selection and experiment orchestration stay outside the library.

Aggregate only after every scene has been evaluated with that same frozen
threshold:

```python
from panorai.evaluation import aggregate_surface_reconstruction

report = aggregate_surface_reconstruction(
    {"scene-001": metrics_001, "scene-002": metrics_002}
)
print(report.per_scene["scene-001"].fscore, report.aggregate["fscore"])
```

The aggregate is the unweighted median of scene-level values, so a finely
tessellated scene cannot dominate a corpus result.

## 5. Optional Open3D conversion

```python
from panorai.pcd import surface_mesh_to_open3d

o3d_mesh = surface_mesh_to_open3d(mesh)
```

Open3D is imported only by this adapter. It is not the canonical surface
contract and is not required to import reconstruction or geometry.
