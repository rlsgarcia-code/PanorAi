# Estimate pose, depth and radial range

This application is a family of related problems. Choose the first unknown in
your pipeline rather than treating pose and depth as one operation.

![Registered spherical images and radial range estimation](../../_static/tutorials/application-pose-range.jpg)

```text
matched panoramas -> relative pose -> metric scale -> spherical stereo
one panorama      -> sphere-native CNN or tangent-view depth model
                                             |
                                             v
                              radial range + validity + provenance
```

## Pose and scale

1. Estimate rotation and translation direction with
   [relative pose](relative_pose.md).
2. [Recover metric scale and triangulate](metric_scale_and_triangulation.md)
   accepted tracks.

## Two-view dense range

3. Given metric pose, estimate
   [dense spherical stereo range](dense_stereo.md).

## Single-view range

4. [Choose the monocular execution strategy](monocular_depth.md).
5. [Port the Metric3D CNN](metric3d_sphere_native.md) directly to spherical
   operations.
6. Keep [Depth Anything over tangent views](depth_anything_views.md).

## Validate and consume

7. [Validate radial range](validate_range.md) before creating points or
   accepting a range product.
