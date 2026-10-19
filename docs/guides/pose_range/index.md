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

3. Run the complete two-ERP workflow with
   [panoramic dense stereo](panoramic_dense_stereo.md).
4. Use [direct spherical stereo range](dense_stereo.md) when the
   sphere-native estimator is specifically required, or compare the
   [classical rectified matchers](classical_stereo.md) at their pinhole input.

## Multiview surface reconstruction

5. Fuse registered metric ERPs into a world-frame surface with
   [MVS, TSDF and Marching Cubes](surface_reconstruction.md). This route starts
   from supplied camera poses; it does not estimate registration.

## Single-view range

6. [Choose the monocular execution strategy](monocular_depth.md).
7. [Port the Metric3D CNN](metric3d_sphere_native.md) directly to spherical
   operations.
8. Keep [Depth Anything over tangent views](depth_anything_views.md).

## Validate and consume

9. [Validate radial range](validate_range.md) before creating points or
   accepting a range product.
