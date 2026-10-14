# Extract and match visual features

**Outcome:** obtain keypoints, descriptors, ERP coordinates and unit bearings
that remain meaningful across the seam and can feed spherical pose estimation.

![Matched visual features across a spherical panorama](../../_static/tutorials/application-features-mosaic-tunnel.jpg)

## Start

Use the established face-based pipeline:
[extract and match spherical features](quickstart.md).

## Compose

Understand face sampling, bearing lifting and angular deduplication:
[projected features](projected_features.md).

## Extend

Detect scale-space extrema directly on the sphere:
[sphere-native features](sphere_native_features.md).

## Evaluate

Match two panoramas and inspect descriptor and geometric diagnostics:
[matching and diagnostics](matching_and_diagnostics.md).

The full detector comparison and parameter reference remain in the
[features tutorial](../../tutorials/03_features_and_matching.md).
