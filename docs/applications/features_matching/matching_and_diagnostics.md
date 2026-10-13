# Match features and inspect the evidence

**Level:** Evaluate · **Strategy:** Hybrid appearance and spherical geometry

Descriptor distance is not geometric proof. Before using matches for pose,
inspect:

- repeatability and spatial/angular coverage in both panoramas;
- seam and face-boundary concentration;
- ratio-test and reciprocal-match effects;
- bilateral angular duplicate suppression;
- Essential-model inlier count and residual percentiles;
- parallax, cheirality and competing rotation-only evidence.

`matches.to_bearing_correspondences()` is the boundary between appearance
matching and pose estimation. Continue with the
[relative-pose quickstart](../pose_range/relative_pose.md) or consult the full
[matching parameter reference](../../tutorials/03_features_and_matching.md).

## A useful diagnostic sequence

1. Count extracted and supported features separately for both images.
2. Record tentative, ratio-filtered, reciprocal and angular-NMS match counts.
3. Plot bearings, not only ERP line segments, to expose polar concentration.
4. Fit the spherical Essential model and report accepted inliers and angular
   residual quantiles.
5. Compare against a rotation-only explanation and inspect parallax and
   cheirality before accepting translation.

These stages localize failure. A low descriptor-match count points to
appearance or sampling; many matches but few geometric inliers point to
ambiguity or bad calibration; many inliers with weak parallax do not support a
stable translation direction.

The accepted output for pose is a pair of aligned bearing arrays plus the
matching provenance. It is not a set of 3-D points. Triangulation happens only
after pose acceptance.
