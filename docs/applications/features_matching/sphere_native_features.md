# Detect features directly on the sphere

**Level:** Extend · **Strategy:** Sphere-native detection with local projected
description · **Stability:** Experimental

`SphericalDoGSIFTPipeline` builds its Gaussian and DoG scale spaces with
spherical operations. Extrema are compared in tangent neighbourhoods, so the
detector crosses the ERP seam without creating overlapping faces.

The current descriptor remains hybrid: PanorAi opens one small gnomonic patch
at each accepted bearing and asks OpenCV SIFT to describe its centre. The patch
does not decide where the keypoint exists.

Use this route when constant angular scale and seam continuity matter enough to
justify an Experimental detector. Compare repeatability, high-latitude
behaviour, runtime and pose quality against the Stable face-based route.

## Composition from core operations

The detector is built from spherical Gaussian smoothing, scale changes,
difference images and tangent-neighbourhood extrema. Pointwise subtraction and
thresholding remain ordinary NumPy/Torch operations. Only steps that interpret
spatial neighbourhoods need spherical implementations.

This separation matters: direct spherical detection avoids face-boundary
duplicates, while projected description reuses a mature descriptor without
pretending that SIFT's planar orientation model was rewritten. The output is
still the same feature-set contract used by the projected route, so matching
and pose code do not depend on the detector topology.

## Compare routes fairly

Hold the descriptor, matching policy and pose thresholds fixed. Report
repeatability under longitude rotation, seam-crossing retention, latitude
distribution, descriptors per steradian, runtime and final pose acceptance.
More detected points alone is not evidence of a better spherical detector.

See [composing spherical models](https://rlsgarcia-code.github.io/PanorAi/tutorials/12_composing_spherical_models.html)
for core composition and the
[features tutorial](../../tutorials/03_features_and_matching.md) for the full
comparison. Next: [matching and diagnostics](matching_and_diagnostics.md).
