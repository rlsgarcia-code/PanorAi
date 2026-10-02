Spherical features and matching
===============================

.. note::

   ``panorai.features`` is Experimental after 3.2. It is versioned as
   ``panorai-spherical-features/v1``. OpenCV owns feature detection,
   descriptors, and nearest-neighbour matching; PanorAi owns spherical
   geometry, masks, deduplication, result objects, and provenance.

Install and use the façade
--------------------------

OpenCV is already present in the 3.x compatibility installation. The
``features`` extra is an explicit installation alias, and PyCOLMAP export is
separate::

   pip install "panorai[features]"
   pip install "panorai[features,pycolmap]"

The versioned presets require OpenCV 4.9 or newer. This lets the AKAZE preset
declare ``max_points`` explicitly instead of inheriting a version-dependent
backend default.

The normal path never requires importing ``cv2`` and never returns
``cv2.KeyPoint`` or ``cv2.DMatch`` objects:

.. literalinclude:: ../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_FEATURES_START = None
   :end-before: DOCS_FEATURES_END = None
   :dedent: 4

The versioned presets are ``sift-flann``, ``sift-bf``, ``orb-hamming``, and
``akaze-hamming``. Each freezes the detector, descriptor, distance metric,
matcher, ratio policy, minimum OpenCV version, and scalar algorithm
parameters. ``pipeline.describe()`` is JSON-serializable.

NumPy ``HW``/``HWC`` and Torch ``HW``/``CHW`` panoramas are accepted. OpenCV
runs on contiguous CPU images, so Torch inputs are explicitly transferred to
CPU for detection and the public feature/descriptor results are NumPy arrays;
this path does not claim differentiability. Batched extraction is deliberately
not implicit: give each panorama its own ID and extraction call.

Face geometry and bearings
--------------------------

Features are detected on gnomonic virtual cameras and represented by unit
bearings in the panorama frame. Each feature retains its face pixel,
optional ERP source pixel, ``GnomonicSpec``, response, scale, angle, octave,
descriptor row, source checksum, and duplicate provenance.

The public geometry helpers are vectorized::

   from panorai.geometry import (
       gnomonic_pixels_to_rays,
       rays_to_gnomonic_pixels,
   )

   projected = gnomonic_pixels_to_rays(pixels_xy, face.spec)
   recovered = rays_to_gnomonic_pixels(projected.rays_xyz, face.spec)

``face.geometry`` exposes ``K``, ``R_panorama_from_face``, the support mask,
and the resolved specification. Use
``face.get_geometry(include_source_pixels=True)`` to include the exact
face-to-ERP map used by rasterization.

``K`` uses image coordinates with x right and y down. The panorama frame uses
``+Y`` up, so ``R_panorama_from_face`` is an orthogonal direction transform
with determinant ``-1``; this is intentional, not a pose in ``SO(3)``.
Relative transforms between two face cameras are proper rotations. The
PyCOLMAP exporter writes those relative rotations and reports the reference
face transform needed to recover panorama-frame bearings.

Masks and overlap deduplication
-------------------------------

The extractor combines geometric support, data validity, an optional caller
mask, and ``edge_margin_px`` before invoking OpenCV. No validity is inferred
from color, zero, response, or descriptor value. Overlap duplicates are
removed by deterministic angular non-maximum suppression: highest response
wins and source order breaks exact ties. Alternative face IDs, descriptor
rows, angular distances, and the selection reason remain in provenance.

Matching and spherical estimation
----------------------------------

``matches.to_bearing_correspondences()`` returns aligned
``bearings_a``, ``bearings_b``, validity, and deliberately uniform weights.
Descriptor distances from different descriptor families are not treated as a
universal scientific confidence score.

Use those bearings for spherical epipolar estimation. Do not pass a mixture
of 2D pixels from different virtual cameras directly to
``cv2.findEssentialMat``. The relevant constraint is
``b2.T @ skew(t) @ R @ b1 = 0`` in the panorama frames.

PanorAi now provides an isolated Experimental first implementation that does
not call OpenCV or PyCOLMAP for geometry::

   from panorai.estimators import (
       RelativePoseOptions,
       SphericalRelativePoseEstimator,
   )

   estimator = SphericalRelativePoseEstimator(
       RelativePoseOptions(
           max_angular_error_deg=0.5,
           random_seed=7,
       )
   )
   pose = estimator.estimate(matches.to_bearing_correspondences())
   if pose is not None:
       R_panorama2_from_panorama1 = pose.R
       translation_direction = pose.t
       inliers = pose.inlier_mask

The estimator uses five-correspondence essential hypotheses, locally optimized
RANSAC, spherical tangent-Sampson scoring, nonlinear refinement on rotation
and translation direction, and cheirality selection. The minimal kernel now
constructs the calibrated cubic constraints in the four-dimensional nullspace
and enumerates their real roots through an action matrix; a numerical root
search remains an explicit fallback for singular charts.
``max_angular_error_deg`` is an approximately angular threshold, independent
of ERP or face pixel density.
Its default spatially weighted sampler operates inside RANSAC: it proposes
each five-point minimal set using coverage in both panoramas, but every valid
correspondence is still scored and remains eligible to become an inlier. The
proposal relaxes its spatial constraints and has a uniform fallback rather
than filtering clustered correspondences before RANSAC. Proposal diagnostics
are available as ``pose.sampling_diagnostics``.

Do not treat every returned pose as trustworthy. Inspect
``pose.quality_report.accepted`` and ``rejection_reasons``. The report combines
angular coverage, continuous residual quality, parallax, cheirality,
independent consensus re-estimation stability, and
Essential-versus-rotation/projective
model competition. The pose is still returned when rejected so research code
can audit the evidence. ``raw_quality_score`` is only a ranking value; use the
leakage-aware calibrator with a distinct labeled dataset before interpreting a
value as a probability.

Only ``R`` and the unit direction of ``t`` are observable. Metric translation
scale, tracks, triangulation, bundle adjustment and SfM are outside the
pairwise estimator. Low-parallax solutions are returned with ``degenerate=True`` and an
explicit reason so callers can retain the evidence without silently treating
the pose as well conditioned. The implementation is versioned as
``panorai-spherical-relative-pose/v1`` and is intended to be improved before
promotion from Experimental.

For three or more panoramas, the separate Experimental
``panorai.reconstruction.SphericalGlobalMapper`` can consume a sequence of
these match objects. It admits quality-accepted pairwise poses by default,
performs global rotation averaging, builds tracks, jointly positions cameras
and points, and refines an arbitrary-scale reconstruction with spherical
bundle adjustment. See :doc:`../reference/reconstruction` for frames, gauges,
failure semantics and limitations.

PyCOLMAP export
---------------

Build a virtual rig with ``pipeline.build_virtual_camera_rig(panorama)`` and
call ``pipeline.export_pycolmap(...)`` with matching feature sets. PanorAi
writes PINHOLE cameras, fixed zero-baseline rig relations, frames, images,
keypoints, descriptors, and optional matches. This remains the downstream
route for COLMAP tracks, registration, triangulation and bundle adjustment.
The independent PanorAi global mapper is an alternative Experimental research
route; neither implementation is presented as a numerical substitute for the
other.

OpenCV SIFT descriptors are losslessly converted from their integer-valued
``float32`` representation to COLMAP's ``uint8`` database representation.
Other floating descriptor families fail explicitly; binary ORB and AKAZE
descriptors are preserved as bytes.

PanorAi feature pixels use first-pixel centre ``(0, 0)``. COLMAP stores the
upper-left image corner at ``(0, 0)`` and therefore the first pixel centre at
``(0.5, 0.5)``. Export adds ``0.5`` to both keypoints and camera principal
points, preserving every normalized camera ray while conforming to COLMAP's
database convention.

The current exporter targets PyCOLMAP 3.13 or newer and is integration-tested
with PyCOLMAP 4.2.1. It creates evidence in a database; it does not start SfM
or silently optimize the virtual-camera extrinsics. To avoid corrupting or
partially merging unrelated evidence, the output database path must not
already exist.

Advanced OpenCV injection
-------------------------

Advanced users can import and construct ``OpenCVFeatureBackend(extractor=...,
matcher=...)`` or call ``extract_opencv_features`` and
``match_opencv_features``. These routes accept OpenCV-compatible objects but
still convert their results immediately into PanorAi-owned arrays and
spherical result objects. They are not a second implementation of SIFT, ORB,
AKAZE, BFMatcher, or FLANN.
