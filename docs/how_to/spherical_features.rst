Spherical features and matching
===============================

.. note::

   The extraction and matching core of ``panorai.features`` is Stable as
   ``panorai-spherical-features/v1``. OpenCV owns feature detection,
   descriptors, and nearest-neighbour matching; PanorAi owns spherical
   geometry, masks, deduplication, result objects, provenance, and their
   versioned orchestration. Direct spherical DoG detection, multiscale routing,
   and virtual-rig/PyCOLMAP export remain Experimental extensions.

Install and use the façade
--------------------------

OpenCV is already present in the 3.x compatibility installation. The
``features`` extra is an explicit installation alias, and PyCOLMAP export is
separate::

   pip install "panorai[features]"
   pip install "panorai[features,pycolmap]"

The versioned presets require OpenCV ``>=4.9,<5``. The 4.9 minimum lets the
AKAZE preset declare ``max_points`` explicitly instead of inheriting a
version-dependent backend default. OpenCV 5 moved AKAZE out of the standard
headless distribution, so PanorAi keeps the backend below 5 rather than
silently dropping the ``akaze-hamming`` preset.

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

For spherical relative pose, prefer the complete reference profile instead of
assembling its settings manually::

   pipeline = SphericalFeaturePipeline.for_relative_pose()
   matches = pipeline.extract_and_match(
       panorama_a,
       panorama_b,
       validity_mask_a=valid_a,
       validity_mask_b=valid_b,
   )

The v1 profile fixes cube/95°/1024² sampling, SIFT/FLANN, 4096 features, Lowe
ratio 0.72, a 16-pixel face-edge margin, 1.5× scale-aware validity exclusion,
and 0.15° spherical overlap/match deduplication. Its P74 calibration used
4096×2048 ERPs and no CLAHE. PanorAi does not silently resize the source or
infer masks from black pixels; preserve real acquisition validity and validate
resolution on the deployment domain.

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
from color, zero, response, or descriptor value. The relative-pose reference
profile sets ``validity_scale_margin=1.5`` and requires each keypoint center to
be farther than 1.5 times its OpenCV scale from invalid support and the virtual
face border. General presets retain ``0.0`` because ORB/AKAZE need independent
calibration. ``validity_margin_px`` can add a fixed sensor-specific erosion.
Overlap duplicates are removed by deterministic angular non-maximum
suppression: highest response wins and source order breaks exact ties.
Alternative face IDs, descriptor rows, angular distances, and the selection
reason remain in provenance.

Matching and spherical estimation
----------------------------------

``matches.to_bearing_correspondences()`` returns aligned
``bearings_a``, ``bearings_b``, validity, and deliberately uniform weights.
Descriptor distances from different descriptor families are not treated as a
universal scientific confidence score.

Match deduplication is bilateral in spherical geometry: a lower-distance
correspondence suppresses another only when both their A-side bearings and
their B-side bearings are within ``angular_dedup_threshold_deg``. This differs
from ``cross_check``, which reruns descriptor search B→A and requires a mutual
nearest-neighbour assignment. The relative-pose reference profile enables the
bilateral NMS and keeps ``cross_check=False`` to preserve the recall validated
by the P74 study.

Direct spherical DoG with SIFT description
-------------------------------------------

``SphericalDoGSIFTPipeline`` is an opt-in Experimental route under
``panorai-spherical-dog-sift/v1``. It builds the Gaussian and difference-of-
Gaussian scale spaces with constant-angle spherical convolution, tests extrema
against tangent neighbours on adjacent scales, and reports keypoint scale in
degrees. It then creates exactly one gnomonic patch around each accepted
bearing and asks OpenCV to compute the SIFT descriptor at the patch centre::

   from panorai.features import SphericalDoGSIFTConfig, SphericalDoGSIFTPipeline

   pipeline = SphericalDoGSIFTPipeline(
       SphericalDoGSIFTConfig(
           octaves=3,
           levels_per_octave=3,
           max_features=1000,
           root_sift=False,
       )
   )
   matches = pipeline.extract_and_match(panorama_a, panorama_b)

The detector does not first divide the ERP into virtual cameras. The tangent
projection is descriptor support only, so longitude-seam detections are not
split between faces and do not require overlap deduplication. PanorAi assigns
the dominant tangent orientation but does not reimplement the SIFT descriptor;
``cv2.SIFT.compute`` remains responsible for its 128 values. NumPy panoramas
are currently required. Use ``convolution_backend="numpy"`` for the reference
path or ``"native"`` when compiled spherical filtering is mandatory.

Multiscale visual-context routing
---------------------------------

``MultiscaleSphericalFeaturePipeline`` is an opt-in Experimental workflow for
cases where one fixed virtual-camera FOV is insufficient.  It does not require
semantic segmentation or class labels.  Instead it constructs a graph of
visual-context nodes:

1. materialize the configured wide views;
2. sample lower-FOV candidates inside each wide view;
3. compare wide and local embeddings, and compare high/low raster density on
   the *same* local angular support;
4. retain the most novel local views;
5. detect OpenCV keypoints on the retained wide/local views and run one
   spherical NMS across scales;
6. route additional OpenCV descriptor matching through mutually similar graph
   nodes while retaining the complete global match as a fallback.

The embedding graph never decides geometric inliers.  Every global and routed
correspondence remains visible to the spherical estimator.  Embedding and
cross-scale evidence only alter proposal mass inside RANSAC, through the
``weights`` returned by ``matches.to_bearing_correspondences()``::

   from panorai.features import (
       MultiscaleEmbeddingConfig,
       MultiscaleSphericalFeaturePipeline,
   )

   pipeline = MultiscaleSphericalFeaturePipeline.from_preset(
       "sift-flann",
       face_sampler="icosahedron",
       face_fov_deg=80.0,
       face_overlap_deg=15.0,
       face_shape_hw=(512, 512),
       multiscale_config=MultiscaleEmbeddingConfig(
           local_fov_deg=(42.0, 42.0),
           local_grid_size=2,
           max_local_views_per_root=2,
       ),
   )

   matches = pipeline.extract_and_match(panorama_a, panorama_b)
   pose = estimator.estimate(matches.to_bearing_correspondences())

The included ``OpenCVContextEmbedding`` is deterministic and has no weights or
downloads.  It combines coarse colour, luminance, gradient and DCT evidence so
the routing mechanics can be reproduced in a clean installation.  It is **not**
advertised as a learned semantic representation.  Research users can inject a
learned DINO/CLIP-style provider by implementing ``VisualEmbeddingProvider``;
the provider must return finite ``(N, D)`` floating vectors and declare a
``name`` and ``version``.  PanorAi does not vendor model weights or silently
download them.

``feature_set.describe()`` records every candidate and selected node, both
similarity measurements, solid angle, keypoint density normalized by solid
angle, embedding checksum and feature count.  ``matches.describe()`` records
how many correspondences came from the global fallback, regional routing or
both.  This provenance is required because the thresholds are a research
policy rather than a stable universal calibration.

Two cautions matter when interpreting this first implementation.  A dense
small-FOV view is evidence that the local scale may be useful, not proof that
the scene is important.  Also, matching improvements on a dataset whose
references were already inspected are post-hoc evidence; promotion requires a
new frozen, outcome-blind set.

The built-in reference provider is currently **mechanism evidence, not an
accuracy recommendation**.  In the recorded post-hoc 2,340-pair study,
``opencv-context/v1`` increased returned poses from 2,148 to 2,221 but reduced
the primary success count from 824 to 660.  Matterport360 changed from
740/1,890 to 618/1,890 and Stanford2D3D from 84/450 to 42/450.  The regional
route added too many visually plausible but geometrically wrong local matches;
therefore it is not enabled by ``SphericalFeaturePipeline`` and must not be
presented as more robust than the single-FOV baseline.  The next evaluation
should separately ablate local-view extraction, graph routing and proposal
weights, and should use a learned provider with calibrated node-pair
acceptance before a new independent blind test.

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

The recorded VAL-006 transfer study quantifies why that separation matters.
On returned poses, an isotonic calibrator fitted on Matterport360 and evaluated
on disjoint Stanford2D3D samples reduced Brier score from 0.225 (treating the
raw score as a probability) to 0.0766. It still overpredicted the Stanford
success rate: 27.99% predicted versus 18.71% observed, with ECE 0.0928. The
reverse Stanford-to-Matterport direction reached Brier 0.0957 and ECE 0.0611.
These are post-hoc domain-transfer bounds, not default probabilities: the
population is conditional on a returned pose and Stanford contains only three
independent areas. Keep calibration and evaluation IDs disjoint and calibrate
again for the deployment domain.

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
