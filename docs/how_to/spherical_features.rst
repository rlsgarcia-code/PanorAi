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
and 0.15° spherical overlap/match deduplication. Its industrial calibration used
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

Descriptor-neutral tangent patches
-----------------------------------

The Experimental ``SphericalDoGDetector`` first constructs a spherical
Gaussian pyramid, subtracts adjacent levels, and tests each sample against its
26 spatial-and-scale neighbours.  Version 2 then localizes every preliminary
extremum continuously before any descriptor is selected.  In local
east/north/scale coordinates, it fits the second-order model

.. math::

   D(\Delta) = D + g^T\Delta + \frac{1}{2}\Delta^T H\Delta,
   \qquad H\Delta = -g.

The implementation solves the linear system directly rather than forming
``inverse(H)``.  If a component of :math:`\Delta` exceeds half a sample, the
candidate moves to the indicated tangent or scale neighbour and the model is
recomputed, for at most ``refinement_max_iterations``. Singular,
ill-conditioned, non-convergent, and excessively large offsets are rejected.

After convergence, contrast uses the interpolated value

.. math::

   \hat D = D + \frac{1}{2}g^T\Delta,

not the original discrete sample.  Edge rejection is then evaluated at the
refined location using the 2-D east/north Hessian.  A candidate is retained
only when its determinant is positive and

.. math::

   \frac{\operatorname{tr}(H_{xy})^2}{\det(H_{xy})}
   < \frac{(r+1)^2}{r},

where ``r`` is ``edge_threshold``.  These are localization and pruning stages;
orientation assignment and descriptor construction remain separate.

``SphericalKeypointSet`` emits ``SphericalRefinedKeypoint`` instances.  Each
one preserves the compatible position, bearing, response, angular scale,
octave, and integer level fields, and additionally records continuous
``refined_level``, east/north ``tangent_offset_rad``, signed
``interpolated_dog_response``, edge score, Hessian condition number, and the
number of localization iterations.  The detector contract is
``panorai-spherical-dog-detector/v2``.

Experimental spherical detectors expose bearing and angular scale without
choosing a descriptor. Use ``TangentPatchProvider`` to materialize local
visual context separately::

   from panorai.features import TangentPatchProvider, TangentPatchRequest

   request = TangentPatchRequest(
       output_shape_hw=(64, 64),
       radius_in_scales=8.0,
       interpolation="bilinear",
       invalid_policy="propagate",
       minimum_valid_fraction=0.95,
       orientation_policy="upright",
   )
   patches = TangentPatchProvider().materialize(
       panorama, keypoints, request, validity_mask=validity
   )

The request distinguishes angular context (``radius_in_scales``) from raster
sampling (``output_shape_hw``). Each patch records the exact
``GnomonicSpec``, camera matrix, patch-to-panorama basis, observed-data mask,
support mask, detector scale, and realized samples per scale. Per-keypoint
roll is opt-in and must be supplied explicitly.

The same patch set can be passed to SIFT, ORB, or AKAZE through the generic
OpenCV adapter::

   from panorai.features import (
       OpenCVTangentDescriptor,
       OpenCVTangentDescriptorConfig,
   )

   sift = OpenCVTangentDescriptor(
       OpenCVTangentDescriptorConfig(method="sift")
   ).describe(patches)
   orb = OpenCVTangentDescriptor(
       OpenCVTangentDescriptorConfig(
           method="orb",
           algorithm_parameters=(("edgeThreshold", 5), ("patchSize", 31)),
       )
   ).describe(patches)

The detector therefore owns location and angular scale, the patch request owns
context and sampling, and the descriptor adapter owns descriptor semantics.
Changing descriptors does not rerun or reinterpret detection.

Descriptor hypotheses and photometric policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``OpenCVTangentDescriptorV2`` is an Experimental extension for controlled
descriptor ablations. It preserves one physical keypoint identity while
allowing explicit scale or orientation hypotheses::

   from panorai.features import (
       OpenCVTangentDescriptorV2,
       OpenCVTangentDescriptorV2Config,
   )

   hypotheses = OpenCVTangentDescriptorV2(
       OpenCVTangentDescriptorV2Config(
           method="sift",
           root_sift=True,
           keypoint_diameter_in_scales=1.5,
           scale_multipliers=(0.8, 1.0, 1.25),
           orientation_policy="fixed-zero",
           photometric_normalization="robust-percentile-2-98",
           minimum_descriptor_valid_fraction=0.99,
       )
   ).describe(patches)

Every output row records its ``physical_keypoint_id``, patch index, scale
multiplier, orientation rank and confidence, and validity inside the declared
circular descriptor support. Matchers must compare physical keypoints rather
than treating hypotheses of the same point as distinct nearest neighbours.

Photometric normalization is mask-aware and applied only after projection.
``none`` preserves v1 pixels, ``local-standardization`` clips standardized
luminance to three standard deviations, and ``robust-percentile-2-98`` maps
the valid 2nd--98th percentile interval to uint8. These policies do not change
patch geometry and can be used by SIFT, RootSIFT, or ORB. AKAZE is excluded
because arbitrary keypoints do not carry its internal nonlinear scale-space
``class_id``.

For two or more same-shape panoramas, ``SphericalDoGDetector.detect_batch``
shares the spherical Gaussian-pyramid traversal while preserving exactly the
same keypoints as independent ``detect`` calls. This is the recommended path
for a cold image pair. ``TangentPatchProvider(max_workers=N)`` can materialize
independent patches concurrently; ordered mapping preserves keypoint order and
``max_workers=1`` remains the resource-conservative default. These options
change execution only and do not change either public data contract.

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
by the industrial-scanner study.

Select a minimum resolution experimentally
------------------------------------------

The reference profile does not resize an ERP or claim that one raster is
optimal for every camera. ``select_feature_resolution`` accepts observations
that an application has already evaluated at explicit increasing ERP/face
resolutions. Each observation contains the real feature sets, matches,
relative-pose result and optional runtime::

   from panorai.features import (
       ResolutionSelectionObservation,
       ResolutionSelectionPolicy,
       select_feature_resolution,
   )

   observations = tuple(
       ResolutionSelectionObservation(
           erp_shape_hw=level.erp_shape_hw,
           face_shape_hw=level.face_shape_hw,
           features_a=level.features_a,
           features_b=level.features_b,
           matches=level.matches,
           pose=level.pose,
           runtime_seconds=level.runtime_seconds,
       )
       for level in evaluated_levels
   )
   report = select_feature_resolution(
       observations,
       policy=ResolutionSelectionPolicy(target_rotation_accuracy_deg=0.15),
   )

The Experimental ``panorai-resolution-selection/v1`` report compares mutual
bearing repeatability, effective-inlier retention, spherical-coverage
retention and rotation convergence. It selects a lower level only when every
remaining higher-resolution transition belongs to the same converged plateau.
A usable highest level may be returned as ``highest-resolution-fallback``;
that decision always has ``converged_plateau=False``.

Keep masks, preprocessing, feature/matching settings and pose-quality policy
fixed across levels. The selector reports evidence; it does not construct
resized images, choose a dataset-specific ladder, or guarantee translation-
direction accuracy. Promotion requires preregistered cross-domain validation
and an installed-wheel end-to-end consumer.

Direct spherical DoG with SIFT description
-------------------------------------------

``SphericalDoGSIFTPipeline`` is an opt-in Experimental route under
``panorai-spherical-dog-sift/v2``. It builds the Gaussian and difference-of-
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
       )
   )
   matches = pipeline.extract_and_match(panorama_a, panorama_b)

This is the canonical high-level direct-spherical extraction route. Its default
descriptor profile uses one fixed-orientation, locally standardized RootSIFT
hypothesis with a 1.25-scale keypoint diameter. The detector does not first
divide the ERP into virtual cameras. The tangent
projection is descriptor support only, so longitude-seam detections are not
split between faces and do not require overlap deduplication. PanorAi does not
reimplement the SIFT descriptor;
``cv2.SIFT.compute`` remains responsible for its 128 values. NumPy panoramas
are currently required. Use ``convolution_backend="numpy"`` for the reference
path or ``"native"`` when compiled spherical filtering is mandatory.

Detector-only and coarse-to-fine composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detection, tangent-patch materialization, and description are also exposed as
advanced low-level components, not as a second recommended extractor. This allows one spherical keypoint to be
reused with SIFT, ORB, or AKAZE instead of detecting the same scene point in
several overlapping faces. The coarse detector evaluates a low-pass-filtered
proposal ERP, promotes bearings to the source raster, and can re-rank and
locally refine proposals on vectorized native-resolution tangent samples::

   from panorai.features import (
       OpenCVTangentDescriptor,
       OpenCVTangentDescriptorConfig,
       SphericalCoarseDoGDetector,
       SphericalCoarseDoGDetectorConfig,
       TangentPatchProvider,
       TangentPatchRequest,
   )

   detector = SphericalCoarseDoGDetector(
       SphericalCoarseDoGDetectorConfig(
           proposal_height=512,
           proposal_resampling="spherical-gaussian",
           proposal_prefilter_sigma_px=1.0,
           proposal_prefilter_intermediate_height=1024,
           fine_verification="tangent-dog",
           max_keypoints=4096,
       )
   )
   keypoints = detector.detect(panorama, validity_mask=validity)
   patches = TangentPatchProvider().materialize(
       panorama,
       keypoints,
       TangentPatchRequest(output_shape_hw=(48, 48)),
       validity_mask=validity,
   )
   descriptors = OpenCVTangentDescriptor(
       OpenCVTangentDescriptorConfig(method="sift")
   ).describe(patches, responses=keypoints.responses)

The coarse path is an explicit speed/quality trade-off, not the default
detector. Its industrial-scanner development run reduced detector time to about 3.9 seconds
per panorama on the measured CPU, but remained below the dense detector in
repeatability and produced no accepted end-to-end pose under the frozen gate.
Keep the dense detector as the accuracy reference and calibrate proposal
height, antialiasing, and fine verification on an independent development
split before deployment.

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
