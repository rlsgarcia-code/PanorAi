API stability in 3.x
====================

The tier describes support expectations, not how useful an object may be for a
particular research project.

The source-of-truth inventory for this page is
``panorai-public-stability/v1`` in
``tests/fixtures/stability/v1.json``.  Tests import every declared symbol,
freeze exact module exports where applicable, classify every feature export
and registered blender, and require each Experimental surface to retain an
explicit promotion gate.  Native C++ availability is deliberately absent from
the public contract: acceleration must preserve the same Python-visible
semantics and fallback.

.. list-table:: Stability tiers
   :header-rows: 1
   :widths: 18 30 52

   * - Tier
     - Surface
     - Contract
   * - Stable
     - ``panorai.geometry``; ``panorai-object-workflow/v1`` methods;
       ``panorai-spherical-features/v1`` core; supported fusion blenders
     - Tested mathematical and mask contract. Compatible throughout 3.x;
       intentional corrections are documented in migration notes.
   * - Compatibility
     - 3.0 containers, factories, samplers, projection registry, depth
       loader/registry names, PCD names
     - Public names remain through 3.x. They adapt to canonical geometry where
       practical but do not define new geometry behavior.
   * - Experimental
     - ``panorai.features`` multiscale and PyCOLMAP extensions;
       ``panorai.estimators`` relative pose,
       ``panorai.object_localization`` two-view semantic association and
       spatial hypotheses, ``panorai.stereo`` dense radial
       range, ``panorai.reconstruction`` global spherical mapper,
       ``panorai.slam`` central-ERP incremental and calibrated-fisheye visual
       SLAM, and PyCOLMAP export;
       Huber spatial/no-confidence and bundle-adjustment blenders
     - Shape and mask behavior is tested; numerical quality lacks an
       independent reference oracle and the surface may evolve with
       documentation. No 3.0 name is removed.
   * - Frozen/internal
     - Underscore-prefixed geometry implementation modules, deep vendored
       model namespaces, research training/data helpers, and release tooling
     - Public behavior is exposed through the stable API. Internal structure
       may change without creating a second geometry contract.

Depth adapter boundary
----------------------

PanorAi 3.1 retains ``ModelRegistry``, loader names and the ``dav2``,
``m3dv2``, ``dust3r`` and ``zoe`` keys as lightweight compatibility adapters.
Their upstream implementations, training code, datasets and checkpoints are
not included in the wheel or sdist. Deep paths under the formerly vendored
Depth Anything V2, DUSt3R/CroCo, Metric3D and legacy ZoeDepth trees are
experimental/internal and are not a stable 3.x API promise.

Invoking an unavailable adapter gives installation and upstream-license
guidance. PanorAi does not claim numerical equivalence without an independently
tested upstream version and checkpoint. DUSt3R remains subject to its separate
CC BY-NC-SA 4.0 terms and is never covered by PanorAi's MIT license.

Stable blenders
---------------

``average``, ``gaussian``, ``feathering``, ``closest``, and ``huber`` accept
explicit masks and preserve the input image shape. ``counter``, ``std``, and
``std_feathered`` are stable diagnostic maps with ``(H, W, 1)`` output.

Optional backends
-----------------

NumPy is required. Torch and Open3D are optional. Importing
``panorai.geometry`` must not require or load either optional backend. Torch
geometry is stable for input-value autograd on the documented layouts; Open3D
belongs to the compatibility PCD surface.

Deprecation policy
------------------

No stable public 3.0 name is removed before 4.0. This promise does not promote
deep vendored or research implementation files into stable API. Corrections to
broken behavior require regression tests and migration notes.

Stable object workflow
----------------------

``EquirectangularImage.with_depth()``, ``with_labels()``, ``views()`` and
``process_views()``, plus ``GnomonicFaceSet.map()``, ``reconstruct()`` and
``describe()``, form the Stable ``panorai-object-workflow/v1`` surface over the
existing objects. They compose the stable ``geometry-v1`` engine and keep
modality data, geometric support and explicit validity separate. ADOPT-001
validated arbitrary-N multimodal NumPy and Torch reconstruction from an
installed wheel; ADOPT-002 independently validated the same object composition
through features, pose-quality handling and real PyCOLMAP export. Legacy 3.0
container methods retain their Compatibility tier.

Stable spherical feature core
-----------------------------

``panorai.features`` is versioned separately as
``panorai-spherical-features/v1``. OpenCV remains the implementation of SIFT,
ORB, AKAZE, BFMatcher, and FLANN. PanorAi provides geometry, masks, angular
deduplication, provenance, public panorama-domain objects, extraction, and
matching. API-003 established the public contract and OpenCV compatibility;
ADOPT-002 exercised the contract from an installed wheel with a real
Essential consumer and deterministic feature/match evidence. The Stable
promise covers the versioned presets, feature and match result objects,
bearing conversion, masks, deduplication, provenance, extraction, and
matching. It does not make an OpenCV implementation detail part of PanorAi's
contract.

Experimental spherical feature extensions
------------------------------------------

The multiscale visual-context pipeline, direct spherical DoG detector with
tangent-patch SIFT description, resolution-selection report, and virtual-camera
rig/PyCOLMAP export methods remain Experimental and have separate inventory
entries. The direct detector
is versioned as ``panorai-spherical-dog-sift/v2``: scale-space extrema are
detected on the sphere, while OpenCV computes each descriptor on a local
gnomonic patch. It therefore does not reimplement SIFT. The descriptor-free
detector is inventoried separately as ``spherical-features-dog-detector``
under ``panorai-spherical-dog-detector/v2``. Version 2 adds descriptor-neutral
second-order Taylor localization in east/north/scale coordinates, interpolated
contrast rejection, and refined spatial-Hessian edge rejection; emitted
keypoints retain the discrete fields and expose their continuous localization
evidence. Descriptor-neutral patch
materialization is inventoried as ``spherical-features-tangent-patches`` under
``panorai-tangent-patches/v1``; its OpenCV adapter remains part of that
Experimental surface. ``panorai-tangent-opencv-descriptor/v2`` adds explicit
mask-aware photometric policies, effective-support validity, and auditable
scale/orientation hypotheses while retaining physical keypoint identity. V1
remains unchanged. Neither contract promotes a specific descriptor or patch
context as a universal default. This v2 surface is inventoried as
``spherical-features-tangent-descriptor-v2``.
PyCOLMAP export materializes virtual-camera rigs and visual evidence, while
COLMAP remains responsible for SfM. Promoting the core does not promote these
extensions, the relative-pose estimator, or any downstream reconstruction.

The ``spherical-feature-resolution-selection`` inventory entry exposes
``panorai-resolution-selection/v1``. It compares feature, match, coverage and
pose evidence that the caller has already measured at explicit increasing
ERP/face resolutions. It never resizes input automatically. A lower level is
selected only when all later adjacent transitions form a converged plateau;
otherwise a usable highest level is returned as an explicitly non-converged
fallback. Promotion requires preregistered cross-domain validation,
performance evidence and an installed-wheel end-to-end consumer.

Experimental spherical image processing
---------------------------------------

``panorai.image_processing`` is versioned as
``panorai-spherical-image-processing/v1``.  It exposes tangent-plane
convolution and smoothing, canonical ERP transforms, east/north gradients,
Laplacian and Canny edges, Gaussian/Laplacian pyramids, and solid-angle
histogram equalization.  Compatible linear filters may dispatch to a private
C++17 kernel, but ``backend="numpy"`` remains the explicit reference and the
native implementation creates no separate public contract.

This surface remains Experimental.  Analytic seam/pole, closed-form
derivative, native/reference, reconstruction, and OpenCV-compatible planar
equalization tests establish the initial contract.  Promotion additionally
requires broad real-panorama feature/matching evidence, Torch policy/parity,
performance measurements, calibrated Canny and scale-space validation, and an
explicit invalid-data convolution policy.

Experimental spherical deep-learning depth
-------------------------------------------

The ``spherical-deep-learning-depth`` inventory entry is versioned as
``panorai-spherical-metric-depth/v1-experimental``.  The explicit
``panorai.experimental.deep_learning`` namespace acquires one checksum-pinned
external Metric3D-v1 ConvNeXt-Tiny/Hourglass source/checkpoint pair and ports
its learned ``Conv2d`` and ``ConvTranspose2d`` layers into differentiable ERP
tangent sampling.  Parameters are reused by object identity and inference is
resize-free on a caller-provided 2:1 lattice.  Output depth is radial range in
metres, not z-depth.

The adapter is not a model redistribution channel.  Acquisition requires an
explicit upstream-terms opt-in, source and checkpoint remain in an external
cache, and the checkpoint's missing separate model-card license remains a
redistribution blocker.  Promotion requires resolved checkpoint terms,
prospective multi-corpus metric-depth validation, native-resolution
angular-scale and memory evidence, an installed-wheel consumer with an
external cache, and the planned CNN/ViT comparison.  No accuracy result is
asserted by this adapter.  The loader proves
mechanical availability and learned-parameter identity only; it does not prove
useful depth accuracy, semantic equivalence, spherical equivariance, or
scientific success.

Experimental tangent DA3 metric depth
-------------------------------------

The ``tangent-da3-metric-depth`` inventory entry is versioned as
``panorai-depth-anything-3-metric-tangent/v1-experimental``. It acquires the
official Apache-2.0 Depth Anything 3 source and DA3Metric-Large checkpoint at
immutable revisions, with full SHA-256 verification, and keeps both artifacts
outside PanorAi distributions. The adapter bypasses the upstream resizing API:
each caller-provided gnomonic raster must already be patch-aligned and is fed
to the unchanged network at exactly that resolution.

The network output is focal-normalized axial depth. PanorAi scales it by the
mean virtual-camera focal length divided by the official 300-pixel canonical
focal length, then converts axial depth to radial range with the canonical
pixel-centre rays and separate ``fx`` and ``fy``. This conversion does not
clip, resize, smooth, or fuse predictions. Model-output numerical validity is
returned separately and the object workflow intersects it with geometric
support before tangent predictions are reconstructed into an ERP.

This is a reproducibility adapter, not a sphere-native ViT claim. Promotion
requires prospective multi-corpus validation, projection and fusion ablations,
native-resolution memory/runtime evidence, an installed-wheel consumer using
an external cache, and learned overlap fusion evaluated against fixed
blending. The earlier P74 experiment is motivation, not a package accuracy
guarantee.

Experimental spherical InternImage portability
-----------------------------------------------

The ``spherical-internimage-portability`` inventory entry is versioned as
``panorai-spherical-internimage/v1``.  It covers the opt-in InternImage-G
loader, differentiable tangent-plane DCNv3 sampler, recursive spherical port,
and exact dense decomposition of the frozen attention classifier.  The pinned
12.3 GB checkpoint remains in an external user cache after explicit acceptance
of upstream terms; it is not part of the PanorAi source, wheel, or sdist.

This interface is experimental.  Reusing every learned parameter and exactly
reconstructing the global classifier logits from the mean of the dense map are
mechanical equivalence checks, not evidence of semantic localization quality.
Promotion additionally requires the registered gnomonic offset oracle, the
complete frozen P74 evaluation, rotation and polar stress tests, native-size
runtime and peak-memory measurements, and installed-wheel verification.

Experimental spherical semantic segmentation
---------------------------------------------

The ``spherical-semantic-segmentation`` inventory entry is versioned as
``panorai-spherical-semantic-segmentation/v1`` and lives entirely under
``panorai.experimental.deep_learning.segmentation``.  It combines frozen
direct-spherical semantic evidence with transient SAM 2.1 Hiera Large
gnomonic charts, bit-packed spherical masks, fixed multimask acceptance,
per-instance chart expansion, solid-angle fusion, and deterministic panoptic
resolution.  Torch, Transformers, checkpoints, and P74 data remain optional
and external to the source, wheel, and sdist.

``SphericalIndustrialSegmenter`` is the convenience facade for the same
Experimental contract.  Its first prediction acquires the pinned external
InternImage-G and SAM assets, generates direct-spherical evidence, releases
InternImage before loading SAM, and returns the existing
``SphericalSegmentationResult``.  ``SphericalSemanticSegmenter`` remains the
lower-level orchestration API.  The facade accepts canonical ERPs only and does
not promote P74-native raster adaptation into the package contract.

Industrial names are aliases backed by exact ImageNet-1K proxy classes; they
are not trained industrial labels or ground truth.  A proposal without enough
dense semantic evidence remains ``unknown``.  Promotion requires independent
annotated panoramic instance/semantic data, frozen prospective thresholds,
cross-scene evaluation, accuracy and calibration evidence, native-resolution
runtime/memory measurements, and installed-wheel verification with external
model caches.

Experimental spherical relative pose
------------------------------------

``panorai.estimators`` is versioned as
``panorai-spherical-relative-pose/v1``. Its first implementation owns a
polynomial five-correspondence essential kernel, locally optimized robust
consensus, spherical tangent-Sampson residuals, pose refinement, cheirality
selection, competing-model evidence, stability diagnostics, and explicit
acceptance policy. Its raw quality score is not a probability; the isotonic
calibrator requires disjoint calibration and evaluation sample IDs. It
estimates a panorama-frame rotation and unit translation direction only;
translation scale, tracks, triangulation, bundle adjustment and SfM are not
claimed by that pairwise module. Promotion requires external geometric
fixtures, a separately frozen confidence-calibration corpus, real
panorama-pair consumers, broader degeneracy evaluation, and evidence-backed
performance.

VAL-006 measured the existing isotonic calibrator on 2,148 returned poses from
the frozen VAL-002 study, with Matterport360 and Stanford2D3D used as wholly
disjoint calibration/evaluation domains. Matterport-to-Stanford calibration
improved Brier score from 0.225 for the raw score to 0.0766, but still
overpredicted success (27.99% predicted versus 18.71% observed; ECE 0.0928).
The reverse direction produced Brier 0.0957 and ECE 0.0611. Because this was a
post-hoc study, is conditional on a returned pose, and Stanford contributes
only three independent areas, it records an envelope rather than satisfying
the prospective promotion gate. Relative pose remains Experimental.

Experimental semantic object localization
-----------------------------------------

The ``semantic-object-localization`` inventory entry exposes
``panorai-object-localization/v1`` through ``panorai.object_localization``.
This first version requires a textual query resolved to explicit IDs in a
declared vocabulary (``imagenet-1k`` by default), feature-indexed semantic
regions, spherical matches, and a two-view relative pose. It performs a
one-to-one, fail-closed association and returns deterministic two-view object
IDs accompanied by metric, scale-free, or bearing-only spatial hypotheses.

The deterministic ID names one hypothesis supported by exactly two region
observations; it is not yet a persistent multiview entity. Scores are rankings,
not calibrated probabilities, and the triangulated regional feature center is
not claimed to be an object's physical center. The module deliberately has no
graph dependency. Promotion requires real multi-category panoramas, an
independent association/localization benchmark, calibrated uncertainty and
ranking scores, installed-wheel evidence, and a downstream consumer.

The same Experimental surface includes the separately versioned
``panorai-semantic-match-prior/v1`` component. It converts query-compatible,
feature-indexed regions into proposal-only RANSAC weights while retaining a
strictly positive global fallback for every valid match. The component does
not claim that CAM scores are probabilities and does not accept or replace a
pose by itself.

``panorai-semantic-region-proposals/v1`` adds deterministic connected CAM
components with periodic ERP seam connectivity and explicit rejection
diagnostics. These components are region candidates only; they are not
segmentation masks or persistent object identities.

``panorai-joint-match-clusters/v1`` may split one accepted broad semantic
association into paired feature-indexed candidates using angular proximity in
both spherical views. It uses pose inliers by default, preserves deterministic
IDs under match-row reordering, and reports discarded support. The angular
radius is uncalibrated, and connected components may chain through background;
the result remains an object hypothesis input rather than a persistent identity
or instance-segmentation claim.

Experimental spherical dense stereo
-----------------------------------

``panorai.stereo`` is versioned as
``panorai-spherical-dense-stereo/v1-experimental``. It performs direct
inverse-range plane sweep on two same-shape central ERPs, using a supplied
metric relative pose to constrain every candidate to the spherical epipolar
curve. Results expose radial range, validity, uncalibrated confidence, matching
cost and the hypothesis lattice. ERP seam wrapping and A→B→A consistency are
part of the current numerical policy.

The first ten-pair Matterport360/Stanford2D3D study is selected, post-hoc
development evidence. It uses the reference baseline magnitude with estimated
rotation and translation direction and therefore does not validate metric
scale recovery. Promotion requires a prospectively frozen corpus,
independently measured scale, broader illumination/texture/occlusion coverage,
calibrated uncertainty, NumPy/Torch parity or an explicit backend boundary,
and installed-artifact performance evidence. The surface must remain
Experimental until those gates are satisfied.

Promotion requires prospectively frozen, spatially disjoint Matterport3D
buildings and Stanford2D3D areas. See
:doc:`../explanation/spherical_dense_stereo` for the derivation and
evidence-driven accuracy roadmap. Native acceleration is not a promotion gate
until the appearance objective establishes a useful accuracy/coverage envelope.

Experimental global spherical reconstruction
--------------------------------------------

``panorai.reconstruction`` is versioned as
``panorai-spherical-reconstruction/v1``. It accepts PanorAi spherical matches
or precomputed match/pose edges, admits only quality-accepted edges by default,
averages rotations, forms conflict-free tracks, jointly positions cameras and
points with positive latent depths, and runs two-stage spherical bundle
adjustment followed by filtering and retriangulation. Results use
``R_world_to_panorama`` and ``center_world`` and explicitly carry arbitrary
scale. It is independently authored NumPy/SciPy code and does not call
OpenCV/PyCOLMAP geometry. Promotion requires real multiview fixtures, external
consumer evidence, broader degeneracy coverage, and measured scalability.

The metadata-blind 423-set census is substantial negative promotion evidence.
After GEO-011's deterministic multistart positioning, worst-first edge pruning,
and default three-view-track corroboration, the conservative policy produced
131 complete maps and all 131 met the strict accuracy criterion. Coverage was
still only 31.0%. A development-tuned group-held-out Matterport gate selected
18/22 successes (81.8%), but that threshold was not prospectively frozen and
Stanford contributed only three groups. The current method therefore remains
Experimental with an explicit do-not-promote recommendation. The next gates
are an independent real consumer, prospectively held-out broader-domain
validation, degeneracy coverage, and scale/runtime/failure-envelope evidence.

Experimental metric spherical landmark BA
------------------------------------------

``panorai-metric-spherical-landmark-ba/v1-experimental`` refines only an
explicit sparse graph of metric cameras and multiply observed landmarks. Its
monocular radial ranges are uncertainty-weighted soft priors, the metric
baseline is an explicit gauge, and the returned support is always
``supplied-landmarks-only``. It neither estimates dense depth nor promotes a
monocular prediction into hard 3D geometry.

VAL-042 is bounded one-pair development evidence: 69 P74 landmarks improved
from ``0.80721`` to ``0.11585`` AbsRel at those landmark locations while mean
angular residual fell from ``0.10774`` to ``0.09843`` degrees. The experiment
also refined the second camera; “landmark-only” describes the evaluation
support, not every optimized variable. The following dense propagation failed
the held-out/normal Pareto gate, so no dense accuracy claim follows from this
sparse result. Promotion requires prospective multi-scene evidence, a separate
consumer, installed-wheel validation and a calibrated failure envelope.

Experimental incremental spherical SLAM
---------------------------------------

``SphericalIncrementalSLAM`` is versioned as
``panorai-spherical-incremental-slam/v1``. It returns online central-ERP poses,
selects keyframes, triangulates a local spherical map, runs bounded-window
local BA, attempts relocalization, detects old/new keyframe loops, and delegates
accepted global correction to ``SphericalGlobalMapper``. Only geometrically
accepted inlier matches enter 2D--3D landmark tracking. All poses obey
``x_frame = R_world_to_frame (X - C_world)`` and scale is arbitrary.

This surface remains Experimental because validation currently includes
synthetic scenes and one strict-success three-panorama real replay, not a
broad continuous-video corpus. It has no IMU, rolling-shutter model,
dynamic-object model, or marginalization prior, and no real-time performance
promise.

API-006 supplies one accurate metadata-separated three-panorama replay, not a
continuous-video corpus. PERF-003 recorded eight valid three-frame samples but
required roughly 72.8--79.2 seconds per registered frame and 739--762 MiB peak
RSS on its reference machine. No long-loop, relocalization census,
cross-camera/cross-dataset ATE/RPE study, sustained memory-growth bound, or
real-time envelope exists. The current method therefore has an explicit
do-not-promote recommendation; those missing gates must be completed in that
order before another stability decision.

Contract inventory and promotion order
--------------------------------------

The inventory separates a public object from a newly added method surface.  In
particular, the 3.0 containers remain ``legacy-composition`` Compatibility
objects under ``panorai-3x-compatibility/v1`` while the newer methods on those
objects form the ``object-workflow`` Stable surface under
``panorai-object-workflow/v1``.  This promotion does not
reclassify every legacy container behavior as canonical geometry.

.. list-table:: Machine-checked public surfaces
   :header-rows: 1
   :widths: 27 25 15 33

   * - Inventory id
     - Contract
     - Tier
     - Promotion boundary
   * - ``canonical-geometry``
     - ``geometry-v1``
     - Stable
     - Already frozen for 3.x.
   * - ``supported-blenders``
     - ``panorai-blenders/v1``
     - Stable
     - Explicit masks, documented output shapes and finite-value checks.
   * - ``legacy-composition``
     - ``panorai-3x-compatibility/v1``
     - Compatibility
     - Recorded 3.0 names remain available without defining new geometry.
   * - ``object-workflow``
     - ``panorai-object-workflow/v1``
     - Stable
     - Promoted after ADOPT-001/002 installed consumers, 3.x compatibility,
       and PERF-009 arbitrary-N evidence.
   * - ``spherical-features-core``
     - ``panorai-spherical-features/v1``
     - Stable
     - Promoted after API-003 OpenCV compatibility and ADOPT-002 installed-wheel
       Essential-consumer evidence.
   * - ``spherical-features-pycolmap``
     - ``panorai-spherical-features-pycolmap/v1``
     - Experimental
     - A real rig-SfM consumer plus PyCOLMAP compatibility and round-trip
       evidence.
   * - ``spherical-features-multiscale``
     - ``panorai-spherical-features-multiscale/v1``
     - Experimental
     - The current method is not a promotion candidate; a new preregistered
       blind evaluation must first beat the single-scale baseline.
   * - ``spherical-features-dog-sift``
     - ``panorai-spherical-dog-sift/v2``
     - Experimental
     - Real-panorama repeatability and matching benchmarks, a seam/pole
       regression corpus, and native scale-space performance evidence.
   * - ``spherical-image-processing``
     - ``panorai-spherical-image-processing/v1``
     - Experimental
     - Real matching evidence, Torch policy/parity, performance, calibrated
       edge/scale-space validation, and invalid-data filtering semantics.
   * - ``spherical-relative-pose``
     - ``panorai-spherical-relative-pose/v1``
     - Experimental
     - External fixtures, prospectively frozen calibration/evaluation corpora,
       cross-domain evidence beyond three Stanford areas, and broader
       degeneracy/performance evidence. VAL-006 is post-hoc supporting evidence.
   * - ``spherical-reconstruction``
     - ``panorai-spherical-reconstruction/v1``
     - Experimental
     - Do not promote the current method: the current conservative policy
       reached 131/423 complete maps. Next require an independent consumer,
       prospective broader-domain
       validation, degeneracy coverage and measured scalability.
   * - ``metric-spherical-landmark-ba``
     - ``panorai-metric-spherical-landmark-ba/v1-experimental``
     - Experimental
     - Sparse landmark support only. Require prospective multi-scene metric
       evidence, an external consumer, installed-wheel validation and explicit
       separation from any dense-depth claim.
   * - ``spherical-dense-stereo``
     - ``panorai-spherical-dense-stereo/v1-experimental``
     - Experimental
     - Prospective multi-corpus depth validation, independently recovered
       metric scale, calibrated coverage/uncertainty, broader degeneracies,
       backend policy, and installed-artifact performance evidence.
   * - ``spherical-slam``
     - ``panorai-spherical-slam/v1``
     - Experimental
     - Do not promote the current method: one real three-frame replay is
       insufficient. Next require long sequences, loop/relocalization,
       cross-camera data, calibrated ATE/RPE, sustained latency and bounded
       memory.
   * - ``research-blenders``
     - ``panorai-research-blenders/v1``
     - Experimental
     - Independent numerical oracles and a documented accuracy/failure
       envelope.
   * - ``implementation-internals``
     - none
     - Frozen/internal
     - Underscore and native modules may change without creating another
       numerical contract.

Promotion is surface-specific and additive.  It changes the support promise
only after its evidence gate is recorded; it does not make an OpenCV release,
a PyCOLMAP implementation detail, an automatic acceptance threshold, or a
native kernel part of PanorAi's stable numerical contract.  Stable names remain
subject to the 3.x deprecation policy above.
