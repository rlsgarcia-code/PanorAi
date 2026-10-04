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
       ``panorai.estimators`` relative pose, ``panorai.reconstruction`` global spherical mapper,
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

The multiscale visual-context pipeline and the virtual-camera rig/PyCOLMAP
export methods remain Experimental and have separate inventory entries.
PyCOLMAP export materializes virtual-camera rigs and visual evidence, while
COLMAP remains responsible for SfM. Promoting the core does not promote these
extensions, the relative-pose estimator, or any downstream reconstruction.

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

The installed-wheel VAL-005 census is substantial negative promotion
evidence: with the safe accepted-edge policy, only 92/423 sets (21.7%) were
fully registered, although all 92 complete results met the strict accuracy
criterion. A group-held-out online capture gate raised selected Matterport
success to 17/30 (56.7%), still below unattended-library reliability, and
Stanford contributed only three groups. The current method therefore remains
Experimental with an explicit do-not-promote recommendation. The next gates
are an independent real consumer, prospectively held-out broader-domain
validation, degeneracy coverage, and scale/runtime/failure-envelope evidence.

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
   * - ``spherical-relative-pose``
     - ``panorai-spherical-relative-pose/v1``
     - Experimental
     - External fixtures, prospectively frozen calibration/evaluation corpora,
       cross-domain evidence beyond three Stanford areas, and broader
       degeneracy/performance evidence. VAL-006 is post-hoc supporting evidence.
   * - ``spherical-reconstruction``
     - ``panorai-spherical-reconstruction/v1``
     - Experimental
     - Do not promote the current method: VAL-005 reached 92/423 safe complete
       maps. Next require an independent consumer, prospective broader-domain
       validation, degeneracy coverage and measured scalability.
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
