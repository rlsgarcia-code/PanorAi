Applications catalog
====================

Applications are immutable, task-level wrappers.  They accept the natural
inputs of a task, expose parameters through ``get_params()`` and
``with_params()``, describe their implementation, and reveal their lower-level
composition through ``as_pipeline()``.  Start here unless you need to replace
one of those internal decisions.

Registered Application surfaces
-------------------------------

This table is generated from the public documentation contract.  A new
Application cannot disappear from the catalog merely because a hand-written
table was not updated.

.. panorai-api-components:: application

Concrete applications
---------------------

.. list-table:: Public task wrappers
   :header-rows: 1
   :widths: 18 9 19 19 16 19

   * - Application
     - Stability
     - Input
     - Output
     - Domain
     - Exposed composition
   * - ``panorai.applications.SphericalSLIC``
     - Stable
     - one canonical ERP with explicit validity
     - ``RasterField(kind="labels")``
     - direct sphere
     - ``OnSphere(Output(SLIC(), kind="labels"))``
   * - ``panorai.applications.SphericalDoGFeatures``
     - Experimental
     - one canonical ERP
     - ``SphericalFeatureSet`` with tangent descriptors
     - sphere + tangent patches
     - one task Stage using ``SphericalDoGSIFTExtractor``
   * - ``panorai.applications.SphericalRelativePose``
     - Experimental
     - ordered reference/source ERPs; optional capture metadata
     - ``RelativePoseResult`` or ``None``
     - features → bearings → spherical pose
     - capture assessment, extraction, matching, bearing conversion, pose estimate
   * - ``panorai.applications.SphericalStereoDepth``
     - Experimental
     - two ERPs plus metric source-from-reference pose
     - ``SphericalStereoResult`` with radial range, confidence and validity
     - direct-spherical plane sweep
     - one typed stereo Stage over ``SphericalDenseStereo``
   * - ``panorai.applications.FullSphereDenseStereo.reviewed_profile()``
     - Stable
     - registered ERPs with supplied metric poses
     - radial range with separate support, observability and validity
     - overlapping atlas + multibaseline spherical warping
     - exact reviewed profile; custom options remain Experimental
   * - ``panorai.applications.SphericalDenseReconstruction``
     - Experimental
     - registered central ERPs with supplied metric world poses
     - accepted/rejected result with mesh, normals, evidence and optional color
     - multireference MVS → world TSDF → surface
     - six public Stages from reference-set construction through validation
   * - ``panorai.applications.SphericalMaskRCNNWorkflow``
     - Experimental
     - one ERP plus configured CAM/SAM adapters and instance heuristic
     - ``SphericalSegmentationResult``
     - spherical evidence + SAM masks
     - CAM or SAM proposals, mask refinement/fusion, heuristic BoxHead
   * - ``panorai.applications.PanoramaStitcher``
     - Experimental
     - :math:`N \geq 2` overlapping planar pinhole images; optional support,
       validity and horizontal FOV
     - partial ``EquirectangularImage``, recovered rotations/intrinsics,
       contribution maps and diagnostics
     - planar evidence → central SO(3) registration → ERP fusion
     - feature detection, overlap discovery, calibration/registration, warp,
       exposure compensation and blend; ``robust_rotation_profile()`` adds
       selective weak-edge recovery while remaining Experimental

``SphericalMaskRCNNPaper`` remains an import-compatible alias for the original
Experimental name. ``mask_rcnn.paper()`` delegates to the canonical
``mask_rcnn.workflow()`` factory with a deprecation warning.

All wrappers run directly with ``application(...)`` or
``application.run(...)``.  ``as_pipeline()`` is the supported escape hatch:
inspect it, retain intermediate results, or reuse individual public Stages.
Internal adapter Stages whose names start with ``_`` are implementation
details and are not catalog entries.

DoG reviewed-profile candidate
------------------------------

``SphericalDoGFeatures.reviewed_profile()`` is inventoried separately as
``panorai-spherical-dog-features/v1``. It freezes the dense spherical DoG
detector, tangent-patch support rules and RootSIFT descriptor. Changing a
scientific parameter with ``with_params()`` deliberately returns the generic
Experimental wrapper and removes the profile identity. Selecting ``numpy``,
``native`` or ``auto`` convolution changes only the execution backend and
preserves the identity.

The named profile is still Experimental. Its separate catalog row describes
the remaining scene-disjoint external evaluation, installed-wheel performance,
Python/platform matrix and independent-review gates. The lower-level generic
DoG detector, coarse-to-fine detector and free-form application configuration
do not inherit a future promotion of this factory.

Registered stereo reviewed profile
----------------------------------

``FullSphereDenseStereo.reviewed_profile()`` is the Stable
``panorai-registered-full-sphere-stereo/v1`` entry point. It accepts only an
explicit ``RegisteredPanoramaSet`` whose source transforms have metric scale.
The general engine, direct construction with custom options, automatic pose,
scale recovery, multiscale AD-Census, tMGM and probability calibration remain
Experimental and keep their separate catalog records.

Stitching robust-rotation v2 profile
------------------------------------

``PanoramaStitcher.robust_rotation_profile()`` is inventoried separately as
``panorai-panorama-stitching/robust-rotation-v2-experimental``. It preserves
the frozen reviewed-v1 path, retries only descriptor-weak pairs, requires
resolution-normalized feature evidence, and exposes accepted recovery edges in
the result diagnostics. Every recovered edge still passes deterministic
homography and calibrated central-rotation gates.

Its analytic and HuginSample results are development regressions. The profile
remains Experimental and requires a new scene-disjoint prospective corpus,
installed-artifact evidence and independent review before any Stable claim.

Framework contracts
-------------------

.. list-table:: Application framework
   :header-rows: 1
   :widths: 26 15 59

   * - Component
     - Stability
     - Responsibility
   * - ``panorai.applications.Application``
     - Experimental
     - Common call, parameter, description and safe-configuration contract.
   * - ``panorai.applications.PipelineApplication``
     - Experimental
     - Application whose executable lower-level form is a public ``Pipeline``.
   * - ``panorai.applications.RelativePoseProfile``
     - Experimental
     - Named, immutable bundle of frontend, estimator, sampler and acceptance policy.
   * - ``panorai.applications.RelativePoseCapturePolicy``
     - Experimental
     - Pre-estimation capture eligibility; it does not estimate pose or guarantee success.

Persistence is available only when every nested component has a registered,
safe representation.  In-memory callables remain executable but are not made
serializable by the wrapper.  No application silently downloads weights or
accepts third-party terms.

Follow :doc:`the Applications tutorial <../../guides/api-levels/applications>`,
read :doc:`the wrapper concept <../../concepts/application-wrappers>`, or use
the signature-level :doc:`Applications API <../applications>`.
