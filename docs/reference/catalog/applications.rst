Applications catalog
====================

Applications are immutable, task-level wrappers.  They accept the natural
inputs of a task, expose parameters through ``get_params()`` and
``with_params()``, describe their implementation, and reveal their lower-level
composition through ``as_pipeline()``.  Start here unless you need to replace
one of those internal decisions.

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
   * - ``panorai.applications.SphericalMaskRCNNWorkflow``
     - Experimental
     - one ERP plus configured CAM/SAM adapters and instance heuristic
     - ``SphericalSegmentationResult``
     - spherical evidence + SAM masks
     - CAM or SAM proposals, mask refinement/fusion, heuristic BoxHead

``SphericalMaskRCNNPaper`` remains an import-compatible alias for the original
Experimental name. ``mask_rcnn.paper()`` delegates to the canonical
``mask_rcnn.workflow()`` factory with a deprecation warning.

All five wrappers run directly with ``application(...)`` or
``application.run(...)``.  ``as_pipeline()`` is the supported escape hatch:
inspect it, retain intermediate results, or reuse individual public Stages.
Internal adapter Stages whose names start with ``_`` are implementation
details and are not catalog entries.

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
