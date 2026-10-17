Pipeline Stages catalog
=======================

A Stage is the smallest named transformation understood by
``panorai.pipeline.Pipeline``.  Public Stages implement ``execute(value)`` and
return ``StageResult(value, details)``.  They can be called through a one-step
Pipeline, nested inside another Pipeline, or composed into an Application.

Stable composition Stages
-------------------------

.. list-table:: Core Stages from ``panorai.pipeline``
   :header-rows: 1
   :widths: 18 18 18 15 31

   * - Stage or factory
     - Input
     - Output
     - Domain
     - Purpose
   * - ``OnSphere``
     - ERP or selected ERP field
     - ERP, ``RasterField``, or declared non-raster result
     - direct sphere
     - Run a registered spherical operation or validated Torch module while transforming grid and masks.
   * - ``SampleViews``
     - ``EquirectangularImage``
     - ``GnomonicFaceSet``
     - ERP → views
     - Sample cube, Fibonacci, icosphere or declared rotated views.
   * - ``PerView``
     - ``GnomonicFaceSet``
     - face set or per-view result
     - views
     - Apply one planar callable independently to each selected view field.
   * - ``PerView.parallel(...)``
     - ``GnomonicFaceSet``
     - mixed ``ResultBundle``
     - views
     - Run named logical branches, optionally batched; no thread-level parallelism is promised.
   * - ``ReconstructERP``
     - face set or bundle containing ``views``
     - ERP or bundle with reconstructed raster fields
     - views → ERP
     - Fuse selected raster modalities and preserve non-raster branches.
   * - ``LiftToBearings``
     - point, region or view observations
     - spherical observations
     - views → sphere
     - Lift non-raster evidence without pretending it is an ERP raster.
   * - ``AggregateViews``
     - per-view scores or embeddings
     - aggregated view evidence
     - spherical support
     - Reduce view evidence with an explicit coverage-aware reducer.
   * - ``Parallel``
     - any value accepted by every branch
     - ``ResultBundle``
     - logical routing
     - Evaluate named branches against the same input in declaration order.
   * - ``Select``
     - nested bundle or typed object
     - selected value
     - logical routing
     - Select a dotted path such as ``views.depth`` or ``erp.labels``.
   * - ``Attach``
     - bundle containing compatible raster data
     - object or bundle with attached field
     - logical routing
     - Attach only grid-compatible fields; reduced lattices require explicit regridding.

``Output`` is a declaration used by ``OnSphere`` and ``PerView``; it is not an
execution Stage.  It gives a callable result an explicit kind, units and
invalid-data policy.

Experimental relative-pose Stages
---------------------------------

.. list-table:: Reusable image-to-pose Stages
   :header-rows: 1
   :widths: 25 22 22 31

   * - Canonical import
     - Input
     - Output
     - Responsibility
   * - ``panorai.applications.AssessRelativePoseCapture``
     - ``RelativePoseCaptureInput``
     - contained ``SphericalImagePair``
     - Apply an explicit pre-estimation eligibility policy and retain the tri-state assessment.
   * - ``panorai.features.ExtractSphericalFeaturePair``
     - ``SphericalImagePair``
     - ``SphericalFeaturePair``
     - Extract features independently with the selected spherical frontend.
   * - ``panorai.features.MatchSphericalFeatures``
     - ``SphericalFeaturePair``
     - ``SphericalFeatureMatches``
     - Match the ordered reference/source feature sets.
   * - ``panorai.features.ToBearingCorrespondences``
     - ``SphericalFeatureMatches``
     - ``SphericalBearingCorrespondences``
     - Convert valid matches to weighted panorama-frame unit bearings.
   * - ``panorai.estimators.EstimateRelativePose``
     - matches or bearing correspondences
     - ``RelativePoseResult`` or ``None``
     - Apply one configured estimator, sampler and acceptance policy.

The complete composition is exposed by
``SphericalRelativePose.as_pipeline()``.  The pose convention is always
source-from-reference and translation is a unit direction, not metric scale.

Experimental semantic-instance Stages
-------------------------------------

.. list-table:: CAM/SAM and instance-fusion Stages from ``panorai.pipeline``
   :header-rows: 1
   :widths: 24 20 20 36

   * - Stage
     - Input
     - Output
     - Responsibility
   * - ``SphericalCAM``
     - ERP
     - ``CAMEvidenceFrame``
     - Produce spherical class-evidence maps from a configured CAM adapter.
   * - ``CAMRegionProposals``
     - CAM evidence
     - ``CAMRegionProposalSet``
     - Turn evidence into angular region proposals.
   * - ``SAMMaskRefinement``
     - proposal set
     - ``SAMMaskProposalBatch``
     - Refine proposals with an explicitly supplied SAM adapter.
   * - ``SphericalInstanceFusion``
     - refined mask proposals
     - spherical segmentation result
     - Fuse masks with spherical support and overlap semantics.
   * - ``CAMGuidedSphericalInstances``
     - CAM evidence and configured SAM adapter
     - spherical segmentation result
     - Convenience composition of proposal, refinement and fusion Stages.

These Stages do not download checkpoints and remain Experimental.  The
trainable ``SphericalMaskRCNN`` topology is a different, deliberately unbound
surface.

See :doc:`Pipeline API <../pipeline>` for signatures and
:doc:`typed composition <../../concepts/pipeline-composition>` for the Stage
contract.
