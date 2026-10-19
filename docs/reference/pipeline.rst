Pipeline API
============

The generic Stage, typed-result, grid, routing, persistence and cache core is
Stable as ``panorai-spherical-pipeline/v1``. CAM/SAM proposal and instance
stages exported by the namespace remain Experimental.

.. panorai-api-components:: stage

Use the :doc:`Stages catalog <catalog/stages>` to choose by accepted input,
produced output, execution domain and role.  Use :doc:`typed data and results
<catalog/data_results>` for the values exchanged between Stages.

Composition
-----------

.. autoclass:: panorai.pipeline.Pipeline
   :members: run, execute, describe, with_params, to_config, from_config, save, load

.. autoclass:: panorai.pipeline.Stage
   :members:

.. autoclass:: panorai.pipeline.StageResult
   :members:

.. py:class:: panorai.pipeline.PipelineRun

   Immutable execution result containing ``output``, retained values, ordered
   stage records and execution metadata. ``describe()`` and ``manifest()``
   return JSON-friendly provenance.

Core stages
-----------

.. py:class:: panorai.pipeline.OnSphere(operation, output="image", input="image")

   Execute a declared sphere-aware operation directly on an ERP and transform
   values, support, validity and grid identity together.

.. autoclass:: panorai.pipeline.SampleViews
   :members: execute

.. autoclass:: panorai.pipeline.PerView
   :members: execute, parallel

.. autoclass:: panorai.pipeline.ReconstructERP
   :members: execute

.. autoclass:: panorai.pipeline.LiftToBearings
   :members: execute

.. autoclass:: panorai.pipeline.AggregateViews
   :members: execute

Sparse relative-pose stages
---------------------------

The Experimental image-to-pose composition uses
``panorai.applications.AssessRelativePoseCapture`` when an explicit capture
policy is configured, followed by
``panorai.features.ExtractSphericalFeaturePair``,
``panorai.features.MatchSphericalFeatures``,
``panorai.features.ToBearingCorrespondences`` and
``panorai.estimators.EstimateRelativePose``. These stages preserve the
ordered source-from-reference convention and return unit translation direction,
not metric scale. They reuse the Stable Pipeline protocol but retain the
Experimental status of relative pose.

The capture Stage consumes ``RelativePoseCaptureInput`` and returns the
contained ``SphericalImagePair``. In enforcing mode, ineligible or unsupported
metadata stops the pipeline before feature extraction. In advisory mode, the
same tri-state assessment is retained in the Stage record without claiming
that capture eligibility guarantees pose correctness.

Routing
-------

.. autoclass:: panorai.pipeline.Parallel
   :members: execute

.. autoclass:: panorai.pipeline.Select
   :members: execute

.. autoclass:: panorai.pipeline.Attach
   :members: execute

Typed values
------------

.. autoclass:: panorai.pipeline.Output
   :members:

.. autoclass:: panorai.pipeline.RasterGrid
   :members: describe

.. autoclass:: panorai.pipeline.RasterField
   :members: describe

.. autoclass:: panorai.pipeline.ResultBundle
   :members: describe

.. autoclass:: panorai.pipeline.ClassificationResult
   :members: describe

.. autoclass:: panorai.pipeline.EmbeddingResult
   :members: describe

.. autoclass:: panorai.pipeline.SphericalObservations
   :members: describe

Start with :doc:`the Pipeline tutorial <../guides/api-levels/pipeline>` and
read :doc:`Typed pipeline composition <../concepts/pipeline-composition>` for
the data model.
