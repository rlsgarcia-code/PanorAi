Pipeline API
============

The generic Stage, typed-result, grid, routing, persistence and cache core is
Stable as ``panorai-spherical-pipeline/v1``. CAM/SAM proposal and instance
stages exported by the namespace remain Experimental.

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
