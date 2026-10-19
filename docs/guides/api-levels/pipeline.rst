Pipeline: compose the method
============================

Choose Pipeline when the task needs a different sequence, execution domain or
result topology. A Pipeline is an inspectable sequence of public ``Stage``
objects and is itself a Stage, so compositions can be nested.

.. panorai-api-components:: stage

1. Execute directly on the sphere
---------------------------------

``OnSphere`` applies a declared sphere-aware operation to an ERP. No tangent
faces or fusion step are created.

2. Process projected views
--------------------------

``SampleViews`` creates canonical gnomonic views, ``PerView`` applies ordinary
view-local callables, and ``ReconstructERP`` fuses raster outputs. The same
Pipeline can retain non-raster branches such as detections.

The executable example below demonstrates both routes and ends with a mixed
``ResultBundle``:

.. literalinclude:: ../../../scripts/run_documentation_examples.py
   :language: python
   :start-after: DOCS_LEVEL_PIPELINE_START = None
   :end-before: DOCS_LEVEL_PIPELINE_END = None
   :dedent: 4

.. image:: ../../_static/readme/hybrid-pipeline.svg
   :alt: One typed Pipeline branching into direct spherical and projected-view
         processing before retaining and fusing results.
   :width: 100%

3. Add semantics before routing
-------------------------------

``Output`` declares whether a callable returns labels, radial-range depth,
detections, classification or another supported family. Raster branches carry
``RasterGrid``; view observations retain face identity, projection, support
and validity. ``Select``, ``Attach`` and ``Parallel`` route these named values
without guessing their meaning.

Labels and masks therefore reconstruct with nearest sampling. Depth requires
units and may opt into mathematically defined validity renormalization. A
generic callable never silently changes either rule.

4. Choose batching deliberately
--------------------------------

``batch=None`` executes one view at a time. A positive integer fixes a
deterministic chunk size. ``batch="auto"`` is accepted only for callables that
declare batch support and uses a bounded element budget. PanorAi never retries
silently after out-of-memory failure.

5. Inspect, retain and persist
------------------------------

Use ``pipeline.run(value, retain=(...))`` for a ``PipelineRun`` containing
output, selected intermediates and stage records. ``describe()`` exposes the
topology. ``to_config()`` is available only when every component is registered;
cache identity includes values, masks, grids, configuration and registered
weight identity.

Read :doc:`Typed pipeline composition <../../concepts/pipeline-composition>`
for the data model and :doc:`Pipeline API <../../reference/pipeline>` for exact
signatures. The larger :doc:`composed API guide <../composed_spherical_api>`
covers persistence, cache and the Experimental CAM/SAM stages.
