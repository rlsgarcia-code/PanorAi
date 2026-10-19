Typed pipeline composition
==========================

A Pipeline describes how spherical data changes domain and meaning. It is more
than a list of callables: every Stage declares accepted and produced kinds, and
every spatial result carries the grid, support and validity needed by the next
stage.

.. panorai-api-components:: stage

Two execution domains
---------------------

``OnSphere`` keeps computation on an ERP or derived spherical lattice.
``SampleViews`` crosses into gnomonic views, ``PerView`` runs local planar
callables and ``ReconstructERP`` crosses back through explicit fusion. Hybrid
pipelines can branch across both domains and retain their results together.

Typed results prevent accidental semantics
------------------------------------------

``RasterField`` binds values to a ``RasterGrid`` and declares whether they are
image, depth, labels, masks, logits, features or confidence. Classification,
embeddings, detections and keypoints use non-raster result types that retain
their view and projection provenance. ``ResultBundle`` holds named branches
without flattening these distinctions.

This typing makes invalid operations fail early. Labels cannot be bilinearly
reconstructed, depth cannot omit units, and a downsampled feature lattice
cannot be attached as though it were the canonical ERP grid.

Execution remains inspectable
-----------------------------

``StageResult`` carries a value and execution details. ``PipelineRun`` records
each transition and may retain selected intermediates. A Pipeline implements
the same Stage protocol, enabling nesting without a second execution model.
Logical branching does not promise threads; deterministic batching is an
explicit adapter capability.

Persistence and cache are fail-closed
-------------------------------------

Topology serialization uses registered identifiers rather than Python imports
named by a file. Torch state uses checksum-bound safetensors for registered
architectures. Cache keys include values, masks, grids, configuration, version
and weight identity, and tensors requiring gradients are never silently
detached into cache.

See :doc:`the Pipeline tutorial <../guides/api-levels/pipeline>` and
:doc:`Pipeline API <../reference/pipeline>`.
