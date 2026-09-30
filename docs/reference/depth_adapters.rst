Optional depth adapters
=======================

``panorai.depth`` is a compatibility layer, not a bundled model collection.
Importing it registers four loader names without importing Torch, Open3D, or
an upstream implementation:

.. code-block:: python

   from panorai.depth import ModelRegistry

   assert set(ModelRegistry.list_models()) == {
       "dav2", "m3dv2", "dust3r", "zoe"
   }

The ``panorai[depth]`` extra installs common adapter dependencies only. Install
the selected upstream project separately, pin its source/checkpoint for
reproducibility, and review its terms before loading it.

.. list-table:: Adapter boundary
   :header-rows: 1
   :widths: 14 24 25 37

   * - Key
     - PanorAi loader
     - Upstream terms
     - Distribution behavior
   * - ``dav2``
     - ``load_dav2_model``
     - Apache-2.0 code; weights may differ
     - Depth Anything V2 source and checkpoints are not bundled.
   * - ``m3dv2``
     - ``load_m3dv2_model``
     - BSD-2-Clause code; check weights
     - Metric3D source, configs and checkpoints are not bundled.
   * - ``dust3r``
     - ``load_dust3r_model``
     - CC BY-NC-SA 4.0
     - DUSt3R/CroCo are non-commercial/share-alike and never included in the
       MIT PanorAi artifacts.
   * - ``zoe``
     - ``load_zoe_model``
     - Apache-2.0 code; check model card
     - The adapter uses a separately installed Transformers backend.

If an upstream import is unavailable or incompatible, the loader raises
``DepthAdapterUnavailableError`` with the upstream URL and license context.
It never substitutes another model silently. PanorAi does not claim numerical
equivalence until an exact upstream revision and checkpoint have independent
integration evidence.

Deep 3.0 artifact paths below ``panorai.depth.DepthAnythingV2``,
``panorai.depth.Dust3r``, ``panorai.depth.Metric3D`` and
``panorai.depth.ZoeDepth_not_used``, plus the research ``custom_data``,
``trainers`` and ``training`` trees, are experimental/internal and absent from
the 3.1 wheel and sdist.

.. automodule:: panorai.depth
   :members:
   :undoc-members:
