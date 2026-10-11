Planned 4.0 destination map
===========================

PanorAi 3.7 documents these destinations without moving imports or emitting
new warnings for Stable or Compatibility surfaces.

.. list-table:: Planned destinations
   :header-rows: 1
   :widths: 35 65

   * - 3.x surface
     - Planned 4.0 destination
   * - ``data`` object workflow
     - ``workflow``
   * - ``projections``
     - ``geometry``
   * - ``samplers``
     - ``workflow.layouts``
   * - ``blenders``
     - ``workflow.fusion``
   * - ``preprocessing``
     - ``image_processing`` or explicit workflow recipes
   * - ``config``, ``factory``, global registries
     - Immutable configurations and explicit constructors
   * - ``pcd``
     - Optional ``io.pointcloud`` adapter
   * - ``object_localization``
     - ``graph.association`` and ``graph.localization``
   * - ``geometry``, ``features``, ``estimators``, ``reconstruction``,
       ``stereo``, ``slam``
     - Remain domain packages

These are architectural destinations, not 3.7 aliases. The 3.x import tree is
the compatibility boundary until a separately authorized major release.
