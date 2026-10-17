Planned 4.0 destination map
===========================

PanorAi 3.8 preserves Stable and Compatibility imports without warnings. It
also provides additive Experimental previews of ``panorai.core`` and
``panorai.workflow`` so the composition contracts can be validated before a
major-version canonical move.

.. list-table:: Planned destinations
   :header-rows: 1
   :widths: 35 65

   * - 3.x surface
     - Planned 4.0 destination
   * - ``data`` object workflow
     - ``workflow`` (additive preview available)
   * - fundamental operations currently spread across ``geometry``,
       ``image_processing``, and learned adapters
     - ``core`` with explicit ``numpy`` and optional ``torch`` backends
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

Existing 3.x imports remain the compatibility boundary. The preview paths
resolve to the same mathematical implementations where migration has already
occurred; they do not authorize removal of an old import before 4.0. Stable
promotion requires independent installed-wheel evidence and an explicit
contract decision.
