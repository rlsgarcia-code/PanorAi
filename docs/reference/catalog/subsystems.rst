Subsystems and advanced surfaces
================================

Not every public component is naturally an Application, Stage, operation or
estimator.  This page routes the remaining audited surfaces by computer-vision
responsibility.  Follow the linked reference for signatures, the linked guide
for a workflow, and the :doc:`complete public inventory <public_surfaces>` for
every exported symbol and exact stability contract.

Foundations and processing
--------------------------

.. list-table:: Foundational subsystems
   :header-rows: 1
   :widths: 25 16 34 25

   * - Canonical namespace
     - Stability
     - Responsibility
     - Continue with
   * - ``panorai.data``
     - Stable
     - ERP/view containers, raster fields and point/region/view observations.
     - :doc:`Spherical observations <../spherical_observations>`
   * - ``panorai.blenders``
     - Stable core; Experimental research variants
     - Mask-authoritative fusion for reconstructed ERP fields and diagnostics.
     - :doc:`Samplers and blenders <../samplers_blenders>`
   * - ``panorai.image_processing``
     - Stable SLIC; other legacy functions Experimental
     - Historical function API reused by configured spherical Ops.
     - :doc:`Core operations <../core_operations>`
   * - ``panorai.core.torch``
     - Stable
     - Lower-level Torch implementations used by ``panorai.ops.spherical.nn``.
     - :doc:`Spherical Ops <../spherical_ops>`
   * - ``panorai.core`` / ``panorai.core.numpy``
     - Experimental
     - Low-level execution specs and NumPy module grammar.
     - :doc:`API stability <../stability>`
   * - ``panorai.workflow``
     - Experimental
     - Model-oriented typed input/output trees and backend adapters.
     - :doc:`Data modalities <../../guides/how-to/data_modalities>`

Features and visual evidence
----------------------------

.. list-table:: Feature subsystems
   :header-rows: 1
   :widths: 25 16 34 25

   * - Canonical namespace
     - Stability
     - Responsibility
     - Continue with
   * - ``panorai.features`` core
     - Stable
     - Projected-view OpenCV extraction/matching with spherical geometry, deduplication and provenance.
     - :doc:`Features API <../features>`
   * - ``panorai.features`` DoG/tangent extensions
     - Experimental
     - Direct spherical DoG, scale selection, tangent patches and descriptors.
     - :doc:`Features and matching guide <../../guides/features_matching/index>`
   * - ``panorai.experimental.deep_learning``
     - Experimental
     - External model acquisition, spherical model ports, CAM and metric-depth adapters.
     - :doc:`Deep model workflows <../../guides/tutorials/12_composing_spherical_models>`
   * - ``panorai.experimental.deep_learning.segmentation``
     - Experimental
     - Semantic evidence, spherical prompts, SAM proposals and panoptic resolution.
     - :doc:`Semantic regions guide <../../guides/semantic_regions/index>`

Pose, depth and mapping
-----------------------

.. list-table:: Geometric inference subsystems
   :header-rows: 1
   :widths: 25 16 34 25

   * - Canonical namespace
     - Stability
     - Responsibility
     - Continue with
   * - ``panorai.estimators``
     - Experimental
     - Spherical Essential/R/t estimation, robust sampling, diagnostics and policies.
     - :doc:`Estimator catalog <estimators>`
   * - ``panorai.stereo``
     - Experimental
     - Direct-spherical dense plane sweep and radial-range result.
     - :doc:`Stereo API <../stereo>`
   * - ``panorai.reconstruction``
     - Experimental
     - Tracks, pose graphs, global mapping and metric landmark bundle adjustment.
     - :doc:`Reconstruction API <../reconstruction>`
   * - ``panorai.slam``
     - Experimental
     - Incremental and visual spherical SLAM, fisheye rays and local BA reports.
     - :doc:`SLAM API <../slam>`

Semantic and spatial reasoning
------------------------------

.. list-table:: Semantic/spatial subsystems
   :header-rows: 1
   :widths: 25 16 34 25

   * - Canonical namespace
     - Stability
     - Responsibility
     - Continue with
   * - ``panorai.pipeline`` semantic-instance stages
     - Experimental
     - CAM proposals, SAM refinement and spherical instance fusion.
     - :doc:`Stages catalog <stages>`
   * - ``panorai.experimental.deep_learning.mask_rcnn``
     - Experimental
     - Mask R-CNN taxonomy, executable workflow factory and unbound trainable topology.
     - :doc:`Applications catalog <applications>`
   * - ``panorai.object_localization``
     - Experimental
     - Spherical regions, cross-view transfer, associations and object hypotheses.
     - :doc:`Object localization API <../object_localization>`
   * - ``panorai.graph``
     - Experimental
     - Spatial-semantic evidence graph, providers, association, localization and queries.
     - :doc:`Graph API <../graph>`

Compatibility namespaces
------------------------

``panorai``, ``panorai.core.geometry`` and selected root feature exports retain
3.x Compatibility paths.  New code should use the canonical imports named in
the role catalogs and public inventory.  Compatibility preserves migration;
it does not define a second geometry or a second implementation.
