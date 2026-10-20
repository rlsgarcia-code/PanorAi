Research evaluation catalog
===========================

Evaluators compare an algorithm result with reference evidence.  They are
kept outside Applications and estimators so the implementation under test
cannot silently choose its own oracle, validity convention, or held-out
split. The NumPy-canonical region, feature, pose and radial-range evaluator
surface is Stable as ``panorai-spherical-evaluation/v1``. Surface-reconstruction
evaluation is separately Stable as
``panorai-surface-reconstruction-evaluation/v1``; neither tier promotes the
algorithm being measured.

Registered evaluator surfaces
-----------------------------

.. panorai-api-components:: evaluator

Available evaluators
--------------------

.. list-table:: Geometry-aware evaluation components
   :header-rows: 1
   :widths: 27 25 25 23

   * - Component
     - Input
     - Output
     - Spherical contract
   * - ``evaluate_binary_regions``
     - two boolean ERP regions plus support and validity
     - ``BinaryRegionEvaluation``
     - solid-angle overlap and great-circle boundary matching
   * - ``evaluate_features``
     - predicted/reference unit bearings and optional scales
     - ``FeatureEvaluation``
     - one-to-one angular assignment
   * - ``evaluate_pose``
     - two rotations and optional translation directions
     - ``PoseEvaluation``
     - geodesic SO(3) and S2 errors
   * - ``evaluate_radial_range``
     - two radial-range ERPs plus support, validity and confidence
     - ``DepthEvaluation``
     - area-weighted accuracy, coverage and confidence ordering
   * - ``evaluate_surface_reconstruction``
     - predicted/reference ``SurfaceMesh`` plus a frozen distance threshold
     - ``SurfaceReconstructionEvaluation``
     - exact bidirectional vertex-to-triangle distance, F-score and oriented
       normals
   * - ``aggregate_surface_reconstruction``
     - named scene evaluations from one frozen cohort
     - ``SurfaceReconstructionEvaluationReport``
     - unweighted per-scene medians; common units and threshold required
   * - ``EvaluationProtocol``
     - evaluator identity, metrics and frozen parameters
     - fingerprinted protocol
     - thresholds are fixed before evaluation
   * - ``EvaluationRecord``
     - protocol, result metrics and provenance
     - canonical JSON evidence
     - fail-closed, finite and pickle-free

The exact signatures are in :doc:`the evaluation API <../evaluation>`.
Protocol design and interpretation are in :doc:`the evaluation concept
<../../concepts/spherical-evaluation>`.
