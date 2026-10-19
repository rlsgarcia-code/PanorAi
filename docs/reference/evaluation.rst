Spherical evaluation API
========================

``panorai.evaluation`` is the Experimental, NumPy-canonical evaluation
surface.  It measures spherical results without downloading datasets,
executing models, choosing thresholds, or changing the stability tier of the
algorithm under test.

.. panorai-api-components:: evaluator

Metrics
-------

.. autofunction:: panorai.evaluation.solid_angle_weights

.. autofunction:: panorai.evaluation.evaluate_binary_regions

.. autofunction:: panorai.evaluation.evaluate_features

.. autofunction:: panorai.evaluation.evaluate_pose

.. autofunction:: panorai.evaluation.evaluate_radial_range

Typed results and evidence
--------------------------

.. autoclass:: panorai.evaluation.BinaryRegionEvaluation
   :members:

.. autoclass:: panorai.evaluation.FeatureEvaluation
   :members:

.. autoclass:: panorai.evaluation.PoseEvaluation
   :members:

.. autoclass:: panorai.evaluation.DepthEvaluation
   :members:

.. autoclass:: panorai.evaluation.EvaluationProtocol
   :members:

.. autoclass:: panorai.evaluation.EvaluationRecord
   :members:

All ERP metrics use pixel-centre geometry.  Region and radial-range metrics
use exact per-pixel solid angle; support and validity are separate boolean
inputs.  Torch tensors are rejected so an implicit device transfer cannot
become part of an evaluation protocol.

See :doc:`the concept <../concepts/spherical-evaluation>` for the measurement
contract and :doc:`the how-to guide <../guides/how-to/spherical_evaluation>`
for a reproducible record.
