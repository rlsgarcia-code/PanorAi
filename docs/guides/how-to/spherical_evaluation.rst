Evaluate spherical results reproducibly
=======================================

.. panorai-api-components:: evaluator

Use ``panorai.evaluation`` after running the algorithm under test.  The
example below evaluates a binary ERP region and stores a protocol-bound record.

.. code-block:: python

   import numpy as np

   from panorai.evaluation import (
       EvaluationProtocol,
       EvaluationRecord,
       evaluate_binary_regions,
   )

   prediction = np.zeros((64, 128), dtype=bool)
   reference = np.zeros_like(prediction)
   prediction[20:40, 10:35] = True
   reference[20:40, 12:37] = True
   support = np.ones_like(prediction)
   validity = np.ones_like(prediction)

   result = evaluate_binary_regions(
       prediction,
       reference,
       support=support,
       prediction_validity=validity,
       reference_validity=validity,
       boundary_tolerance_deg=1.0,
   )

   protocol = EvaluationProtocol(
       protocol_id="my-region-eval",
       version=1,
       evaluator="panorai.evaluation.evaluate_binary_regions",
       metric_names=tuple(result.metrics),
       parameters={"boundary_tolerance_deg": 1.0},
   )
   record = EvaluationRecord(
       protocol=protocol,
       subject_id="scene-017",
       input_id="panorama-003",
       reference_id="annotation-v2:003",
       split="development",
       algorithm_id="my-segmenter",
       algorithm_config={"threshold": 0.65},
       package_version="3.8.0.dev0",
       backend="numpy",
       metrics=result.metrics,
       environment={"numpy": np.__version__},
   )

   payload = record.to_json()
   restored = EvaluationRecord.from_json(payload)
   assert restored.protocol.fingerprint == protocol.fingerprint

Before opening held-out results
-------------------------------

#. Choose scene- or subject-disjoint development and held-out IDs.
#. Fix metric names, thresholds, support rules and empty-set conventions in an
   ``EvaluationProtocol``.
#. Select algorithm parameters using development data only.
#. Record the exact package and artifact identities.
#. Evaluate the held-out split once under the frozen protocol.
#. Keep abstentions and failures as records with explicit reasons.
#. Ask an independent reviewer to reproduce the installed-wheel evidence.

For radial range, pass units and separate support and validity masks to
``evaluate_radial_range``.  Reference values equal to zero remain valid for
absolute error and coverage, but are excluded from ratio metrics such as
AbsRel and delta accuracy.  For keypoints, convert locations to unit bearings
and use ``evaluate_features``; planar pixel thresholds are not interchangeable
with angular thresholds.

See :doc:`the concept <../../concepts/spherical-evaluation>` before defining a
publication protocol and :doc:`the API reference <../../reference/evaluation>`
for exact signatures.
