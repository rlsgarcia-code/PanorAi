Probabilistic two-view pose from two EQR images
===============================================

``panorai.experimental.two_view_probability`` composes the optimized spherical
frontend, the existing ``SphericalRelativePoseEstimator``, and two calibrated
probability stages. It does not run dense stereo and it does not change the
estimated rotation or translation direction.

Runtime inputs
--------------

The complete runtime operation needs only:

* two equal-resolution equirectangular images;
* one explicit boolean validity mask per image; and
* an estimated camera-centre baseline, in metres, optionally with a one-sigma
  uncertainty.

Registered depth clouds are **not** runtime inputs. They were used offline to
measure spatial overlap and supervise the RGB/match overlap proxy. The proxy
therefore reports a posterior over overlap bins, not a measured cloud overlap.

Complete operation
------------------

.. code-block:: python

   import numpy as np
   from panorai.experimental.two_view_probability import (
       BaselineEstimate,
       ProbabilisticSphericalTwoViewEstimator,
   )

   estimator = ProbabilisticSphericalTwoViewEstimator()
   result = estimator.estimate(
       panorama_a,
       panorama_b,
       validity_a=np.asarray(validity_a, dtype=bool),
       validity_b=np.asarray(validity_b, dtype=bool),
       baseline=BaselineEstimate(mean_m=0.80, standard_deviation_m=0.05),
       panorama_ids=("capture-a", "capture-b"),
   )

   print(result.overlap.expected_overlap)
   print(result.overlap.probability_at_least_0_5)
   print(result.capture_advisory.action)  # attempt, recapture, or unsupported
   print(result.accepted, result.decision_reason)

   if result.accepted:
       R_21 = result.pose.rotation
       t_21_m = result.translation_m

The translation returned by the geometric estimator has unit norm. Multiplying
it by ``baseline.mean_m`` supplies metric magnitude from the external baseline;
two calibrated central views cannot recover that magnitude by themselves.

What each model means
---------------------

The pre-pose overlap proxy consumes only frontend evidence: minimum keypoint
count, match count/rate, descriptor-distance quantiles, ratio-test median, and
valid-mask fraction. It is a calibrated proportional-odds model with six
overlap bins: ``[0,.1)``, ``[.1,.3)``, ``[.3,.5)``, ``[.5,.7)``,
``[.7,.9)``, and ``[.9,1]``. Its posterior is marginalized jointly with the
baseline uncertainty through the frozen capture model. The resulting capture
action is advisory; it can recommend recapture, but cannot turn a weak pose
into an accepted one.

The post-pose model consumes the public ``RelativePoseQualityReport`` directly:
inliers, coverage, residual-derived quality, parallax, cheirality, stability,
model competition, and translation-orientation evidence. Final acceptance is:

.. code-block:: text

   pose returned
   AND public quality policy accepted
   AND calibrated post-pose precision probability >= 0.90

The capture probability is intentionally absent from this final gate because
the RGB-only overlap estimate remains a latent proxy.

When a registered-cloud overlap is available for offline evaluation, the
original explicit-overlap capture model remains directly usable:

.. code-block:: python

   from panorai.experimental.two_view_probability import (
       FrozenPoseProbabilityModels,
   )

   models = FrozenPoseProbabilityModels.load_default()
   probability = models.score_capture_explicit(
       overlap=0.65,
       baseline_m=0.80,
   )

Evidence and limits
-------------------

The overlap proxy was fit on 2,385 two-view pairs (4,770 image observations)
from three anonymously identified evaluation domains, split by 69 disjoint
spatial independence components. The aligned table does not preserve source
panorama IDs, so a defensible unique-panorama count is unavailable. On the
held-out 435 pairs, the expected-overlap mean absolute error was 0.139 and the
``overlap >= 50%`` classifier at probability 0.5 had precision 0.811, recall
0.370, and Brier score 0.099. This supports conservative capture advice, not a
release reliability claim.

The capture and post-pose probability bundle was calibrated against PanorAi
3.5.0 commit ``03c5b36``. Its status is retrospective and still requires
prospective confirmation. On the held-out evaluation split, the final rule had
the following pair-level results:

================ ============ ========= ====== ================
Population       Selected     Precision Recall  95% lower bound
================ ============ ========= ====== ================
Pooled           107 / 435    0.981     0.745  0.942
Domain 1          81 / 270    0.988     0.800  0.943
Domain 2           3 / 15     1.000     0.750  0.368
Domain 3          23 / 150    0.957     0.595  0.810
================ ============ ========= ====== ================

The pooled point estimate and lower bound exceed the 0.90 precision target,
but Domains 2 and 3 do not have domain-specific lower bounds above 0.90. This
is not a release reliability claim. The API and both models are Experimental;
callers must persist ``result.provenance``, the complete probability output,
and the public quality report when collecting new validation evidence.

Reuse existing matches
----------------------

Applications that already use the public feature API can avoid repeating the
frontend:

.. code-block:: python

   result = estimator.estimate_matches(
       matches,
       baseline=BaselineEstimate(0.80, 0.05),
       keypoint_counts=(len(features_a), len(features_b)),
       valid_fractions=(validity_a.mean(), validity_b.mean()),
   )

The optimized image route is explicit: native spherical convolution,
``SphericalDoGDetector.detect_batch()`` with batch two and 4,096 keypoints,
48-by-48 tangent patches with six-scale radius through a four-worker provider,
and the calibrated single-scale fixed-orientation RootSIFT descriptor.
