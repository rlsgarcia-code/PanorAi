Spherical application promotion protocol
========================================

The Experimental SLIC, DoG, StereoDepth and Mask R-CNN Paper wrappers are not
promoted by implementation tests alone. Their prospective criteria are frozen
in ``contracts/spherical-application-promotion-v1.json``.

The protocol requires:

* analytic geometry, validity and seam evidence for SLIC;
* analytic spherical blobs plus two external held-out corpora for DoG;
* an analytic textured sphere plus independent Matterport3D and Stanford2D3D
  held-out records for StereoDepth;
* real external CAM/SAM checkpoints and annotated panoramic held-out data for
  the paper Mask R-CNN method.

Evidence is bound to the exact protocol bytes by SHA-256. Missing records,
development-only records, non-finite metrics and a changed protocol hash all
fail closed. Thresholds cannot be relaxed after held-out results are opened.

Print the current protocol identity:

.. code-block:: console

   python scripts/evaluate_spherical_application_promotion.py \\
       --print-protocol-sha256

Evaluate an external evidence record:

.. code-block:: console

   python scripts/evaluate_spherical_application_promotion.py \\
       --evidence /external/results/promotion-evidence.json \\
       --output /external/results/promotion-decision.json

The evaluator checks the submitted measurements and cohort structure. It does
not download datasets, accept model licenses, generate annotations, or turn a
deterministic adapter smoke test into scientific evidence. Until every
requirement passes and is independently reviewed, the corresponding wrapper
remains Experimental.
