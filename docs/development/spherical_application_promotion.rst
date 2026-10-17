Spherical application promotion protocol
========================================

SLIC, DoG, StereoDepth and Mask R-CNN Workflow are evaluated independently; they
are not promoted by implementation tests alone. Their prospective criteria are
frozen in ``contracts/spherical-application-promotion-v1.json``. SLIC has
passed its frozen prospective criteria and is Stable as
``panorai-spherical-slic/v1``. The other three wrappers remain Experimental.

The protocol requires:

* analytic geometry, validity and seam evidence for SLIC, plus public-wrapper
  non-axis-aligned SO(3) rotation oracles and a local planar-limit comparison;
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
deterministic adapter smoke test into scientific evidence. Each requirement is
decided independently; a failure for DoG, StereoDepth, or Mask R-CNN Workflow
does not revoke the completed SLIC promotion.

The SLIC oracle does not compare integer label identities directly. It uses a
maximum-overlap assignment, solid-angle-weighted agreement, and a one-pixel
boundary tolerance. Rotated labels are sampled with nearest-neighbour
semantics. The fixed scenes, rotations, shapes, parameters, and thresholds live
in the protocol rather than in an after-the-fact report.
