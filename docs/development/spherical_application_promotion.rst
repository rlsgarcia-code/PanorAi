Spherical application promotion protocol
========================================

The Experimental SLIC, DoG, StereoDepth and Mask R-CNN Paper wrappers are not
promoted by implementation tests alone. Their prospective criteria are frozen
in ``contracts/spherical-application-promotion-v1.json``.

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
deterministic adapter smoke test into scientific evidence. Until every
requirement passes and is independently reviewed, the corresponding wrapper
remains Experimental.

Registered full-sphere stereo
-----------------------------

The supplied-pose full-sphere route has a narrower prospective protocol in
``contracts/registered-full-sphere-stereo-promotion-v1.json``. It is separate
because promoting the registered route must not promote automatic pose,
scale recovery, optional multiscale/tMGM paths, custom profiles, or calibrated
probability claims.

The protocol freezes one public profile, requires native and NumPy analytic
oracles, a new view-disjoint Stanford2D3D/Matterport3D cohort, installed wheels
on at least two platforms and Python 3.11/3.14, streaming-memory evidence, and
review by someone other than the implementer. Hashes of every previously
observed public cohort are denied explicitly, so replaying development data
cannot close the new held-out gate.

Print or evaluate this protocol with the same fail-closed evaluator:

.. code-block:: console

   python scripts/evaluate_spherical_application_promotion.py \
       --protocol contracts/registered-full-sphere-stereo-promotion-v1.json \
       --print-protocol-sha256

   python scripts/evaluate_spherical_application_promotion.py \
       --protocol contracts/registered-full-sphere-stereo-promotion-v1.json \
       --evidence /external/results/registered-stereo-evidence.json \
       --output /external/results/registered-stereo-decision.json

Creating the protocol does not promote the route. The runtime interface and
both granular inventory records remain explicitly Experimental until every
record passes and the evidence is independently reproduced.

The SLIC oracle does not compare integer label identities directly. It uses a
maximum-overlap assignment, solid-angle-weighted agreement, and a one-pixel
boundary tolerance. Rotated labels are sampled with nearest-neighbour
semantics. The fixed scenes, rotations, shapes, parameters, and thresholds live
in the protocol rather than in an after-the-fact report.
