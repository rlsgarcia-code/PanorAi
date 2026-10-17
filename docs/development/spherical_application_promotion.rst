Spherical application promotion protocol
========================================

SLIC, DoG, StereoDepth and Mask R-CNN Paper are evaluated independently; they
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
decided independently; a failure for DoG, StereoDepth, or Mask R-CNN Paper
does not revoke the completed SLIC promotion.

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

Freeze the new pair selection *before* running dense inference. The builder
uses only registered overlap, metric baseline, asset presence and the frozen
domain/group balance. It automatically excludes every prior public stereo
manifest and refuses to run if any forbidden manifest has drifted:

.. code-block:: console

   python benchmarks/panoramic_stereo/build_registered_stereo_promotion_manifest.py \
       --overlap /licensed/audits/registered-cloud-overlap.jsonl \
       --analysis /licensed/audits/pair-analysis.jsonl \
       --stanford-root /licensed/prepared/stanford2d3d \
       --matterport-root /licensed/prepared/matterport3d \
       --frozen-at 2026-10-17T03:30:00Z \
       --output /external/results/registered-stereo-holdout.json

The output must contain six Stanford2D3D and six Matterport3D pairs with no
pair or view reuse. It records the input-audit hashes, protocol hash, prior
manifest hashes and a canonical payload hash. Dataset range values are used
only later to define the prediction-independent non-sky/no-return population;
they are not inspected during pair selection. The command refuses to overwrite
an already frozen manifest.

After the manifest is frozen, run the exact reviewed profile. The evaluator
streams one pair at a time, repeats each inference to require byte-identical
numerical arrays, and emits one protocol-ready record per corpus:

.. code-block:: console

   python benchmarks/panoramic_stereo/run_registered_stereo_heldout.py \
       --manifest /external/results/registered-stereo-holdout.json \
       --stanford-root /licensed/prepared/stanford2d3d \
       --matterport-root /licensed/prepared/matterport3d \
       --source-commit <40-character-commit> \
       --filter-backend native \
       --output-dir /external/results/registered-stereo-heldout-run

``external-evidence.json`` contains fixed-denominator, solid-angle-weighted
coverage, delta ``<1.25`` and conditional ``>2x`` error. The non-sky/no-return
mask is the positive finite dataset range and is constructed independently of
prediction validity; rejected predictions inside the remaining population
stay in the denominator. Invalid range output must be NaN. The generated
evidence is still only one section of the promotion protocol and cannot promote
the method by itself.

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

The analytic records are generated from an installed wheel, outside the
checkout, with the fixture frozen in the protocol:

.. code-block:: console

   /path/to/venv/bin/python \
       benchmarks/panoramic_stereo/run_registered_stereo_promotion.py \
       --backend native \
       --source-commit <40-character-commit> \
       --output /external/results/registered-stereo-native.json

Run the same command with ``--backend numpy``. The runner executes the reviewed
profile twice, compares every result array for determinism, and evaluates an
alternative unary cost only to prove that geometric observability is
cost-independent. It refuses to import PanorAi from the source checkout.

For each required native installed-wheel matrix cell, bind the exact wheel and
emit its artifact record in the same fresh process:

.. code-block:: console

   /path/to/venv/bin/python \
       benchmarks/panoramic_stereo/run_registered_stereo_promotion.py \
       --backend native \
       --source-commit <40-character-commit> \
       --wheel /external/dist/panorai-<version>-<tag>.whl \
       --output /external/results/analytic-native.json \
       --installed-output /external/results/installed-native.json

The frozen memory oracle uses process peak resident memory, not Python-only
allocation tracing. It includes imports and fixture construction, runs the
512x1024 two-source profile with at most one complete dense result alive, and
requires a peak no greater than 2 GiB. The runner also checks installed origin,
wheel metadata and bytes, development-version commit binding, native extension
availability, the analytic oracle and public-facade configuration round-trip.
Four records are still required: Python 3.11 and 3.14 on at least two
platforms.

The SLIC oracle does not compare integer label identities directly. It uses a
maximum-overlap assignment, solid-angle-weighted agreement, and a one-pixel
boundary tolerance. Rotated labels are sampled with nearest-neighbour
semantics. The fixed scenes, rotations, shapes, parameters, and thresholds live
in the protocol rather than in an after-the-fact report.
