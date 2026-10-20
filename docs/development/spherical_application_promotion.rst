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

DoG reviewed profile
--------------------

The original multi-application protocol above records historical DoG evidence,
but it cannot promote data that had already been used during development. The
candidate ``SphericalDoGFeatures.reviewed_profile()`` therefore has a dedicated
prospective contract in
``contracts/dog-reviewed-profile-promotion-v1.json`` and a bound development
exclusion ledger in ``contracts/dog-development-exclusions-v1.json``.

The public identity
``panorai-spherical-dog-features/reviewed-v1`` fixes dense spherical DoG,
tangent-patch extraction and RootSIFT. Its constructor and catalog entry remain
Experimental at protocol creation time. Free-form configurations,
coarse-to-fine detection and alternative descriptors are outside the candidate
boundary.

Reproduce the already frozen analytic oracle through the exact factory with
both convolution backends::

   python benchmarks/spherical_dog/run_reviewed_analytic.py \
       --backend numpy \
       --source-root /clean/source/checkout \
       --source-commit <40-character-commit> \
       --output /external/results/dog-reviewed-analytic-numpy.json

   python benchmarks/spherical_dog/run_reviewed_analytic.py \
       --backend native \
       --wheel /external/dist/panorai-<exact-version>-<tags>.whl \
       --source-commit <40-character-commit> \
       --output /external/results/dog-reviewed-analytic-native.json

The NumPy development reproduction requires an exact clean source checkout.
The installed/native reproduction verifies ``direct_url.json``, wheel SHA-256,
artifact version and SCM commit marker before evaluating the oracle.

First construct an inventory of registered candidate pairs without feature
outputs. Each JSONL row contains ``corpus``, ``scene_id``, ``pair_id``, a
separate ``exclusion_group_id`` and two views with safe relative RGB/range
paths and their precomputed SHA-256 identities, ``R_world_to_camera`` and
``center_world``, plus precomputed directional cloud overlap. ``scene_id`` is
the independent spatial scene used for the
one-pair-per-scene rule; ``exclusion_group_id`` is the coarser group bound to
the development ledger (for example, a Stanford area). The frozen
eligibility range is overlap at least ``0.50`` and metric baseline from
``0.15`` through ``2.50`` metres. Freeze the cohort before the first feature
inference::

   python benchmarks/spherical_dog/build_reviewed_holdout_manifest.py \
       --inventory /licensed/audits/dog-pair-inventory.jsonl \
       --matterport-root /licensed/prepared/matterport3d \
       --stanford-root /licensed/prepared/stanford2d3d \
       --frozen-at 2026-10-20T00:00:00Z \
       --output /external/results/dog-reviewed-holdout.json

The selector reads identifiers, registered poses, the precomputed geometric
overlap audit, asset presence and precomputed asset identities only. It does
not open RGB pixels,
radial-range values, features, matches or evaluation results. It selects six
scene-, view- and asset-disjoint pairs from each corpus, excludes every development group in the
bound ledger and refuses to overwrite the result.

Only after the manifest is frozen, execute the profile and the independent
pose/radial-range/occlusion correspondence oracle::

   python benchmarks/spherical_dog/run_reviewed_holdout.py \
       --manifest /external/results/dog-reviewed-holdout.json \
       --inventory /licensed/audits/dog-pair-inventory.jsonl \
       --matterport-root /licensed/prepared/matterport3d \
       --stanford-root /licensed/prepared/stanford2d3d \
       --source-commit <40-character-commit> \
       --output /external/results/dog-reviewed-heldout.json

Native 2:1 ERPs are accepted at their corpus resolution and deterministically
resampled to ``512x1024`` for this gate: area interpolation for RGB and nearest
sampling for radial range. This preprocessing is part of the frozen protocol;
it is not selected after seeing results.

For each corpus, the fixed gate requires repeatability at least ``0.60``,
median angular localization error at most ``1 degree`` and verified-inlier
precision at least ``0.80``. Thresholds cannot change after the manifest is
opened. Failure leaves the entire profile Experimental.

The same protocol separately requires installed-wheel timing records at
``512x1024``, ``2048x4096`` and a native resolution, plus an installed-wheel
Python 3.11/3.14 matrix on at least two platforms. Use
``benchmark_reviewed_profile.py`` and ``run_installed_profile.py`` from the
same directory in clean environments. Evaluate the combined external evidence
with::

   python scripts/evaluate_spherical_application_promotion.py \
       --protocol contracts/dog-reviewed-profile-promotion-v1.json \
       --evidence /external/results/dog-reviewed-complete-evidence.json \
       --source-commit <40-character-candidate-commit> \
       --output /external/results/dog-reviewed-decision.json

Creating these tools is the first promotion PR. A second tier-only PR is
permitted only after every record passes and an independent reviewer reproduces
the source, artifact and held-out evidence.

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

The protocol has now passed for the exact supplied-metric-pose profile.
``FullSphereDenseStereo.reviewed_profile()`` therefore reports the Stable
``panorai-registered-full-sphere-stereo/v1`` contract. The low-level engine,
generic application record, automatic pose, custom configurations and
confidence-as-probability claims remain explicitly Experimental.

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
