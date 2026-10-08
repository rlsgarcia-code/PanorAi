Spherical frontend v2 promotion plan
====================================

Purpose
-------

The sphere-native detector and tangent-patch descriptor adapter remain an
``Experimental`` capability until the gates below are met.  Promotion is
evidence based: faster execution is accepted only when keypoints, descriptors,
matches, and estimated pose remain equivalent or satisfy a preregistered
accuracy gate.

The candidate contracts are:

* ``panorai-spherical-detector/v2`` for bearing, angular scale, response,
  octave/level, source ERP location, support, and validity;
* ``panorai-tangent-patches/v1`` for descriptor-neutral patch geometry,
  angular context, resolution, projection, mask, and orientation;
* ``panorai-tangent-opencv-descriptor/v1`` for applying OpenCV SIFT or ORB to
  those patches.  AKAZE remains conditional because its descriptor is coupled
  to its nonlinear scale space.

Current maturity
----------------

The implementation is an **internal candidate (E1)**, not a stable public API.
It already contains second-order Taylor refinement in position and scale,
scale-aware support checks, public orientation records, exact multi-image
detection, ordered parallel tangent-patch materialization, and descriptor
adapters.  Batch execution has been checked for exact output parity with
independent execution on the current development fixtures.

The maturity levels are:

``E0`` prototype
   The numerical route exists, but contracts and repeatable measurements are
   incomplete.

``E1`` internal candidate
   Source, tests, documentation, and reproducible benchmarks are consolidated
   on a branch.  This is the current level.

``E2`` public beta
   Public contracts are frozen, supported platforms pass CI, and the detector
   satisfies preregistered correctness, quality, and resource gates on
   development groups that are spatially independent.

``E3`` stable
   The release candidate passes held-out, multi-corpus evaluation without
   threshold tuning, downstream migration evidence exists, and the wheel is
   reproducible on every supported platform.

Promotion gates
---------------

Gate A -- package and API integrity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Build the source distribution and wheels on every supported Python and OS
   combination.
#. Run the full PanorAi suite with the NumPy/native paths and, where available,
   the optional Torch backend.
#. Expose only documented public entry points.  Benchmarks may use private
   diagnostics, but consumers may not depend on them.
#. Serialize all detector, projector, sampler, patch, orientation, and
   descriptor parameters.  Implicit FOV, resolution, interpolation,
   pixel-center, invalid-policy, or axis conventions are forbidden.
#. Preserve the adapter requirements for partial ERP support, including
   ``shadow_angle`` and ``shadow_padded``.  RGB, mask, and validity must use the
   same projection grid.

Exit criterion: the supported CI matrix is green, documentation builds without
new warnings, and a clean environment can reproduce the declared interfaces.

Gate B -- numerical correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Verify synthetic extrema with known subpixel and subscale offsets against
   the second-order Taylor solution.
#. Exercise seams, poles, masks, partial ERP support, rotations, and the
   configured invalid-value policy.
#. Require exact equality between independent and batched detection for
   position, bearing, scale, response, octave/level, validity, and ordering.
#. Require exact ordered equality between serial and parallel tangent-patch
   materialization.
#. Check projection/ray round trips and cross-resolution angular consistency.
#. Test orientation assignment separately from descriptor extraction,
   including multiple strong orientation peaks.

Exit criterion: no unexplained numerical difference and no geometry contract
violation on the frozen fixtures.

Gate C -- detection and descriptor quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate sphere-native, cubemap, and icosahedral detection using the same
angular support and point budget.  Report each corpus independently before any
aggregate.  At minimum record:

* raw and unique keypoint counts;
* spherical-cell coverage and concentration;
* repeatability under known rotations at 0.5 and 1 degree;
* overlap duplicates and descriptor divergence for duplicate multiface points;
* descriptor valid-yield, match count, GT-compatible match precision/recall,
  and spatial coverage of compatible matches;
* strict rotation and translation-direction errors, acceptance/abstention,
  and catastrophic accepted poses.

Proposed beta gates, to be frozen before evaluating held-out groups, are:

* repeatability no more than one percentage point below the selected multiface
  reference at 1 degree;
* GT-compatible match precision at least 86% on the frozen indoor development
  protocol;
* at least 11/15 strict-pose successes and at least 10/15 accepted strict
  successes on that protocol;
* zero catastrophic *accepted* poses.

P-74 is reported as a separate industrial stress facet, with the audited
native polar adapter and fixed shadow policy.  It must not be pooled with the
indoor result.  Before beta, either the preregistered P-74 acceptance target is
met or the public scope must explicitly exclude that operating condition.

Gate D -- performance and resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure cold and warm execution separately, per image and per pair.  Record
P50/P90 wall time, peak RSS, image shape, detector batch size, patch worker
count, thread environment, compiler, CPU, and resolved PanorAi version.

Initial reference-machine budgets are:

* cold detection for a 1024x2048 pair: at most 5 seconds;
* cold detection for a 2048x4096 pair: at most 20 seconds;
* cold complete frontend: at most 10 and 32 seconds respectively;
* peak RSS: at most 3 GiB.

These are release gates, not permission to alter scientific output.  Each
optimization must first pass exact detector parity and then the downstream
descriptor, matching, and pose comparison.

Gate E -- independent generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Freeze hyperparameters and thresholds using only development spatial groups.
#. Evaluate untouched spatial groups from each permitted corpus.  Stanford
   areas, Matterport buildings, and audited industrial components remain
   indivisible.
#. Never use the frozen Matterport test split or a held-out P-74 group to choose
   a threshold, patch FOV, detector budget, or acceptance rule.
#. Report continuous error distributions conditioned on observable scene and
   pair covariates, and separately report probability of a valid estimate.

Exit criterion: beta gates reproduce on the preregistered held-out protocol,
with confidence intervals and no corpus hidden by a macro-average.

Gate F -- release and downstream adoption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Publish a release-candidate wheel, changelog, migration guide, and complete
   configuration example.
#. Replay one downstream WP1 integration using only the public API and the
   release-candidate dependency, never a source-path import.
#. Keep the former multiface route available as a documented reference during
   the beta period.
#. Promote to stable only after review of the held-out report and a clean
   release build.  Updating the research program's canonical dependency is a
   separate, explicit integration task.

Required experiment order
-------------------------

The recommended order minimizes wasted tuning and information leakage:

#. CI/package integrity and deterministic frozen fixtures;
#. detector-only correctness, repeatability, and timing;
#. patch/orientation/descriptor yield and timing;
#. matching quality with GT used only for audit;
#. robust Essential estimation, acceptance, and pose error;
#. held-out multi-corpus confirmation;
#. release-candidate downstream replay.

Every run must retain stage timings and counts so a final pose failure can be
attributed to detection, patch validity, description, matching, robust
consensus, decomposition, or acceptance rather than only reported end to end.
