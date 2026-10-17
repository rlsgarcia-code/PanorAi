Stereo APIs
===========

``panorai.stereo`` is an Experimental direct matching surface for two
same-shape central ERPs with known relative pose. It returns radial range in
the translation unit together with explicit validity, cost, confidence, and
the inverse-range hypothesis lattice.

Start with :doc:`../guides/tutorials/06_spherical_dense_stereo`. The full geometry,
objective, four-path aggregation, refinement, consistency policy, complexity,
and public validation limitations are derived in
:doc:`../concepts/spherical_dense_stereo`.

The pose convention is

.. math::

   X_B = R_{BA} X_A + t_{BA}.

Supplying only a unit translation direction produces range in baseline units,
not metres. Confidence is a best-versus-second-best separation on the current
cost volume and is not a calibrated probability.

PanorAi also provides six Experimental non-learned baselines for an
**already-rectified pinhole pair** under :mod:`panorai.stereo.classical`:
StereoBM, AD-Census, StereoSGBM, Census + MGM/tMGM, an ELAS-style support
prior, and PatchMatch Stereo. They are deliberately separate from the direct
spherical estimator above. Use them after sampling and rectifying overlapping
views, then return their dense fields through the object workflow.

The low-level classical surface follows
:doc:`../development/application_extensions`: matching costs, local
aggregation and global optimization are reusable operations; domain-changing
steps are public ``Stage`` objects; each complete baseline is an inspectable
``Pipeline``; and ``panorai.applications`` supplies the task wrapper.

Start with :doc:`../guides/pose_range/classical_stereo` for the common input,
the six compositions, result semantics, and current validation limits.

For an A-to-Z workflow, use
``panorai.applications.PanoramicDenseStereo``. It accepts two ERPs,
optionally estimates relative pose, resolves metric or scale-free baseline,
plans and streams paired rectified views, runs one of the classical matchers,
and fuses radial range back into the reference ERP. See
:doc:`../guides/pose_range/panoramic_dense_stereo`.

``panorai.applications.FullSphereDenseStereo`` extends the registered-pose
case to an explicit ``RegisteredPanoramaSet`` containing one reference and
multiple ``StereoSourceView`` relations. It streams an overlapping cube-oriented
gnomonic atlas, evaluates radial hypotheses against every usable baseline and
reports sampling support, stereo observability and accepted validity
separately.  A cubemap covers the raster sphere; only non-collinear baselines
can cover the pairwise epipolar degeneracies. ``observable`` and
``observable_source_count`` depend only on source data support, parallax and
the configured radial interval—not on Census/descriptor support or the range
selected by the matcher. ``source_count`` records cost-valid evidence at that
selected range. ``source_consensus_count`` is a separate per-pixel diagnostic;
the optional minimum-consensus validity gate is disabled by default.

``FullSphereStereoOptions`` also separates the dense cost pipeline into
``matching_cost`` (``intensity-gradient``, ``ad-census`` or
``multiscale-ad-census``),
``local_aggregation`` (``none`` or ``box``), and ``regularizer`` (``none``,
``sgm4`` or ``tmgm``). These stages run once, in that order, and are recorded
in result diagnostics and serialized application configuration. The balanced
default is reviewed ``ad-census`` + ``box`` + ``sgm4`` with angular ERP and
face taps. Explicit ``intensity-gradient`` and ``None`` tap values preserve the
historical profile. This evidence-backed default does not make the full-sphere
surface Stable.

``multiscale-ad-census`` averages the reviewed angular AD-Census cost at one
and two tap spacings before the same box aggregation and SGM4 stages. On a
prospectively frozen, view-disjoint twelve-pair audit it improved fixed
``delta < 1.25`` by 0.0039 on Stanford 2D-3D and 0.0054 on Matterport3D,
reduced conditional gross error in both domains, lost at most 0.0052 aggregate
coverage and cost 1.11x median paired runtime. All 24 numerical payloads and
population masks reproduced exactly. The gain is real but small, so the
single-scale reviewed default is unchanged and the multiscale cost remains an
explicit Experimental option.

Panoramic fusion exposes ``matching_confidence`` and
``consistency_confidence`` separately. Face-edge taper affects radial-range
fusion but is normalized out of matching confidence. Consistency is
``exp(-disagreement / radial_range)`` and the reported ``confidence`` is their
product. These Experimental fields support diagnostics and within-result
ranking; public multipair evidence rejects treating them as calibrated
probabilities or applying one universal threshold.

A frozen public two-source audit excludes sky/no-return pixels and evaluates
single-source, multibaseline, and strict-consensus routes on identical
prediction-independent populations. Multibaseline aggregation improves
fixed-population accuracy in five of six triplets, but strict two-source
consensus retains only about 4% of Stanford and 10% of Matterport observable
support. The result supports the multiview architecture while still rejecting
Stable promotion of the current matcher and visibility policy.

A replayed weighted-mean source-cost ablation underperforms the existing lower
median in both public domains. It is retained as an explicit policy, not
promoted to the default; visibility-aware source selection remains an open
promotion gate.

A fixed parameter-free confidence ablation also tested multiplying the
reported score by half of one plus the fraction of source labels agreeing with
the fused label. It slightly reduced top-quartile accuracy in both public
domains and did not improve absolute-threshold transfer. The runtime confidence
is therefore unchanged; source agreement remains a separate diagnostic rather
than a calibrated probability.

An outcome-blind two-pass ablation then selected a fixed compatible source
group from each source's complete cost curve before recomputing the fused
volume. It reduced Stanford gross outliers but reduced coverage and fixed-
population accuracy in both domains, while Matterport gross outliers increased.
The rejected policy is not a public option. Unary cost-curve sharpness is not a
sufficient visibility or occlusion model.

A separately frozen three-source public protocol confirms that the existing
true median reduces conditional gross-label outliers in both domains. It also
reduces accepted coverage relative to two sources, so fixed-population accuracy
does not improve overall. Requiring all three source costs yields almost the
same result. The remaining gate is therefore a source-count-portable cost and
acceptance scale, not merely another minimum-source switch.

These promotion metrics exclude sky/no-return with a prediction-independent
positive finite dataset-range mask before applying geometric support and
observability. Rejected predictions inside the remaining population are not
silently removed from the fixed accuracy denominator.

A frozen acceptance-gate decomposition also shows that the current absolute
matching-cost gate is inactive: disabling it changes none of sixty public
two/three-source conditions. Removing the confidence gate recovers part of the
third-source coverage loss, particularly on Matterport, while removing the
texture gate raises absolute coverage but makes the source-count regression
larger. No default changes. Per-reason rejection instrumentation and a
prospectively specified source-count-aware confidence remain promotion gates.

``FullSphereStereoResult.diagnostics`` includes a cumulative per-face
acceptance audit. It reports counts after support, interior-label, finite-range,
texture, confidence, cost, source-evidence, consensus, parallax and final
acceptance predicates. Overlapping face samples remain separate by design;
use the fixed non-sky ERP protocol—not these counts—for accuracy claims.
Instrumented public replay preserved all prior output-array hashes and shows
that boundary labels dominate the Stanford source-count loss while confidence
dominates the Matterport loss.

Accepting endpoint winners as exact discrete ranges was evaluated
prospectively and rejected. It raises coverage but slightly lowers fixed
accuracy in both public domains and increases conditional gross error by more
than the frozen tolerance. The candidate option is not exposed. Endpoint
winners remain invalid rather than silently converting an out-of-range bound
into a metric point estimate.

A second prospective protocol selected twelve entirely new public pairs before
inference and compared the existing confidence threshold with report-only
confidence on the same fixed non-sky populations. Report-only confidence
improved coverage and fixed-population accuracy in both Stanford and
Matterport, but Stanford conditional gross error regressed by ``0.04386``,
above the frozen ``0.04`` limit. The result reproduced exactly, the candidate
option was removed, and the threshold default is unchanged. Confidence remains
a ranking score, not a calibrated probability; improving its cross-domain
acceptance behavior is still a promotion gate.

Twelve additional view-disjoint public pairs prospectively compared the two
available unary costs with box aggregation and SGM4 held fixed. AD-Census
improved fixed-population accuracy in both domains, increased coverage, reduced
Stanford gross error, stayed within the Matterport gross-error tolerance, and
was faster in two complete runs. Numerical evidence and all 24 payloads
reproduced exactly. A separate reviewer reproduced all numerical payloads,
populations, metrics, and aggregates; the only raw-JSON difference was the
additive angular-neighbourhood diagnostic introduced after the original run.
AD-Census is therefore the integrated default for new option objects. Explicit
``intensity-gradient`` and old saved configurations retain historical behavior.

A subsequent development-only factorial on the previously observed VAL-098
pairs rejects resolution scaling as a free quality improvement. At a common
``512x1024`` evaluation grid, doubling the inverse-range lattice from 48 to 96
hypotheses improves fixed ``delta < 1.25`` by ``0.0312`` on Stanford and
``0.0010`` on Matterport. Doubling face resolution from 128 to 256 with the
unchanged pixel-window settings instead lowers fixed accuracy by about
``0.034`` in both domains. The current pixel-valued window controls both ERP
feature construction and face-local aggregation, so angular support is not
resolution invariant. This is now an explicit promotion blocker rather than a
reason to advertise a larger input size.

``FullSphereStereoOptions`` therefore exposes two additive Experimental
controls: ``erp_feature_tap_deg`` for the tangent-plane ERP feature filter and
``face_cost_tap_deg`` for the common Census/aggregation tap grid on projected
faces. ``None`` preserves the historical one-pixel behavior byte for byte.
The face option is converted to a deterministic integer stride from actual
centre-ray spacing; it preserves tap count and streaming memory rather than
allocating a denser large descriptor. On the eleven development pairs, using
both angular supports at ``512x1024`` improves fixed ``delta < 1.25`` by about
``0.11`` in both Stanford and Matterport and exactly reproduces on replay.
A prospectively frozen test on twelve new view-disjoint pairs then passed the
fixed-accuracy, coverage, conditional-gross-error, and ``1.35x`` runtime gates
in both domains. Sky and no-return directions were excluded using the
prediction-independent positive finite dataset-range proxy; geometric
observability and prediction validity remained separate. The scientific
evidence, 24 output payloads, and 24 population masks reproduced exactly. The
angular configuration was independently reproduced exactly and is now the
default for new option objects. Explicit ``None`` taps and saved configurations
that predate these fields retain the historical one-pixel behavior. The
stability tier remains Experimental until the remaining pose, uncertainty, and
installed-wheel gates pass.

API
---

.. automodule:: panorai.stereo
   :members:
   :member-order: bysource
   :undoc-members:

Classical rectified stereo
--------------------------

.. automodule:: panorai.stereo.classical
   :members:
   :member-order: bysource

Panoramic stereo workflow
-------------------------

.. automodule:: panorai.stereo.panoramic
   :members:
   :member-order: bysource
