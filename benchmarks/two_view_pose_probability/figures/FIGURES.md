# Conceptual figures for the two-view pose study

These raster illustrations support the paper narrative. They are conceptual,
not experimental evidence: they contain no measured values and must not be used
to infer thresholds, probabilities, or estimator performance. Quantitative
figures must be regenerated from the frozen study tables and plotting scripts.

All three assets were generated on 2026-10-08 with the built-in OpenAI image
generation tool. Captions and panel labels should be typeset by the paper build,
not baked into the raster.

The files prefixed `quantitative-` are measured figures, not generated
illustrations. They were rendered deterministically by
`render_paper_results.py` or `render_narrative_figures.py` from frozen VAL-018
evidence. `two-model-probability-story.png` is a deterministic methods diagram:
it contains the declared model structure but no measured performance. Source
tables, model cards, commands, and checksums are recorded in the corresponding
`.agents/results/VAL-018-*` run report.

## Recommended paper sequence

1. `graphical-abstract-two-view.png` — graphical abstract only.
2. `two-model-probability-story.png` — the two inferential questions and gates.
3. `quantitative-evidence-base.png` — sample size and independent support.
4. `capture-boundary-conditions.png` — operator-visible capture regimes.
5. `quantitative-overlap-response.png` — measured response versus overlap.
6. `controlled-runtime-overlap-response.png` — primary idle-host runtime
   versus overlap under the exact optimized PanorAi 3.5.0 route;
   `runtime-overlap-response.png` is retained as a shared-host diagnostic.
7. `quantitative-calibration-heldout.png` and
   `quantitative-cross-dataset-transfer.png` — calibration and transfer.
8. `post-processing-evidence.png` and `quantitative-post-ablation.png` — the
   second model and its diagnostics.
9. `capture-probability-surface.png` — calibrated acceptance, conditional
   precision, and usable-pair response over supported overlap × baseline cells.
10. `quantitative-selective-rule-evaluation.png` — the retrospective verdict.
11. `quantitative-optimized-main-replay.png` — mechanism evidence for the
   persistent translation ambiguity.
12. `quantitative-prospective-confirmation.png` — the evidence still required.

## `graphical-abstract-two-view.png`

Suggested role: graphical abstract or opening methods figure. It introduces the
two camera stations, shared spherical scene support, relative rotation and
translation, and the distinction between capture evidence and algorithmic
evidence.

Prompt:

> Use case scientific-educational; wide graphical abstract for academic CV
> paper; exactly two panoramic camera stations in industrial scene; translucent
> spherical domains, baseline, rotation arc, coherent rays to shared 3D points;
> left capture evidence and right algorithm evidence; restrained
> blue-green/amber, light background; no text/numbers/logos/watermark; exactly
> two cameras.

SHA-256: `50b30979449c7d54cb32c42644fcff90590b12eb805a8723a77b04a2e89c64be`;
dimensions: 1672×941 px.

## `capture-boundary-conditions.png`

Suggested role: capture-guidance figure. The four panels illustrate low
parallax, a favorable operating region, insufficient overlap, and concentrated
or unstable evidence. The panels are qualitative and deliberately do not encode
numeric cutoffs.

Prompt:

> Use case: conceptual scientific illustration for an academic computer-vision
> paper about robust two-view relative pose from spherical panoramas. Create a
> wide 16:9 capture-conditions plate, no title and no text, divided into four
> visually clean vertical scenarios separated by subtle whitespace. In every
> scenario show exactly two spherical panoramic camera stations observing one
> coherent 3D scene, with translucent viewing spheres and rays only to genuinely
> shared scene structures. Scenario 1: nearly coincident stations, high overlap
> but very low parallax, subtle amber caution. Scenario 2: moderate baseline,
> rich spatially distributed overlap, healthy parallax, blue-green favorable
> condition. Scenario 3: large displacement with only a narrow shared spatial
> region, amber/red caution for insufficient overlap. Scenario 4: adequate
> overlap but visible scene change/occlusion and features concentrated in one
> small region, amber caution. Use an industrial indoor environment with pipes,
> columns, platforms and distinctive geometry. Precise technical editorial
> illustration, orthographic/isometric hybrid, light neutral background,
> restrained navy, teal, green, amber and coral palette, crisp geometry,
> print-ready, generous margins, no decorative flourishes. Leave a narrow empty
> band below each scenario for later typeset labels. Do not include equations,
> numbers, letters, legends, logos, watermark, dashboards, UI, or more than two
> camera stations per scenario.

SHA-256: `3fd455eaf0fa4d364124b0478ab73c05f30159aa87f0b83299bff470c8e07f06`;
dimensions: 1672×941 px.

## `post-processing-evidence.png`

Suggested role: second-model figure. It connects distributed spherical matches
to the post-processing diagnostics used to predict whether a returned pose will
be precise: coverage, residual concentration, inlier consensus, cheirality, and
pose uncertainty.

Prompt:

> Use case: conceptual methods figure for an academic computer-vision paper on
> two-view spherical relative-pose estimation. Create a wide 16:9 scientific
> editorial illustration with a left-to-right visual story and no text. On the
> left, show exactly two aligned equirectangular panorama strips of the same
> industrial interior, one above the other, with a modest set of
> color-consistent feature correspondences spanning different longitudes,
> elevations and depths; include a few subtle amber outliers. In the center,
> transform those matches into unit bearing rays on exactly two translucent
> spheres with two camera centers, connected through a robust consensus
> geometry; show a clear rotation arc and translation direction arrow but no
> coordinate letters. On the right, show abstract post-processing evidence as
> clean visual glyphs: broad spatial coverage on a spherical grid, a compact
> residual distribution, a strong inlier consensus ring,
> cheirality-consistent points in front of both views, and a small uncertainty
> ellipse around the recovered pose. Visually distinguish trustworthy evidence
> in teal/blue/green and rejected evidence in amber/coral. Light neutral
> background, crisp vector-like geometry with subtle dimensionality,
> publication-quality, restrained palette, consistent line weights, generous
> whitespace and an empty lower margin for typeset caption. Do not include
> words, numbers, equations, logos, watermark, UI chrome, fake performance
> values, or a third camera.

SHA-256: `dc5a7d6a54ed9fe87a79b82342cfa32d5a900f28b87d80302a6081665ac579d7`;
dimensions: 1672×941 px.

## `quantitative-overlap-response.png`

Suggested caption: Observed pose-return, acceptance, and usable-pose rates as a
function of minimum bidirectional registered-cloud overlap. Error bars are 95%
Wilson intervals for pairs; `n` is the pair count and `g` the number of
independence components represented in each bin. The curves are descriptive,
not causal, and show that the same overlap does not imply the same success rate
across capture domains.

SHA-256: `d544bafaeff2d48651de7e5a4c9bc4f2268ebbf70ff6ec15142f137c211787b2`.

## `runtime-overlap-response.png`

Suggested caption: Observed complete two-panorama wall time under the exact
optimized PanorAi 3.5.0 route as a function of registered-cloud overlap.
Markers show the median and upper whiskers the P95; `n` and `g` report pairs
and independent components. The population replay ran on a shared host with
observed contention, so this figure is a diagnostic, not a controlled
performance benchmark. Runtime is not a probability predictor.

SHA-256: `8b7636be0a3a3c28755d6cbeaadf884446ce936f3fb336c4c37552810db6b3ea`.

## `controlled-runtime-overlap-response.png`

Suggested caption: Controlled idle-host runtime for the exact optimized
PanorAi 3.5.0 two-view route. Curves show the median and whiskers the P95 over
three repetitions of five outcome-blind pairs in every dataset × overlap
cell. Repetitions of the same pair are repeated engineering measurements, not
independent scenes; pair and independent-component counts are reported in the
companion JSON summary. Stage decomposition separates convolution-dominated
detection from matching and R,t estimation. Runtime is never used by either
probability model. Generated only after a signed host gate; checksum is
recorded with that run.

## `two-model-probability-story.png`

Suggested caption: Two complementary probability models. The pre-capture model
uses only conditions controllable or visible to the operator and screens a pair
before pose estimation. The post-processing model adds correspondence and
estimator evidence after a pose is returned. Reference R,t errors define
outcomes and are never predictors. A pose can be released only when both gates
are supported for the deployed capture domain and frontend.

SHA-256: `abf1bdb6ca56e8f3202cf7e665167f8452216f1233b994db43fd11c70df532dc`.

## `quantitative-evidence-base.png`

Suggested caption: Frozen evidence base for the spherical two-view study. Bar
heights show unique panoramas, unique unordered two-image pairs, and connected
independence components after pairs sharing an image are forced into the same
split. The strong imbalance in independent support explains why Stanford2D3D
and P74 are failure-discovery domains rather than standalone group-generalized
confirmations.

SHA-256: `98e9e2421ee69273d896c2d01b2e9b2da611292edfe33c0bb0d0ed73e7ecf22d`.

## `quantitative-calibration-heldout.png`

Suggested caption: Component-held-out reliability of the pre-capture
acceptance model and post-processing precision model. Marker area is
proportional to the number of pairs in each fixed probability bin. The P74
post-processing panel exposes overconfidence under a frontend/domain shift.

SHA-256: `1fbdc7f45c61b2a8fc5087c8c1912fcc421a2eb444e5eea1e173675354d77789`.

## `quantitative-post-ablation.png`

Suggested caption: Brier-score ablation of post-processing predictors. The
aligned analysis adds translation-orientation margins, triangulation support,
and ambiguity diagnostics from the unified PanorAi 3.5.0 frontend. Richer
public-dataset diagnostics improve evaluation on the public corpora but are not
available for P74; even common overlap/geometry diagnostics can worsen P74
calibration. Lower is better.

SHA-256: `5406aff6076fa3fb02c095b971a7ac54bbcdad0f5f818d45b8caa340dc074d90`.

## `capture-probability-surface.png`

Suggested caption: Calibrated scanner-assisted capture probabilities over
minimum registered-cloud overlap and baseline quartiles. Panels show
acceptance, precision conditional on acceptance, and their product. Each cell
is evaluated at the observed median capture condition and reports independent
group support; grey cells with fewer than five supporting groups are withheld
rather than presented as reliable interpolation.

SHA-256: `2853678e07a8a2325fa09fe8bef07a9ab8535468dacf00df9197d7d4f73838fd`.

## `quantitative-cross-dataset-transfer.png`

Suggested caption: Component-held-out versus leave-one-dataset-out Brier score
for capture acceptance and post-processing precision. The transfer experiment
never opens outcomes from the target dataset during fitting. Capture acceptance
transfers with moderate degradation; post-processing precision remains strongly
domain-dependent, especially for P74. Lower is better.

SHA-256: `5abde5271083a8feee376233296651e5215e65f67a61feeb9a4f23ebeb5b9e5d`.

## `quantitative-selective-rule-evaluation.png`

Suggested caption: Calibration and untouched-split evaluation of the frozen
selective operating rule. Bars show precision and pair coverage; downward error
bars terminate at the exact one-sided 95% lower confidence bound. Labels give
selected pairs (`n`), independent components (`g`), and catastrophic accepted
poses (`cat`). The calibration rule meets its target, but evaluation lacks
support in Matterport and exposes six catastrophic accepts in Stanford; the
operational verdict is NO-GO.

SHA-256: `2c0143f85a8463c6ca09ea8f233a29f1226e7c89391c374bcd75603958fca067`.

## `quantitative-optimized-main-replay.png`

Suggested caption: Five representative failure and success mechanisms replayed
with the exact optimized PanorAi `v3.5.0` spherical frontend at commit
`03c5b36`. The optimized route
safely returns no pose for the negligible-overlap P74 pair and the historically
wrong Matterport pair, but it still accepts the Stanford repetitive-scene case
with 440 matches and 295 inliers. Its rotation is accurate (0.71 degrees) while
translation direction is catastrophically wrong (112.74 degrees), showing that
match volume and low rotation error do not resolve translation ambiguity. This
five-pair replay is a mechanism check, not population-level validation.

SHA-256: `d8092ff1680b39e248c2edf79318144afc42eb43d3a15698571c7752c0831321`.

## `quantitative-prospective-confirmation.png`

Suggested caption: Frozen beta-binomial power plan for prospective confirmation
at intraclass correlation 0.20 and at most three selected pairs per independent
capture group. Under the declared 97% true-precision planning scenario, 40 new
groups and 120 selected pairs give estimated power 0.808 to meet all three
gates: observed precision at least 95%, an exact one-sided 95% lower bound at
least 90%, and zero catastrophic accepted poses. The non-monotone steps arise
from the discrete exact acceptance criteria.

SHA-256: `418826631ce0d250fe672f69cd8c8724f05e73bb135a24dd614e6d973483d285`.
