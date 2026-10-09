# Conceptual figures for the two-view pose study

These raster illustrations support the paper narrative. They are conceptual,
not experimental evidence: they contain no measured values and must not be used
to infer thresholds, probabilities, or estimator performance. Quantitative
figures must be regenerated from the frozen study tables and plotting scripts.

All three assets were generated on 2026-10-08 with the built-in OpenAI image
generation tool. Captions and panel labels should be typeset by the paper build,
not baked into the raster.

The four files prefixed `quantitative-` are measured figures, not generated
illustrations. They were rendered deterministically by
`render_paper_results.py` from the frozen VAL-018 pair table and evaluation
records. Their source tables, model cards, commands, and checksums are recorded
in the corresponding `.agents/results/VAL-018-*` run report.

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

## `quantitative-overlap-response.png`

Suggested caption: Observed pose-return, acceptance, and usable-pose rates as a
function of minimum bidirectional registered-cloud overlap. Error bars are 95%
Wilson intervals for pairs; `n` is the pair count and `g` the number of
independence components represented in each bin. The curves are descriptive,
not causal, and show that the same overlap does not imply the same success rate
across capture domains.

SHA-256: `452d1607a3918b49f1c41e5fd2cc623801204965e19943d2f7c2e3f6ce7ed288`.

## `quantitative-calibration-heldout.png`

Suggested caption: Component-held-out reliability of the pre-capture
acceptance model and post-processing precision model. Marker area is
proportional to the number of pairs in each fixed probability bin. The P74
post-processing panel exposes overconfidence under a frontend/domain shift.

SHA-256: `a8dbf2704dd4ffa07286e75a10620f993d2d9c6dc973b7371d7a963eb6dc5d98`.

## `quantitative-post-ablation.png`

Suggested caption: Brier-score ablation of post-processing predictors. Richer
public-dataset diagnostics improve evaluation on the public corpora but are not
available for P74; even common overlap/geometry diagnostics can worsen P74
calibration. Lower is better.

SHA-256: `3c3c819febc970222744b3fb51f1af0038994fcc2d93b7b4c81f9b1450f08059`.

## `quantitative-cross-dataset-transfer.png`

Suggested caption: Component-held-out versus leave-one-dataset-out Brier score
for capture acceptance and post-processing precision. The transfer experiment
never opens outcomes from the target dataset during fitting. Capture acceptance
transfers with moderate degradation; post-processing precision remains strongly
domain-dependent, especially for P74. Lower is better.

SHA-256: `a09fc92e636564654c32194a98c710d4917691f1e2828e80147cbd5ed72f9946`.
