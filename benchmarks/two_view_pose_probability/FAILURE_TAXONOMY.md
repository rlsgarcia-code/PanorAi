# Real-pair failure taxonomy

Status: historical pre-alignment mechanism evidence over all 2,385 frozen
pairs. The population counts below combine the previously archived frontends;
they are not PanorAi 3.5.0 population-performance claims. The categories
overlap and must not be summed as mutually exclusive states. A new aligned
taxonomy will be generated only after all 2,385 exact-wheel results complete.

| Category | Matterport360 | Stanford2D3D | P74 | Total |
|---|---:|---:|---:|---:|
| No pose with <10% registered-cloud overlap | 181 | 1 | 9 | 191 |
| Pose returned but rejected by public quality policy | 1,081 | 377 | 9 | 1,467 |
| Catastrophic pose accepted | 1 | 11 | 0 | 12 |
| Returned imprecise pose with post probability ≥0.80 | 7 | 4 | 1 | 12 |
| Usable precise pose with overlap ≥50% | 118 | 48 | 11 | 177 |

## Representative evaluation pairs

### A. Negligible shared scene, no returned pose

- Dataset: P74.
- Pair: `G049 → G069` in acquisition family `MD-05_concluido_326`.
- Registered-cloud overlap: 0.0%; matches: 9.
- Interpretation: the two images are real captures but do not form a valid
  reconstruction pair. The correct system response is to refuse R,t rather than
  force a pose from appearance coincidences.

### B. Wrong pose returned and rejected

- Dataset: Matterport360.
- Pair: `essential-stat-0179`.
- Overlap: 0.0%; matches: 128; post probability: 0.058.
- Errors: rotation 179.68°, translation direction 90.08°.
- Interpretation: the estimator can produce a finite hypothesis in a
  non-overlapping regime, but the existing post evidence correctly makes it
  low-confidence and the public quality policy rejects it.

### C. Catastrophic translation accepted with high confidence

- Dataset: Stanford2D3D, auditorium.
- Pair: `essential-stat-1964`.
- Overlap: 84.1%; matches: 895; post probability: 0.937.
- Errors: rotation 0.73°, translation direction 117.38°.
- Interpretation: high overlap, many matches, and an accurate rotation do not
  guarantee translation direction. The visually repetitive auditorium creates
  a symmetry/degeneracy that the current diagnostics fail to reject. This pair
  is a direct counterexample to an overlap-only or match-count release rule.
- Exact-v3.5.0 replay: the isolated PanorAi 3.5.0 optimized spherical route still
  accepted the pose with 440 matches, rotation error 0.71°, and translation
  error 112.74°. The mechanism therefore survives the frontend replacement.

### D. High confidence just outside the precise-pose definition

- Dataset: P74.
- Pair: `G041 → G042` in acquisition family `MD-05_concluido_326`.
- Overlap: 42.9%; matches: 63; post probability: 0.846.
- Errors: rotation 1.73°, translation direction 2.28°.
- Interpretation: translation is accurate, but rotation exceeds the frozen 1°
  precise threshold. This is not catastrophic; it demonstrates why calibrated
  probability must refer to an explicit accuracy definition.

### E. Supported precise outcome

- Dataset: Stanford2D3D, office.
- Pair: `essential-stat-2024`.
- Overlap: 97.6%; matches: 332; post probability: 0.990.
- Errors: rotation 0.53°, translation direction 0.63°.
- Interpretation: distributed shared structure, high overlap, and coherent
  diagnostics coincide with an accurate pose. It is a positive comparator, not
  proof that the same signals suffice in every scene.

## Academic implication

The main residual failure is no longer simply “too few matches.” There are
three qualitatively different boundaries:

1. **pair invalidity** — there is insufficient shared scene, so no two-view
   pose should be attempted;
2. **estimator rejection** — a pose is returned, but diagnostics recognize weak
   or competing geometry;
3. **false confidence** — overlap and correspondence support look strong while
   translation remains ambiguous because scene structure is repetitive or
   degenerate.

Capture guidance addresses the first boundary. Post-processing confidence
addresses the second. The third requires better degeneracy/model-competition
evidence or a capture instruction that introduces discriminative depth and
angular structure; simply increasing overlap is insufficient.

## Figure and distribution note

`build_failure_taxonomy.py` renders a local contact sheet from the original
dataset files and records every source hash. The raster is intentionally stored
only in ignored experiment evidence because redistribution rights for the
Matterport360, Stanford2D3D, and P74 source images have not been audited. The
tracked paper package contains pair IDs, numerical summaries, and the
reproduction script, but not the dataset pixels.
