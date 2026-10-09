# Two-view spherical relative pose: current paper-ready results

Status: retrospective validation complete through E7; prospective confirmation
not yet collected. The current operational verdict is **NO-GO** for a universal
release rule.

The paper-ready visual sequence, figure roles, captions, and content hashes are
indexed in [`figures/FIGURES.md`](figures/FIGURES.md). Quantitative figures are
script-generated from frozen evidence; generated conceptual art is explicitly
marked as non-evidentiary.

## Study question

For exactly two spherical panoramas, can PanorAi estimate relative rotation and
translation direction reliably enough that capture-time conditions predict
acceptance and post-processing evidence predicts whether a returned pose is
precise?

The study separates two probabilities:

```text
p_usable_capture = P(accepted | capture) × P(precise | accepted, capture)
p_precise_post = P(precise | returned, capture, algorithm evidence)
```

The first model contains only quantities available to the operator or capture
system. The second adds evidence produced by detection, matching, and pose
estimation. Reference pose errors are outcomes only.

## Evidence base

| Dataset | Unique images | Two-view pairs | Independent components | Evaluation components |
|---|---:|---:|---:|---:|
| Matterport360 | 3,305 | 1,890 | 63 | 9 |
| Stanford2D3D | 638 | 450 | 3 | 1 |
| P74 native polar | 74 | 45 | 3 | 1 |
| **Total** | **4,017** | **2,385** | **69** | **11** |

Pairs sharing an image are assigned to the same component and split. Model
fitting never receives evaluation outcomes. Stanford2D3D and P74 have only one
evaluation component each and therefore provide failure-discovery and external
replication evidence, not standalone group-generalized inference.

## Spatial-overlap result

Spatial overlap is the minimum of the two directional registered-cloud support
fractions at a 0.25 m surface tolerance. The public corpora use 4,096 equal-area
ERP depth samples per view; P74 uses 6,000 scanner samples. The metric is
available operationally only when depth clouds are already registered in a
shared frame.

Observed response is strongly increasing but not domain invariant:

| Dataset | Usable below 10% | Usable 25–50% | Usable 50–70% | Usable ≥70% |
|---|---:|---:|---:|---:|
| Matterport360 | 56/1,072 | 295/407 | 86/100 | 32/34 |
| Stanford2D3D | 0/212 | 2/46 | 4/36 | 44/109 |
| P74 | 0/9 | 1/9 | 4/9 | 7/9 |

Consequently, 50% registered-cloud overlap is defensible as a provisional
minimum capture-eligibility boundary, but not as a universal guarantee of pose
quality. Even at ≥70%, the observed usable rate ranges from 40.4% in Stanford
to 94.1% in Matterport.

## Probability-model result

The common capture model uses minimum registered-cloud overlap and planned or
measured baseline. On evaluation components its acceptance Brier scores are
0.0868 for Matterport, 0.1038 for Stanford, and 0.1384 for P74. Leave-one-
dataset-out scores are 0.1385, 0.1270, and 0.1385 respectively. Capture
acceptance therefore transfers with moderate degradation.

Post-processing precision is more domain sensitive. A common model using raw
quality, match/inlier support, overlap, baseline, parallax, and cheirality has
evaluation Brier scores of 0.0206, 0.0576, and 0.3422. In P74 it predicts mean
precision 0.875 while only 0.500 of the eight returned evaluation poses are
precise. Leave-one-dataset-out training raises the P74 Brier score to 0.3826.

The raw-score-only post model transfers less badly than the richer common
model, but is still not a calibrated universal confidence measure. This is
consistent with the frozen frontend mismatch: the public corpora use the
historical frontend, while P74 uses the optimized spherical DoG and RootSIFT
route. A five-pair mechanism replay described below shows that this mismatch is
not sufficient to explain the critical repetitive-scene failure.

## Exact PanorAi `v3.5.0` mechanism replay

Five representative pairs were rerun through an isolated PanorAi 3.5.0 wheel
whose release tag peels to commit `03c5b36`. That commit was `origin/main` when
the protocol was frozen. The route used
`SphericalDoGDetector.detect_batch()` with batch two, native convolution,
4,096-keypoint capacity, explicit validity masks, four patch workers, and the
calibrated tangent RootSIFT profile. No source-checkout PanorAi import, NumPy
convolution fallback, multiface extraction, or sequential detection was used.

| Mechanism | Dataset | Matches | Optimized-main outcome | R error | t error |
|---|---|---:|---|---:|---:|
| negligible overlap | P74 | 6 | no pose | — | — |
| wrong historical pose | Matterport | 2 | no pose | — | — |
| repetitive-scene catastrophe | Stanford | 440 | accepted catastrophic | 0.71° | 112.74° |
| precise-threshold near miss | P74 | 43 | accepted imprecise | 1.64° | 2.77° |
| supported success | Stanford | 138 | accepted precise | 0.31° | 0.25° |

Median detection was 4.84 s per pair and median complete processing was 9.59 s
per pair at 1024×2048, consistent with the optimized-route reference order of
magnitude. This sample is a mechanism check, not population-level validation.
It nevertheless proves that the Stanford catastrophic translation ambiguity
persists under the best spherical frontend and cannot be dismissed as an
artifact of the historical detector. Better translation degeneracy/model-
competition evidence is required.

## Frozen selective rule and evaluation

The E7 rule was selected from calibration outcomes only. It requires:

- public quality acceptance;
- registered-cloud overlap ≥50%;
- baseline within the Matterport-supported interval 0.162–2.155 m;
- `p_usable_capture ≥ 0.50`;
- raw-score post-processing `p_precise_post ≥ 0.80`.

The baseline interval is a corpus support interval, not a universal physical
law. A scale-independent future rule should use baseline/depth or predicted
parallax after those variables are made available consistently in P74.

Calibration selected 37/270 pairs from seven Matterport components. All 37 were
precise, the exact one-sided 95% lower confidence bound was 0.922, and no
catastrophic pose was accepted. The untouched evaluation result did not retain
that evidence:

| Evaluation dataset | Selected / eligible | Components | Precision | Exact lower 95% | Catastrophic |
|---|---:|---:|---:|---:|---:|
| Matterport360 | 10/270 | 4 | 1.000 | 0.741 | 0 |
| Stanford2D3D | 25/150 | 1 | 0.880 | 0.718 | 2 |
| P74 | 2/15 | 1 | 1.000 | 0.224 | 0 |

Matterport retains perfect observed precision but has insufficient selected
pair and component support. Stanford exposes both imprecision and catastrophic
acceptance. P74 is too small for inference. The rule therefore receives a
**NO-GO** verdict and cannot justify a reliable multi-domain release.

## Capture recommendation supported now

The evidence supports a capture protocol, not a reliability certification:

1. Estimate pose from exactly two panoramas; do not describe this evidence as
   multiview geometry.
2. When registered depth is available, reject capture pairs below 50% minimum
   bidirectional cloud overlap before running the RGB pose estimator.
3. Prefer higher overlap, broadly distributed static structure, visible depth
   variation, and non-negligible parallax; ≥70% overlap improves the observed
   response but remains domain dependent.
4. Record the planned/measured baseline and representative scene distance. Do
   not reuse the Matterport metre interval in another domain without validating
   the scene scale.
5. Preserve explicit validity masks. Pixel intensity or black regions are not
   validity evidence.
6. Release a pose only when both the capture envelope and a post-processing
   confidence gate are supported for the deployed frontend/domain. At present,
   no such universal gate is validated across all three datasets.
7. If any input lies outside the calibrated support, reacquire the pair rather
   than extrapolate a probability.

For pure RGB capture, registered-cloud overlap is unavailable. A separately
validated online RGB overlap proxy is required before the 50% rule can be
advertised without scanner assistance.

## Prospective evidence required

The frozen beta-binomial planning scenario assumes 97% true selected-pose
precision, intraclass correlation 0.20, at most three selected pairs per new
independent group, marginal catastrophic rate 0.1%, and an 80% chance of
meeting all of the following:

- observed precision ≥95%;
- exact one-sided 95% lower bound ≥90%;
- zero catastrophic accepted poses.

It requires at least **40 new independent groups and 120 selected pairs**.
Because retrospective selection coverage ranged from 3.7% to 16.7%, accrual
must be sequential and counted by selected pairs, not merely by captured
pairs. All predictions must be frozen before reference poses are opened.

## Interpretation

The main scientific result is not that one overlap threshold solves spherical
relative pose. It is that the capture and algorithm stages answer different
questions and fail differently. Overlap and baseline can screen whether a pair
is plausible; correspondence and estimator diagnostics can assess a returned
pose. Neither stage can compensate for unsupported conditions or frontend
domain shift. A trustworthy PanorAi release therefore needs both gates,
domain-aligned calibration, and prospective independent-group evidence.
