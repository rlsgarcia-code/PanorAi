# PanorAi 3.5.0 spherical two-view R,t: aligned retrospective results

Status: **NO_GO**. This is retrospective validation, not prospective
confirmation or a release-reliability claim. Dense stereo, bundle adjustment,
and multiview estimation are outside this analysis; every estimate uses exactly
two spherical panoramas.

## Study question and artifact

The study estimates two complementary quantities:

```text
p_usable_capture = P(accepted | capture) × P(precise | accepted, capture)
p_precise_post = P(precise | returned, algorithm evidence)
```

The aligned replay used PanorAi `3.5.0`, source commit
`03c5b36b28225b24d3909286bf53250d7b532aa3`, native spherical DoG batch-two
detection, explicit validity masks, four tangent-patch workers, calibrated
tangent RootSIFT, and the frozen spherical R,t estimator. The primary post
model is `post-precise-aligned-orientation`.

## Evidence base and leakage control

| Dataset | Unique images | Pairs | Independent groups | Evaluation groups |
| --- | ---: | ---: | ---: | ---: |
| Matterport360 | 3,305 | 1,890 | 63 | 9 |
| Stanford2D3D | 638 | 450 | 3 | 1 |
| P74 | 74 | 45 | 3 | 1 |

The total is 4,017 unique images, 2,385 unordered pairs, and 69 independence
components. Duplicate/reversed pairs, shared-image split leakage, component
split leakage, and group split leakage are all zero. Fitting commands received
development/calibration outcomes only; evaluation outcomes were opened by a
separate evaluation stage.

## Aligned estimator response

| Dataset | Pairs | Returned | Accepted | Precise | Usable | Catastrophic accepts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matterport360 | 1,890 | 980/1,890 (51.9%) | 745/1,890 (39.4%) | 752/1,890 (39.8%) | 698/1,890 (36.9%) | 2 |
| Stanford2D3D | 450 | 207/450 (46.0%) | 115/450 (25.6%) | 102/450 (22.7%) | 91/450 (20.2%) | 11 |
| P74 | 45 | 21/45 (46.7%) | 11/45 (24.4%) | 13/45 (28.9%) | 10/45 (22.2%) | 0 |

`precise` means rotation error ≤1° and oriented translation-direction error
≤5°. `usable` means accepted and precise. Translation magnitude is not scored,
because central two-view geometry recovers translation direction only up to
scale.

## Response versus registered-cloud overlap

| Dataset | Overlap | Pairs | Groups | Return rate | Accept rate | Usable rate | Catastrophic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matterport360 | <10% | 1,072 | 59 | 0.225 | 0.103 | 0.083 | 2 |
| Matterport360 | 10–25% | 277 | 47 | 0.783 | 0.545 | 0.505 | 0 |
| Matterport360 | 25–50% | 407 | 51 | 0.953 | 0.885 | 0.848 | 0 |
| Matterport360 | 50–70% | 100 | 34 | 1.000 | 0.920 | 0.920 | 0 |
| Matterport360 | ≥70% | 34 | 17 | 1.000 | 0.941 | 0.941 | 0 |
| Stanford2D3D | <10% | 212 | 3 | 0.137 | 0.005 | 0.005 | 0 |
| Stanford2D3D | 10–25% | 47 | 3 | 0.426 | 0.149 | 0.064 | 1 |
| Stanford2D3D | 25–50% | 46 | 3 | 0.522 | 0.130 | 0.130 | 0 |
| Stanford2D3D | 50–70% | 36 | 3 | 0.833 | 0.583 | 0.472 | 1 |
| Stanford2D3D | ≥70% | 109 | 3 | 0.954 | 0.734 | 0.587 | 9 |
| P74 | <10% | 9 | 3 | 0.000 | 0.000 | 0.000 | 0 |
| P74 | 10–25% | 9 | 3 | 0.333 | 0.000 | 0.000 | 0 |
| P74 | 25–50% | 9 | 3 | 0.222 | 0.222 | 0.111 | 0 |
| P74 | 50–70% | 9 | 3 | 0.889 | 0.333 | 0.333 | 0 |
| P74 | ≥70% | 9 | 3 | 0.889 | 0.667 | 0.667 | 0 |

Overlap is a scanner-assisted capture variable only when the two depth clouds
and their shared coordinate transforms are available before RGB pose
estimation. It must not be advertised as an RGB-only observable without a
separately validated online proxy.

## Supported capture probability surface

| Overlap | Baseline (m) | Support groups | P(accept) | P(precise\|accept) | P(usable) |
| --- | --- | ---: | ---: | ---: | ---: |
| 10–25% | 0.10–1.83 | 18 | 0.619 | 0.931 | 0.576 |
| 25–50% | 0.10–1.83 | 37 | 0.940 | 0.961 | 0.903 |
| 50–70% | 0.10–1.83 | 25 | 0.993 | 0.977 | 0.970 |
| ≥70% | 0.10–1.83 | 15 | 0.991 | 0.957 | 0.948 |
| <10% | 1.83–3.84 | 10 | 0.035 | 0.849 | 0.030 |
| 10–25% | 1.83–3.84 | 22 | 0.219 | 0.881 | 0.193 |
| 25–50% | 1.83–3.84 | 21 | 0.538 | 0.896 | 0.482 |
| 50–70% | 1.83–3.84 | 9 | 0.787 | 0.892 | 0.702 |
| ≥70% | 1.83–3.84 | 5 | 0.939 | 0.910 | 0.854 |
| <10% | 3.84–6.96 | 10 | 0.002 | 0.733 | 0.002 |
| 10–25% | 3.84–6.96 | 8 | 0.037 | 0.775 | 0.029 |
| 25–50% | 3.84–6.96 | 6 | 0.149 | 0.794 | 0.118 |

Cells with fewer than five independent groups are withheld rather than
interpolated into capture advice.

## Calibrated component-held-out evaluation

| Model | Dataset | n | Brier | Log loss | ECE | Predicted | Observed | Groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Capture acceptance | Matterport360 | 270 | 0.073 | 0.272 | 0.077 | 0.300 | 0.356 | 9 |
| Capture acceptance | Stanford2D3D | 150 | 0.164 | 0.733 | 0.175 | 0.422 | 0.320 | 1 |
| Capture acceptance | P74 | 15 | 0.212 | 0.572 | 0.254 | 0.422 | 0.267 | 1 |
| Capture conditional precision | Matterport360 | 96 | 0.039 | 0.158 | 0.022 | 0.936 | 0.958 | 7 |
| Capture conditional precision | Stanford2D3D | 48 | 0.316 | 1.090 | 0.307 | 0.926 | 0.646 | 1 |
| Capture conditional precision | P74 | 4 | 0.174 | 0.490 | 0.199 | 0.901 | 0.750 | 1 |
| Post raw score | Matterport360 | 127 | 0.055 | 0.185 | 0.058 | 0.793 | 0.787 | 8 |
| Post raw score | Stanford2D3D | 89 | 0.136 | 0.423 | 0.131 | 0.461 | 0.416 | 1 |
| Post raw score | P74 | 8 | 0.070 | 0.229 | 0.175 | 0.633 | 0.500 | 1 |
| Post primary | Matterport360 | 127 | 0.053 | 0.171 | 0.045 | 0.787 | 0.787 | 8 |
| Post primary | Stanford2D3D | 89 | 0.120 | 0.388 | 0.109 | 0.512 | 0.416 | 1 |
| Post primary | P74 | 8 | 0.032 | 0.149 | 0.129 | 0.578 | 0.500 | 1 |

The capture usable product is evaluated directly against `accepted AND
precise`; it is not assumed calibrated merely because its two factors were
calibrated separately.

| Dataset | n | Brier | Log loss | ECE | Predicted | Observed | Groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matterport360 | 270 | 0.072 | 0.258 | 0.062 | 0.286 | 0.341 | 9 |
| P74 | 15 | 0.153 | 0.418 | 0.198 | 0.382 | 0.200 | 1 |
| Stanford2D3D | 150 | 0.208 | 0.758 | 0.216 | 0.398 | 0.207 | 1 |

## Leave-one-dataset-out transfer

No outcome from the named target dataset was opened during its fit.

| Model | Held-out dataset | n | Brier | Log loss | ECE | Groups |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Capture acceptance | Matterport360 | 270 | 0.083 | 0.307 | 0.090 | 9 |
| Capture acceptance | Stanford2D3D | 150 | 0.178 | 0.832 | 0.178 | 1 |
| Capture acceptance | P74 | 15 | 0.216 | 0.592 | 0.260 | 1 |
| Post precision | Matterport360 | 127 | 0.059 | 0.191 | 0.066 | 8 |
| Post precision | Stanford2D3D | 89 | 0.121 | 0.394 | 0.109 | 1 |
| Post precision | P74 | 8 | 0.030 | 0.145 | 0.126 | 1 |

Stanford2D3D and P74 each have only three independent groups in the complete
corpus and one evaluation group. Their results diagnose transfer and failure
modes; they do not support standalone group-generalized reliability claims.

## Replay wall-time diagnostic

Resolution is 1024×2048 per panorama. Times refer to a complete two-panorama
pair. These observations were collected on a shared host with independently
observed contention and are therefore **not a controlled performance
benchmark**.

| Dataset | Pairs | Detection median | Detection P95 | Total median | Total P95 | RSS P95 MiB | RSS max MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matterport360 | 1,890 | 4.17 | 6.11 | 8.36 | 15.72 | 780.2 | 913.9 |
| Stanford2D3D | 450 | 4.70 | 14.83 | 8.74 | 27.04 | 754.1 | 819.9 |
| P74 | 45 | 6.22 | 8.43 | 14.45 | 22.01 | 2,620.5 | 2,640.5 |

Runtime and memory are engineering outcomes only; they are not probability
model inputs. These population-replay values remain diagnostic because the
replay ran on a shared host.

## Controlled timing result

The frozen controlled protocol completed all 225 observations: 75
outcome-blind pairs, five pairs in every dataset × overlap cell, and three
repetitions. The signed gate recorded an Apple M3 Max with 16 physical/logical
cores, 64 GiB RAM, 30 GiB free, AC power, no thermal/performance warning,
Python 3.12.4, NumPy 1.26.4, OpenCV 4.11.0 reporting 16 threads, and four patch
workers. Every pair used a fresh process and the exact PanorAi 3.5.0 wheel.

| Dataset | Obs / pairs | Detection pair median / P95 | Image-ready per image median / P95 | Patches pair median / P95 | R,t median / P95 | Complete pair median / P95 | RSS median / P95 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matterport360 | 75 / 25 | 4.315 / 5.129 | 3.210 / 4.145 | 1.713 / 2.958 | 2.420 / 3.233 | 8.820 / 11.285 | 659.9 / 710.4 |
| Stanford2D3D | 75 / 25 | 4.666 / 5.327 | 3.155 / 4.179 | 1.391 / 2.079 | 2.249 / 2.736 | 8.676 / 10.368 | 706.3 / 748.9 |
| P74 | 75 / 25 | 5.731 / 6.584 | 5.607 / 7.869 | 4.472 / 7.889 | 2.065 / 2.608 | 13.225 / 15.829 | 1,739.0 / 2,625.9 |

Against the 1024×2048 reference of 4.22 s detection per pair, 3.44 s
image-ready per image, and 8.99 s per complete pair, Matterport360 and
Stanford2D3D remain within the same order of magnitude and below the frozen
1.5× diagnostic boundary. P74 detection is 1.36× and complete-pair time is
1.47× the reference, while image-ready time is 1.63× and therefore requires a
stage diagnosis.

The excess is localized to patch/descriptor work rather than spherical
convolution or R,t estimation. P74 has a median of 5,430 combined keypoints per
pair, versus 2,045 for Matterport360 and 1,666 for Stanford2D3D. Its median
patch time is 2.61× Matterport360 and 3.21× Stanford2D3D, and median RSS is
about 1.74 GiB. Detection remains 2.87 s per image and pose estimation 2.07 s
per pair. Runtime is not monotonic in overlap inside any dataset; scene/domain
feature load dominates the timing differences, so overlap must not be used as
a causal runtime predictor.

## Selective operating rule

The calibration-only rule requires overlap ≥50%, baseline in
[0.162, 2.155] m, capture usable probability ≥0.50, and post precision
probability ≥0.50.

| Dataset | Selected/eligible | Groups | Precision | Exact lower 95% | Catastrophic | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Matterport360 | 11/270 | 5 | 1.000 | 0.762 | 0 | no |
| Stanford2D3D | 33/150 | 1 | 0.727 | 0.572 | 6 | no |
| P74 | 2/15 | 1 | 1.000 | 0.224 | 0 | no |

The retrospective verdict cannot itself authorize a release. A favorable
`GO_FOR_PROSPECTIVE_CONFIRMATION` result would authorize only E8 collection.

## Capture recommendation

1. Use exactly two calibrated spherical panoramas for the claimed estimator.
2. With registered depth in a shared frame, require at least 50% minimum
   bidirectional cloud overlap as an eligibility boundary; prefer ≥70% where
   the table shows independent support, but do not treat either value as a
   guarantee.
3. Plan or measure baseline independently of the reference pose, and use it
   only inside the calibrated scene-scale support. Record representative scene
   distance or predicted parallax when available.
4. Prefer static structure distributed broadly over the sphere, depth
   variation, and non-negligible parallax. Repetitive structure can preserve a
   strong-looking match set while reversing translation direction.
5. Preserve explicit masks; black pixels are not validity evidence.
6. Release a pose only when public quality acceptance, the supported capture
   envelope, and the frozen post-processing probability gate all agree.
7. Reacquire rather than extrapolate outside any calibrated support region.

For pure RGB capture, the 50% cloud-overlap rule is unavailable until an online
RGB overlap proxy is independently validated.

## Prospective confirmation still required

The frozen primary design requires 40 new groups and 120 selected pairs. The
confirmatory gate also requires observed selected-pose precision ≥95%, a
one-sided exact 95% lower bound ≥90%, zero catastrophic accepts, at most three
selected pairs per independent group, and predictions sealed before reference
poses are opened.

## Figures

- [Response versus overlap](figures/quantitative-overlap-response.png)
- [Supported capture probability surface](figures/capture-probability-surface.png)
- [Component-held-out calibration](figures/quantitative-calibration-heldout.png)
- [Leave-one-dataset-out transfer](figures/quantitative-cross-dataset-transfer.png)
- [Post-processing model ablation](figures/quantitative-post-ablation.png)
- [Controlled runtime versus overlap](figures/controlled-runtime-overlap-response.png)
- [Shared-host runtime diagnostic](figures/runtime-overlap-response.png)
- [Selective-rule evaluation](figures/quantitative-selective-rule-evaluation.png)

The eight quantitative files, dimensions, content hashes, PanorAi identity,
analysis-manifest digest, and independent-verification digest are sealed in
[`aligned-quantitative-figures.json`](figures/aligned-quantitative-figures.json).

## Reproducibility record

- PanorAi version: `3.5.0`;
- source commit: `03c5b36b28225b24d3909286bf53250d7b532aa3`;
- wheel SHA-256:
  `e861dafbaa5991aef77dd512b3ef1bf6fdc7967d10fbd236d850bab5c1a5f8a7`;
- population replay status SHA-256:
  `ef0b3ec8f5be6a3619a499968c5e401d08ef2f48de5cb2e610d59af419100600`;
- aligned analysis manifest SHA-256:
  `7caddd6c3bfb5e2375035cfcc3161c0233965842f37f23db286e834b13c645b9`;
- paper-results data SHA-256:
  `f1de4d702efa0a4641784ae8539362a95226ca6f6397be38631380b44820281b`;
- independent verification: PASS 59/59, SHA-256
  `2c9d86f4da5bdc35d6fb3645d0222ac171f33efd861581698257778c7b0e678e`;
- controlled timing host gate SHA-256:
  `6d4d9b0f379194817f015ba8474d7667a4a0daac44f41ef5d856ac379265bcf3`;
- controlled timing summary SHA-256:
  `80aaeabe9e8da9d78003af38482a514655e0aa057bc33874b0c2f49e8d4b97ee`;
- controlled observations SHA-256:
  `648454223db730e9adf7c2f3a343d6878dead7948c1177ec1c60eda6f0a9334b`.

Commands and artifact layout are documented in [README.md](README.md). The
scientific protocol, data contract, capture standard, failure taxonomy, and
prospective protocol are respectively documented in
[STUDY_PROTOCOL.md](STUDY_PROTOCOL.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md),
[CAPTURE_STANDARD.md](CAPTURE_STANDARD.md),
[FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md), and
[PROSPECTIVE_CONFIRMATION_PROTOCOL.md](PROSPECTIVE_CONFIRMATION_PROTOCOL.md).

## Limitations

- This is retrospective validation on frozen groups, not E8 confirmation.
- The scanner-assisted overlap model is not an RGB-only deployment model.
- Stanford2D3D and P74 have insufficient independent groups for standalone
  probability claims.
- Baseline in the archived corpora is a retrospective surrogate for a
  quantity that must be planned or tracked independently at deployment.
- No result in this document validates dense stereo, translation magnitude,
  bundle adjustment, or multiview estimation.
- Population-replay wall times remain diagnostic; the separate signed-gate
  controlled timing result is the performance evidence.
- P74 image-ready time exceeds the reference by more than 1.5× because of its
  larger retained-keypoint and tangent-patch workload; this is a domain/profile
  cost and not evidence that higher overlap itself causes longer runtime.
