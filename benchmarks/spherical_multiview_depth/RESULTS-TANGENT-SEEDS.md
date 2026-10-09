# Native DoG and tangent-RootSIFT depth-seed result

## Protocol

This is a P74 development diagnostic on W121 with W119 and W124 as optimization
views. The frozen ConvNeXt-L Metric3D radial-range prior, registered metric pose,
RGB, feature configuration, and all gates were fixed before ground truth was
opened. The prediction was hashed in `FROZEN-BEFORE-GT.json` first.

The 4128x8256 ERP was not resized. PanorAi's direct native spherical DoG
detector produced refined keypoints, and the existing descriptor-neutral
tangent path materialized 48x48 patches. The v2 adapter used locally
standardized RootSIFT with radius six detector sigmas, SIFT diameter 1.5
sigmas, and fixed orientation. Matching used reciprocal Lowe 0.72 and bilateral
spherical NMS.

PanorAi's five-point LO-RANSAC plus nonminimal consensus refit was executed as
a pose diagnostic. The depth result intentionally retained registered `R,t`:
the goal of this control is to measure depth matching without mixing in pose
error. The W119 refined-pose diagnostic was accepted and differed from
registration by 0.1675 degrees in rotation and 0.4972 degrees in translation
direction. W124 was correctly quality-rejected: its estimate was wrong by
10.5661/47.4531 degrees. This is direct evidence that the registered-pose
control and the refined-pose ablation must remain separate.

## Result

W121/W119/W124 extraction returned 4,041/4,035/4,040 features. W119 yielded 32
mutual matches and eight accepted depth seeds; W124 yielded 27 mutual matches
and three accepted seeds. All nine accepted seeds that landed on valid
evaluation ground truth improved:

| Support | CNN prior AbsRel | Tangent seed AbsRel | Improved |
| --- | ---: | ---: | ---: |
| Nine sparse GT-valid seeds | 0.250384 | **0.096588** | **9/9** |
| 10,302 propagated evaluation pixels | 0.144156 | **0.075559** | **90.03%** |

This confirms the geometric depth solve locally. It does not yet produce a
materially different full panorama because only 10,302 of 23,122,418 evaluation
pixels changed (0.0446%).

| Full native map | CNN seed | DoG + tangent seeds |
| --- | ---: | ---: |
| AbsRel | 0.310326 | **0.310283** |
| Scale-aligned relative 3D RMSE | 0.400933 | **0.400845** |
| Scale-invariant log RMSE | 0.374323 | **0.374307** |
| Mean normal error | **48.2036 deg** | 48.2400 deg |
| ERP seam MAE | 0.089395 m | 0.089395 m |

The slight normal regression is consistent with sparse correction footprints
ending inside a much larger unchanged field. It is a warning against claiming
that the current support splat is a satisfactory dense regularizer.

## Interpretation

The earlier dense photometric route failed mainly because its per-pixel signal
was ambiguous. The tangent-RootSIFT route gives much stronger local geometry:
all evaluable accepted seeds improved. Its present failure mode is coverage.
The next justified experiment is not to loosen geometric gates blindly, but to
retain explicit source agreement and solve an edge-aware continuous residual
field whose sparse unary constraints are these audited seeds. Pose-refined and
registered-pose variants must remain separate ablations.

## Artifacts

The primary two-source run is stored at
`/private/tmp/panorai-val030-tangent-depth-seeds-two-source`:

- `results.json`: complete configuration, pose diagnostic, metrics, and hashes;
- `P-74+MD-04_concluido_408+W_121-tangent-depth-seeds.npz`: sparse seed audit;
- native radial maps, residual/weight/support arrays, 15 m PNG, and two binary
  PLY clouds;
- `viewer.html`: self-contained browser viewer for CNN and tangent-seed clouds;
- `p74-w121-tangent-depth-seeds-panel.png`: decimated visual comparison only.

P74 bytes remain external. This is source-checkout benchmark evidence, not an
installed-wheel, public API, or publication claim.
