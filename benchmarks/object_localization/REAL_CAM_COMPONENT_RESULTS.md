# Real ImageNet CAM object-component pilot

Status: **useful negative boundary; one real pair, not promotion evidence**

This run connected the complete stage-1 path on the frozen Stanford2D3D
`area_2/office_3` pair:

```text
text query -> native spherical CAM -> seam-aware components
-> feature-indexed regions -> geometric association
-> deterministic object hypothesis ID -> scale-free spatial hypothesis
```

The two target classes were the explicit textual query `desk + bookcase`.
Predictions were frozen before the three manual object instances and their
expected cross-view identities were opened. Dataset images and model weights
remain external.

## Main result

The end-to-end path executes and returns deterministic IDs with spatial
hypotheses, but a low-resolution class CAM is not yet a reliable instance
proposal by itself.

| CAM threshold | Connectivity | Regions A/B | Object hypotheses | Correct links | False links | Identity precision | Identity recall | Merged components |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 4 | 2 / 2 | 2 | 2 | 0 | 1.000 | 0.667 | 2 |
| 0.25 | 8 | 2 / 2 | 2 | 2 | 0 | 1.000 | 0.667 | 2 |
| 0.50 | 4 | 2 / 3 | 2 | 0 | 1 | 0.000 | 0.000 | 1 |
| 0.50 | 8 | 2 / 3 | 2 | 0 | 1 | 0.000 | 0.000 | 1 |
| 0.75 | 4 | 5 / 2 | 1 | 0 | 0 | 0.000 | 0.000 | 0 |
| 0.75 | 8 | 4 / 2 | 1 | 0 | 0 | 0.000 | 0.000 | 1 |

At threshold 0.25, the pipeline recovered the work desk and bookcase links
without a false identity. It missed the second desk instance. Both desk CAM
components also overlapped more than one annotated desk instance, so the
perfect precision must not be interpreted as correct instance segmentation.

At threshold 0.50, view B split the desk activation into work-desk and
conference-desk components, but view A retained a merged desk component. The
one-to-one assignment then selected the conference desk in view B for the
merged/work-desk component in view A, creating a false identity. The accepted
bookcase hypothesis was geometrically strong but did not overlap the annotated
bookcase in view B, so it was not counted as a correct object identity.

At threshold 0.75, the components became purer in parts of view A but lost
shared cross-view support. Only one geometrically valid hypothesis remained,
and it did not map to an expected annotated identity.

Connectivity 4 versus 8 made no meaningful difference at thresholds 0.25 and
0.50. At 0.75 it changed the number of fragments but did not recover an
identity. On this pair, threshold sensitivity dominates connectivity choice.

## Geometry is not the limiting stage

Every accepted association produced a localized, scale-free 3D hypothesis.
At threshold 0.25:

- `desk`: 161 regional matches, 114 pose inliers, 0.0236 degree median
  reprojection error, and 17.50 degrees median parallax;
- `bookcase`: 166 regional matches, 112 pose inliers, 0.0254 degree median
  reprojection error, and 17.27 degrees median parallax.

At threshold 0.50, the selected desk link still had 35/36 geometric inliers
and 0.0126 degree median reprojection error, even though its manual identity was
wrong. Geometry can prove that two selected image regions contain a coherent
shared 3D structure; it cannot repair a region proposal that merged different
instances or attached the class activation to contextual background.

## V1 operating contract

The first version should therefore make a narrower claim:

1. A text query selects semantic class maps explicitly; it must not depend on
   top-k ImageNet predictions.
2. CAM agreement may guide pose sampling and propose *candidate semantic
   regions*.
3. A candidate becomes an object-ID hypothesis only after descriptor and
   epipolar/pose consistency checks.
4. The result must expose proposal diagnostics, including component size,
   purity proxies, support, inlier count, parallax, and localization state.
5. A class CAM alone must not be presented as an instance mask. Repeated
   objects require an additional instance-splitting cue.
6. Threshold-dependent outputs should remain hypotheses, not persistent graph
   identities, until corroborated across another view or detector.

The most direct next experiment is to split broad CAM support using the spatial
clusters of geometrically verified matches, then associate those clusters
one-to-one. This keeps the V1 simple: CAM supplies semantic relevance;
descriptor matches and geometry supply instance separation.

## Reproduction

```bash
python benchmarks/object_localization/run_real_cam_object_components.py \
  --preflight

python benchmarks/object_localization/run_real_cam_object_components.py \
  --output-dir /private/tmp/panorai-real-cam-object-components
```

Artifacts and hashes:

- `prediction.json`:
  `0883fe80a3587b5688df2a4ddb923ef01d295adf9b7dc32d6faf93226e484fc3`
- `evaluation.json`:
  `21ad8d1da648f35f11a94e88799308c8f2a25f6a645641e306566ff9c3bcb0a6`
- `summary.csv`:
  `ef3e60a9156d2c0b8ca864053a093c98529329f14c1bf9b4bac6e43eaeb9254c`
- `native_cams.npz`:
  `6a17bfad08abf173afd4598006f29b1c159f81a726a87daf5425c0627d1d66ec`

This single easy indoor pair does not establish generalization, calibrated
thresholds, instance segmentation, metric object location, or graph-ready
identity persistence. It establishes the interface path and identifies
instance proposal as the next bottleneck.
