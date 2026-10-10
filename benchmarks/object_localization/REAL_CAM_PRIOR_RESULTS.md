# Real ImageNet CAM pose-prior pilot

Status: **mixed positive diagnostic; one real pair, not promotion evidence**

This run used the frozen Stanford2D3D `area_2/office_3` pair, cached official
Torchvision ResNet18 weights, native `7x14` spherical CAM lattices, 4,096 and
3,658 spherical features, and 310 unchanged descriptor matches. The baseline
and every guided estimate used the same pose estimator, 1,000-trial budget,
validity mask, and random seed. CAM weights affected only minimal-set proposal
sampling.

The prediction was frozen before the registered pose or manual RGB rectangles
were opened. Dataset images and model weights remain external.

## Main result

The uniform-weight baseline was already strong: 152 inliers, accepted quality,
0.268 degrees rotation error, and 0.283 degrees translation-direction error.
Every target-guided estimate was also accepted and retained 152 inliers. Pose
errors remained between 0.254–0.268 degrees rotation and 0.283–0.303 degrees
translation, so this pair provides no evidence that semantic sampling improves
an already-saturated final pose.

It does show that the textual query changes proposal quality. With a CAM
threshold of 0.5:

| Query | Profile | Supported matches | Reference-inlier mass | Enrichment |
| --- | --- | ---: | ---: | ---: |
| `desk + bookcase` | conservative | 120 | 69.1% | 1.383x |
| `desk + bookcase` | aggressive | 120 | 79.5% | 1.591x |
| `flamingo + volcano` | conservative | 20 | 50.3% | 1.006x |
| `flamingo + volcano` | aggressive | 20 | 50.7% | 1.014x |

The uniform reference-inlier fraction was exactly 50%. Thus, the target query
concentrated proposal mass on geometrically correct matches while the unrelated
textual control was essentially neutral. This is the first real-pair evidence
for keeping the text query in the stage-1 contract.

## CAM limitations

The evidence is weak localization, not segmentation. Requested target classes
were not necessarily top-ranked: `desk` ranked 20 and 93 in the two views;
`bookcase` ranked 16 and 18. A top-3-only pipeline would have missed both
requested concepts. Explicit query-conditioned class extraction is therefore
required.

Manual-region alignment was inconsistent:

- `bookcase` in view A had higher mean CAM inside than outside and 39.2% top-area
  precision against a 17.1% feature-area baseline;
- `bookcase` in view B had lower mean CAM inside and zero top-area precision;
- `desk` mean CAM was lower inside than outside in both views, although its
  top-area precision modestly exceeded the large manual-region baselines.

Despite weak object-mask alignment, cross-view CAM agreement enriched correct
matches. This suggests the CAM currently acts as a scene-structure prior, not
a reliable instance mask. It must not yet be used alone to create object IDs or
object boundaries.

At threshold 0.75, only `bookcase` retained shared match support; `desk` dropped
out. Threshold selection is therefore a major operating parameter and should
remain diagnostic until evaluated across more pairs.

## Reproduction

```bash
python benchmarks/object_localization/run_real_cam_pose_prior.py --preflight

python benchmarks/object_localization/run_real_cam_pose_prior.py \
  --output-dir /private/tmp/panorai-real-cam-pose-prior-final
```

Artifacts and hashes:

- `prediction.json`:
  `270e46bdcaac89f20280577710844a6ed8bb8ffebfca53f1efab5c7f99908abe`
- `evaluation.json`:
  `d1a23359defd89b103443a102997659de93e2cc7b893d62040673404f4b70f09`
- `pose_comparison.csv`:
  `b78ba30f231301a68f2935a486230cc477c4f249077389d8150b609defa77fc2`
- `native_cams.npz`:
  `6a17bfad08abf173afd4598006f29b1c159f81a726a87daf5425c0627d1d66ec`

The result is specific to one easy indoor pair. It does not establish
generalization, instance separation, pose recovery under a failed baseline,
calibrated thresholds, or safe automatic selection between uniform and guided
poses.
