# Real joint match-cluster pilot

Status: **negative automatic-promotion result; useful diagnostic only**

This run tested whether pose-inlier matches could split the broad real CAM
components from VAL-048 into repeated object instances. The frozen Stanford2D3D
`area_2/office_3` pair contains two desks and one bookcase shared by both
panoramas.

The prediction path was:

```text
text query -> native 7x14 class CAM -> broad component at threshold 0.25
-> accepted broad region association -> joint angular match clusters
-> paired feature-indexed regions -> object IDs + scale-free locations
```

Predictions were frozen before manual instance regions and expected identities
were opened. All six radii used the same images, CAMs, 310 matches, estimated
pose, minimum three-match support, and pose inliers.

## Result

| Radius | Hypotheses | Correct unique links | Unique-link recall | Unmapped hypotheses | Strict hypothesis precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 deg | 4 | 0 | 0.000 | 4 | 0.000 |
| 2 deg | 16 | 1 | 0.333 | 14 | 0.125 |
| 3 deg | 18 | 1 | 0.333 | 16 | 0.111 |
| 5 deg | 24 | 2 | 0.667 | 20 | 0.167 |
| 8 deg | 26 | 2 | 0.667 | 22 | 0.154 |
| 12 deg | 20 | 2 | 0.667 | 17 | 0.150 |

The narrow 1-degree radius fragmented nearly all support into clusters smaller
than three matches. Radii of 2-3 degrees recovered only the work desk. Radii of
5-12 degrees recovered unique links for the work desk and bookcase, but created
20-26 localized object hypotheses. Only 11-17% of the promoted hypotheses
mapped to a correct annotated identity. The second, conference desk was never
recovered as a cross-view identity.

The unique-link precision among the few mappable hypotheses was 1.0, but that
number is not a safe promotion metric: it ignores the 14-22 localized
hypotheses that landed on contextual scene structure outside every annotated
target. The stricter metric counts every promoted hypothesis and remained at
or below 0.167.

## Why the simple clustering fails

The low-threshold `desk` and `bookcase` CAM envelopes are broad and overlap.
Their accepted parent associations contained 114 and 112 pose inliers and
produced nearly identical spatial cluster-size sequences. Geometry correctly
found coherent shared 3D patches, but many patches belonged to walls, floor,
or other context and appeared under both semantic classes.

This distinguishes two claims:

- joint spherical match clustering can split coherent geometry and is useful
  for diagnostics or proposals;
- it does **not** establish that a cluster is an instance of the queried class.

The experiment also confirms that triangulation success is not a semantic
gate. Every promoted cluster in the sweep produced a localized spatial
hypothesis, including the many clusters with no target-instance overlap.

## V1 decision

`cluster_region_association_matches` remains an explicit Experimental helper.
It must not run as an automatic promotion step in the simple V1 pipeline.

For V1, a unique object ID should require one of these additional cues before
promotion:

1. an instance-aware detector or segmentation mask;
2. a region proposal with strong class evidence in both views and a negative
   background/competing-class test; or
3. corroboration across another view that rejects context-only clusters.

The safest simple contract is therefore:

```text
CAM/text semantics -> proposal prior
descriptor + pose geometry -> correspondence validation/localization
instance-aware region cue -> unique object-ID promotion
```

CAM plus geometry alone remains useful for ranking probable areas, explaining
support, and producing candidate spatial hypotheses, but not for persistent
instance identity.

## Reproduction

```bash
python benchmarks/object_localization/run_real_joint_match_clusters.py \
  --preflight

python benchmarks/object_localization/run_real_joint_match_clusters.py \
  --output-dir /private/tmp/panorai-real-joint-match-clusters
```

Artifacts and hashes:

- `prediction.json`:
  `1b23d1af0d0dbfd14b562431b7d53bbf6a803bd370cfe1ba346e942658da9b15`
- `evaluation.json`:
  `ebcadfe5ec892e6c88de9b6f3d7291f1664f631005309f33e835f7e2abf42501`
- `summary.csv`:
  `9bbf9d6d20c9d8123bad5047c97d4c86afa063773dfeec669d2a5ce3290dfd44`
- `native_cams.npz`:
  `6a17bfad08abf173afd4598006f29b1c159f81a726a87daf5425c0627d1d66ec`

This is one easy indoor pair and a post-hoc development study. It does not
calibrate the angular radius, measure broad generalization, or validate a
specific instance detector.
