# Prospective image and pair registry contract

Status: outcome-blind collection contract. It does not authorize E8.

## Boundary

The registry exists before frontend execution, post-processing diagnostics, or
reference R,t are visible. It records the exact number of unique panoramas and
candidate two-image estimations rather than inferring sample size from pair
counts.

Every registered variable is either controlled by the capture operator or
visible/computable at capture time:

- group, domain stratum, sensor, timestamp, resolution, and image identity;
- an explicit validity-mask identity and valid fraction;
- blur, clipped-pixel fraction, and spherical texture occupancy per image;
- registered-cloud overlap, tracked or planned baseline, representative scene
  distance, baseline/depth ratio, predicted parallax, and RGB similarity per
  pair.

The raw scanner transforms used upstream to register clouds do not enter the
model registry. Only the declared capture scalars enter. Reference rotation,
translation direction, pose-error fields, matches, inliers, estimator
acceptance, selection decisions, and model probabilities are recursively
forbidden.

## Image rows

`images.jsonl` contains one row per unique panorama:

```json
{
  "schema": "panorai-two-view-prospective-image/v1",
  "image_id": "site-001/pano-0001",
  "group_id": "site-001",
  "domain_id": "p74-like-industrial",
  "sensor_id": "scanner-01",
  "capture_timestamp": "2026-10-09T12:00:00Z",
  "panorama_sha256": "<64 lowercase hex characters>",
  "validity_mask_sha256": "<64 lowercase hex characters>",
  "width": 4096,
  "height": 2048,
  "validity_source": "explicit-mask-file",
  "valid_fraction": 0.998,
  "blur_score": 12.4,
  "clipped_fraction": 0.018,
  "texture_occupancy": 0.74
}
```

Validity must not be derived from black pixels. An image ID can belong to only
one independent group and one domain stratum.

## Candidate-pair rows

`pairs.jsonl` contains one row per registered two-view input:

```json
{
  "schema": "panorai-two-view-prospective-pair-candidate/v1",
  "pair_id": "site-001/pair-0001",
  "group_id": "site-001",
  "domain_id": "p74-like-industrial",
  "image_ids": ["site-001/pano-0001", "site-001/pano-0004"],
  "capture": {
    "registered_cloud_overlap_min": 0.68,
    "baseline_m": 0.72,
    "baseline_depth_ratio": 0.14,
    "representative_scene_distance_m": 5.1,
    "predicted_parallax_deg": 8.0,
    "rgb_similarity": 0.77
  }
}
```

Pairs must contain two distinct registered images from the same group and
domain. Duplicate and reversed pairs fail. The scanner-assisted draft requires
overlap at least 50% and baseline inside 0.162–2.155 m.

## Audit while collection is incomplete

The audit mode accepts the deliberately unauthorized candidate draft. It
reports exact images, pairs, groups, counts by domain and group, missing quotas,
capture-envelope violations, and input hashes. It does not create an estimator
registry.

```bash
python benchmarks/two_view_pose_probability/prepare_prospective_registry.py \
  audit \
  --candidate /path/to/candidate-draft.json \
  --acquisition-plan /path/to/acquisition-plan.json \
  --images /path/to/images.jsonl \
  --pairs /path/to/pairs.jsonl \
  --output-dir /path/to/new-registry-audit
```

## Freeze after explicit authorization

`freeze` requires both:

- candidate authorization `GO_FOR_PROSPECTIVE_CONFIRMATION`;
- acquisition-plan status `AUTHORIZED_FOR_COLLECTION` bound to that exact
  candidate hash.

It also requires every planned group, panorama, and candidate-pair quota. If a
condition fails, it creates no partial output directory. On success it writes
`prospective-registry.jsonl` using the seal executor's registry schema and
binds each pair to image, validity-mask, candidate, and acquisition-plan
hashes.

```bash
python benchmarks/two_view_pose_probability/prepare_prospective_registry.py \
  freeze \
  --candidate /path/to/authorized-candidate.json \
  --acquisition-plan /path/to/authorized-acquisition-plan.json \
  --images /path/to/images.jsonl \
  --pairs /path/to/pairs.jsonl \
  --output-dir /path/to/new-frozen-registry
```

This frozen registry becomes the input to prediction, deterministic within-
group primary selection, and finally `run_prospective_confirmation.py seal`.
