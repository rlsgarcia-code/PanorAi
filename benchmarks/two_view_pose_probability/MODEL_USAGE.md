# Using the spherical two-view probability models

Status: **ready for research use and prospective confirmation**. These models
are not yet a release-reliability guarantee and are intentionally not part of
the stable `panorai` wheel API.

## Frozen identity

| Item | Value |
| --- | --- |
| Model bundle | `model_artifacts/panorai-3.5.0-authorized-v1.json` |
| Bundle SHA-256 | `4c481d6b84e9507c8656b1b57fb24359b3db30ff639839df5c510abcae9a7f62` |
| PanorAi version | `3.5.0` |
| PanorAi source commit | `03c5b36b28225b24d3909286bf53250d7b532aa3` |
| Authorization | `AUTH-20261009-PPOST090-BALANCED60` |
| Capture snapshot SHA-256 | `9ffedc15f37d9af82dd4882b61dd7598050946bdeae872885eed3d773a8f288e` |
| Post snapshot SHA-256 | `9999857efb9fa404bc51f5fb31bc1e50760f1899c6c32bc334e278c8add55051` |

The JSON contains numerical coefficients, scalers, feature transforms,
calibration parameters, thresholds, sample counts, and provenance. It contains
no source image, depth map, point cloud, or reference pose. It is a tracked
source-tree research artifact; it is not currently included in the wheel or
sdist. Public redistribution claims still require the repository's normal
license/provenance review.

## What the scorer returns

The capture side evaluates:

```text
p_accept = P(public quality policy accepts R,t | explicit overlap, baseline)
p_precise_given_accept = P(precise | accepted, explicit overlap, baseline)
p_usable = p_accept * p_precise_given_accept
```

The post-processing side evaluates:

```text
p_precise_post = P(precise | returned pose diagnostics)
```

`precise` means rotation error at most 1 degree and oriented
translation-direction error at most 5 degrees. Translation magnitude is not
predicted because central two-view geometry observes translation only up to
scale.

## Python API

Run from the repository root:

```python
from benchmarks.two_view_pose_probability.probability_inference import (
    TwoViewProbabilityModels,
)

models = TwoViewProbabilityModels.load_default()

capture = models.score_capture(
    registered_cloud_overlap_min=0.65,  # fraction, not percent
    baseline_m=0.80,                    # metres
)

print(capture.p_accept)
print(capture.p_precise_given_accept)
print(capture.p_usable)
print(capture.inside_supported_envelope)
```

To score a complete result produced by the frozen optimized-pair route:

```python
import json
from pathlib import Path

result = json.loads(Path("pair-result.json").read_text())

decision = models.decide_from_pair_result(
    registered_cloud_overlap_min=0.65,
    baseline_m=0.80,
    pair_result=result,
)

print(decision.p_precise_post)
print(decision.selected)
print(decision.reason)
```

By default this method requires the pair result to declare PanorAi 3.5.0 and
the exact source commit above. This prevents silently applying the model to a
different estimator route. A caller with its own validated adapter may pass
the 14 post-processing fields directly to `score_post()` or `decide()`.

## Command line

Capture geometry only:

```bash
python benchmarks/two_view_pose_probability/probability_inference.py \
  capture \
  --overlap 0.65 \
  --baseline-m 0.80
```

Complete pair result:

```bash
python benchmarks/two_view_pose_probability/probability_inference.py \
  pair \
  --overlap 0.65 \
  --baseline-m 0.80 \
  --result /path/to/pair-result.json
```

The program writes one JSON object to standard output. Invalid units,
non-finite values, missing diagnostics, a changed bundle hash, and a mismatched
PanorAi identity fail explicitly.

## Input contract

### Explicit capture geometry

| Field | Unit/range | Meaning |
| --- | --- | --- |
| `registered_cloud_overlap_min` | fraction `[0, 1]` | minimum directed overlap of the two valid registered clouds |
| `baseline_m` | metres, non-negative | Euclidean distance between capture centres |

The supported operating envelope is:

```text
registered_cloud_overlap_min >= 0.50
0.16196559975867944 <= baseline_m <= 2.1546692039871624
```

The scorer returns probabilities outside this envelope for analysis, but marks
`inside_supported_envelope=false`; the complete decision always abstains.
Overlap must come from valid clouds and known transforms in a common coordinate
system. Do not pass `65` for 65%; pass `0.65`. Do not infer validity from black
pixels or zero depth.

### Post-processing diagnostics

The direct `score_post()` API accepts a mapping with these exact fields:

| Field | Unit/range |
| --- | --- |
| `raw_quality_score` | fraction `[0, 1]` |
| `match_count` | non-negative count |
| `inlier_count` | non-negative count |
| `inlier_ratio` | fraction `[0, 1]` |
| `median_parallax_deg` | non-negative degrees |
| `cheirality_ratio` | fraction `[0, 1]` |
| `coverage_entropy_a`, `coverage_entropy_b` | finite normalized entropy |
| `stability_translation_p90_deg` | non-negative degrees |
| `essential_score_margin` | finite score margin |
| `translation_orientation_cheirality_margin` | finite margin |
| `translation_orientation_weighted_margin` | finite margin |
| `translation_orientation_median_triangulation_angle_deg` | non-negative degrees |
| `translation_orientation_ambiguous` | boolean |

`extract_post_features()` maps a
`panorai-unified-optimized-pair/v2` result to this flat contract using exactly
the same field mapping as the aligned population replay. Required missing
values are errors; the scorer does not impute a convenient zero.

## Selection policy

A pair is selected only when all conditions hold:

1. explicit overlap and baseline are inside the supported envelope;
2. the estimator returned a finite R and unit translation direction;
3. the unchanged public PanorAi quality policy accepted the pose;
4. `p_usable >= 0.50`;
5. `p_precise_post >= 0.90`.

Otherwise `PairDecision.reason` identifies the first failed boundary:

- `outside-supported-capture-envelope`;
- `pose-not-returned`;
- `public-quality-rejected`;
- `capture-probability-below-threshold`;
- `post-probability-below-threshold`.

## Numerical conformance

The portable standard-library scorer was compared with every frozen evaluation
prediction generated by the NumPy study implementation:

| Model | Predictions | Maximum absolute difference |
| --- | ---: | ---: |
| Capture acceptance | 435 | `4.44e-16` |
| Capture precision given acceptance | 435 | `3.33e-16` |
| Post precision | 224 | `2.22e-16` |

The tracked regression fixture also verifies a real PanorAi 3.5.0 result:

```text
overlap = 0.8724329548200048
baseline = 0.7238950273624603 m
p_accept = 0.9975318441265014
p_precise_given_accept = 0.9702999686868229
p_usable = 0.9679051171200530
p_precise_post = 0.9922201383491895
decision = selected
```

## Interpretation boundary

The bundle is frozen retrospective evidence authorized for a prospective
trial. It is useful now for research operation, data collection, audit, and
selective abstention, but it must continue to report its status string. Do not
rename `selected` to “guaranteed correct”. The current release verdict remains
`NO_GO` until the prospective E8 protocol passes.
