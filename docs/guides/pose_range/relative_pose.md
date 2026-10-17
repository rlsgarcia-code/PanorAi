# Estimate relative pose

**Level:** Start · **Input:** two ERP images · **Output:** Essential matrix,
rotation and translation direction · **Stability:** Experimental application

```python
from panorai.applications import SphericalRelativePose

application = SphericalRelativePose.reference()
pose = application(reference, source)
if pose is None or not pose.quality_report.accepted:
    raise RuntimeError("relative pose was not trustworthy")

E = pose.essential_matrix
R_b_from_a = pose.R
t_direction_b_from_a = pose.t
```

The translation has direction but no metric magnitude. Inspect inliers,
angular coverage, residual percentiles, parallax, cheirality, stability and
rejection reasons before continuing.

`reference()` names the complete evidence-backed computation, not only the
image frontend. It combines `SphericalFeaturePipeline.for_relative_pose()`,
the spatially weighted five-point sampler, MSAC-first hypothesis ranking,
bounded all-inlier Essential refit, joint pose refinement, parallax-weighted
translation orientation and the conservative acceptance policy. The plain
`SphericalRelativePose()` constructor is equivalent.

The `v1` payload is fully enumerated rather than inherited from mutable class
defaults. Its canonical regression includes the complete frontend, estimator,
sampler and acceptance configuration. A named custom `RelativePoseProfile`
keeps its `profile_id` and `evidence_scope` through application JSON
persistence; changing a computational parameter with `with_params()` clears
that identity instead of attributing new behavior to the old profile.
PanorAi-reserved profile IDs also fail closed when loaded with any noncanonical
payload. Custom thresholds or components require a custom `profile_id` and an
honest `evidence_scope`.

Use
`application.as_pipeline()` to inspect or replace the explicit
`extract_features`, `match_features`, `bearing_correspondences`, and
`estimate_pose` stages. If matches already exist, call
`application.estimate_matches(matches)`. At the lowest level, use
`SphericalRelativePoseEstimator` directly with bearing correspondences.

## Capture eligibility is a separate decision

The estimator cannot infer physical overlap, baseline or scene distance from
RGB values. Supply those facts explicitly when a supported capture envelope is
available:

```python
from panorai.applications import (
    RelativePoseCaptureMetadata,
    RelativePoseCapturePolicy,
    SphericalRelativePose,
)

capture_policy = RelativePoseCapturePolicy.scanner_assisted_reference()
application = SphericalRelativePose.reference(capture_policy=capture_policy)
pose = application(
    reference,
    source,
    capture_metadata=RelativePoseCaptureMetadata(
        registered_cloud_overlap_min=0.74,
        baseline_m=0.8,
        representative_scene_distance_m=3.2,
    ),
)
```

This policy uses 50% minimum and 70% preferred bidirectional registered-cloud
overlap. Registered overlap requires independently registered depth support;
it is not estimated from image similarity. Fifty percent is an eligibility
boundary supported by public indoor and separately governed industrial
observations, not a universal precision guarantee. No universal baseline
interval in metres is embedded in the reference policy.

The scanner and Hilti IDs are reserved for their exact documented payloads.
To define a different operating envelope, construct a policy with a distinct
organization-owned `profile_id` and describe its evidence scope; reusing a
PanorAi ID with changed limits raises an error.

An enforcing capture Stage is prepended only when `capture_policy` is
configured:

```text
assess_capture → extract_features → match_features
               → bearing_correspondences → estimate_pose
```

Missing required metadata produces `insufficient-metadata`; a measured limit
violation produces `ineligible`. `capture_mode="enforce"` stops before feature
extraction, while `capture_mode="advisory"` records the assessment and allows
diagnostic execution. Capture eligibility never replaces
`pose.quality_report.accepted`.

`RelativePoseCapturePolicy.hilti_fisheye_development()` records the narrower
development envelope measured on one Hilti sequence: at least 5 Hz, target
steps no larger than roughly 0.4 m / 15 degrees and maximum steps of 0.6 m /
30 degrees. Its name and evidence scope deliberately prevent presenting that
single-sequence observation as a cross-domain law.

The retrospective probability rule that combined overlap, a Matterport-only
metric-baseline interval and post-estimation probabilities failed its untouched
multidataset gate. It is therefore not exposed as a reliable release preset.

The capture study is not the single Hilti sequence used for temporal guidance.
Its retrospective evidence base contains 4,017 unique images, 2,385 two-view
pairs and 69 independent components: 1,890 Matterport360 pairs, 450
Stanford2D3D pairs and 45 separately governed industrial pairs. A separate
423-set/66-group multiview census and a 7,058-candidate industrial eligibility
census support the operational interpretation. These broad studies justify
exposing the capture variables and eligibility boundary; their untouched-split
failures are why the API does not call the resulting decision a universal
calibrated confidence.

## Geometry and convention

For paired unit bearings $b_a$ and $b_b$, a valid central relative pose obeys

$$b_b^T [t]_{\times} R b_a = 0.$$

The returned transform maps camera-A coordinates into camera-B coordinates:
`X_b = R_b_from_a @ X_a + t_b_from_a`. The estimator normalizes `pose.t`, so
its magnitude is arbitrary. This convention must remain unchanged when the
pose is passed to triangulation, stereo or region transfer.

## Acceptance is part of the API

`pose.quality_report.accepted` is not decorative metadata. A high raw inlier
count can still be rejected for poor angular coverage, low parallax, unstable
translation, weak cheirality or a competitive rotation-only explanation.
Store the report beside `R` and `t`; otherwise downstream failures cannot be
distinguished from a bad pose.

Capture and pose acceptance answer different questions:

- `RelativePoseCapturePolicy` asks whether the acquired pair lies inside a
  measured operating envelope before computation;
- `RelativePoseAcceptancePolicy` asks whether the returned geometric evidence
  is trustworthy after computation.

## Common failure modes

- Repeated industrial structures create descriptor-consistent outliers.
- Pure rotation constrains $R$ but not a useful translation direction.
- Far structure produces small parallax and ill-conditioned triangulation.
- Bearings expressed in different camera-axis conventions invalidate the
  Essential relation even if ERP pixels appear aligned.

The Essential matrix, minimal solver, consensus and refinement are derived in
[spherical two-view geometry](../tutorials/04_two_view_geometry.md). Next:
[recover scale and triangulate](metric_scale_and_triangulation.md).
