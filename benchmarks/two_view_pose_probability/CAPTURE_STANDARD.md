# Provisional capture standard for spherical two-view R,t

Status: operational draft frozen before the aligned-population aggregate. This
standard guides acquisition; it is not a PanorAi release certification. Final
probability values and any selective threshold are populated only from the
complete locked PanorAi 3.5.0 analysis.

## Scope

This document applies to relative rotation and translation-direction
estimation from exactly two calibrated central spherical panoramas. It does
not cover translation magnitude, dense stereo, bundle adjustment, multiview
estimation, or rolling changes to a video trajectory.

The workflow keeps three decisions separate:

1. **capture eligibility** — are the two observations plausibly informative?;
2. **estimator acceptance** — did the unchanged PanorAi quality policy accept
   a returned R,t?;
3. **selective release** — is an accepted pose inside both calibrated capture
   support and calibrated post-processing confidence?

Passing one decision never substitutes for another.

## Operator-facing variables

`x_capture` contains only quantities that the operator or capture system can
know before the RGB relative-pose result. The archived datasets may reconstruct
an operational quantity from reference station geometry for retrospective
validation; deployment must measure or control that quantity independently.

| Variable | Availability to capture | Required interpretation |
|---|---|---|
| Number of panoramas | controlled | exactly two; it is fixed, not a model predictor |
| Sensor and panorama resolution | controlled | record the calibrated profile and ERP dimensions |
| Physical baseline | controlled or tracked | planned/measured station displacement; never recovered from the reference pose at deployment |
| Representative scene distance | depth-visible | robust radial distance summary when depth is available |
| Baseline/depth and predicted parallax | derived-visible | derive only from planned baseline and capture-visible depth |
| Minimum registered-cloud overlap | capture-system | valid only if both point clouds and transforms into one frame already exist |
| Valid-support masks | directly visible | preserve explicit masks; never infer validity from black pixels |
| Blur, clipping, exposure discontinuity | directly visible | compute independently per panorama before matching |
| Spherical texture occupancy | directly visible | compute independently per panorama; broad angular support is preferable |
| Dynamic or occluded fraction | visible when a detector exists | record method and uncertainty; do not silently impute |

Dataset/site identity, ground-truth R,t, reference epipolar residuals, true
match correctness, and final pose error are prohibited capture predictors.
Matches, inliers, cheirality, stability, and translation-orientation margins
belong to `x_post`, after processing.

## Capture profiles

### Profile D — registered depth or scanner assisted

Use this profile only when the clouds are already registered in a shared frame
independently of the two-view RGB pose being evaluated.

- Require at least **50% minimum bidirectional cloud overlap** as a provisional
  eligibility boundary.
- Prefer **70% or more** when the intended baseline and scene class lie inside
  independently supported cells.
- Treat 50% and 70% as eligibility guidance, not guarantees. Repetitive scenes,
  weak depth variation, motion, or translation ambiguity can fail even at high
  overlap.
- Record both directional support fractions, the symmetric summary, surface
  tolerance, sampling density, cloud provenance, and transforms used.
- Do not use a metre-only baseline interval outside its calibrated scene scale.
  Record representative distance and predicted parallax whenever possible.

### Profile RGB — panoramas without registered depth

The cloud-overlap percentages are unavailable. They must not be advertised as
image-visible quantities.

- Record planned baseline, sensor, masks, blur, clipping, exposure continuity,
  and independent spherical texture occupancy.
- Use overlap only after an RGB proxy has been independently validated against
  spatial cloud overlap on held-out groups.
- Until such a proxy and an RGB-only capture model pass group-held-out and
  prospective validation, the current 50% rule cannot certify RGB-only pairs.

## Acquisition procedure

1. Assign immutable view IDs and an independent acquisition-group ID before
   opening reference geometry.
2. Calibrate the spherical camera model and record sensor, firmware,
   resolution, timestamp, and exposure metadata.
3. Choose and record the station baseline. Avoid effectively coincident
   stations, but do not maximize baseline at the cost of shared support.
4. Capture explicit validity masks with each panorama.
5. Inspect for motion, severe blur, saturation, stitching discontinuities,
   and evidence concentrated in one narrow angular region.
6. For Profile D, register both clouds using the external capture system and
   compute the frozen bidirectional-overlap measure before RGB R,t estimation.
7. Reject or reacquire an ineligible pair. Do not run it merely to obtain a
   confidence score that compensates for unsupported capture conditions.
8. Process the eligible pair with the exact frozen wheel and configuration.
9. Save every frontend, matcher, estimator, orientation, timing, and memory
   diagnostic before opening reference R,t.
10. Apply the frozen capture and post-processing probability gates. Preserve
    failed pairs in all coverage denominators.

## Reacquisition triggers

Reacquire or add a new two-view pair when any of the following occurs:

- registered-cloud overlap is below 50% in Profile D;
- the capture lies outside a supported overlap × baseline/scene-scale cell;
- either mask is missing or validity provenance is ambiguous;
- severe blur, clipping, stitching discontinuity, motion, or occlusion is
  visible;
- texture or candidate features occupy only a narrow spherical region;
- the estimator returns no pose or rejects it;
- translation orientation is ambiguous, its cheirality margins are weak, the
  triangulation angle is unsupported, or translation stability is poor;
- the pair belongs to a sensor/domain/frontend combination outside calibration.

Reacquisition should change viewpoint or evidence distribution, not merely
repeat an indistinguishable frame. Multiple pairs may be captured, but every
claim in this study remains a separate two-image estimate.

## Required capture record

For each unordered pair, retain:

- pair, view, station, and independent-group IDs;
- timestamps, sensor profile, ERP dimensions, masks, and mask provenance;
- planned/measured baseline and its measurement method;
- depth summary, predicted parallax, and registered-cloud overlap when valid;
- blur, clipping, exposure, dynamic-area, valid-support, and spherical texture
  summaries for each panorama;
- exact wheel name, PanorAi version, source commit, wheel SHA-256, Python,
  OpenCV, CPU, thread settings, and serialized configuration;
- immutable capture-model probability, post-model probability, acceptance,
  and selection decision written before reference poses are opened;
- exclusion or failure reason for every missing or unsuccessful result.

## Current evidence boundary

Matterport360 provides 63 independent components and supports primary
retrospective inference. Stanford2D3D and P74 each provide only three
independent groups and remain transfer/failure-discovery evidence. The aligned
population analysis may authorize prospective confirmation, but only E8 with
new independent groups can support a release-reliability claim.

The frozen E8 gate requires at least 40 new independent groups, 120 selected
pairs with at most three per group, observed selected-pose precision of at
least 95%, a one-sided exact 95% lower bound of at least 90%, and zero
catastrophic accepted poses. Predictions and selection decisions must be
sealed before reference R,t is opened.
