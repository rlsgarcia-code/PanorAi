# Draft capture standard for robust spherical relative pose

Status: **provisional development guidance**, not yet a release guarantee.

This document translates the VAL-017 synthetic operating envelope into an
actionable capture procedure. It applies to estimating rotation and the
**oriented direction** of translation between central spherical panoramas. A
two-view estimator cannot recover the metric magnitude of translation.

## 1. Required capture envelope

### 1.1 Scene intersection

- Plan for at least **70% bidirectional registered-cloud overlap** between
  adjacent capture stations.
- Treat **50% as a hard eligibility floor**, not the nominal target.
- Spatial overlap alone is insufficient: a pair may overlap strongly while
  having too little translation, too little texture, or correspondences
  concentrated on one structure.

The 70% target and 50% floor come from the P74 selection work. They must still
be replayed against the final frontend and the operating checks below before
being promoted to a release contract.

### 1.2 Translation parallax

- Target median inlier parallax: **at least 2 degrees**.
- Warning band: **1 to 2 degrees**. Keep the pair only when all other evidence
  is strong and a neighboring view confirms the motion.
- Reject or recapture below **1 degree**. Rotation may remain accurate while
  oriented translation becomes unstable.
- A camera rotation around one station is not a substitute for translation.

For predominantly lateral motion and modest angles,

`baseline / representative scene distance ~= tan(parallax)`.

| Representative scene distance | Baseline for 1 deg warning floor | Baseline for 2 deg target |
| ---: | ---: | ---: |
| 2 m | 3.5 cm | 7.0 cm |
| 5 m | 8.7 cm | 17.5 cm |
| 10 m | 17.5 cm | 34.9 cm |
| 20 m | 34.9 cm | 69.8 cm |
| 50 m | 87.3 cm | 1.75 m |

These distances are planning values. The release check must use the measured
median inlier parallax because forward motion, mixed depth and viewing
direction change the effective baseline.

### 1.3 Correspondence support

- Capture enough texture to obtain at least **160 geometrically verified
  candidate correspondences** per adjacent pair.
- Target at least **55% true geometric support**, corresponding to roughly
  **88 inliers out of 160**. Below 40%, abstention becomes common even with
  adequate parallax.
- Never use the raw keypoint count as the acceptance criterion. Record the
  robust inlier count, inlier ratio, residual and stability evidence.

The present synthetic grid found 100% strict accuracy and acceptance in 3/3
trials at 2 degrees with 55% or more inliers. Three trials locate a boundary;
they do not establish a production failure probability. The 160/55% target is
therefore deliberately more conservative than the estimator's public minimum.

### 1.4 Angular distribution

- Matches must span the panorama in both longitude and latitude.
- Operational target: at least **10 of the 20 equal-area quality cells** in
  each image and normalized coverage entropy of at least **0.70**.
- Add an explicit two-dimensional spread check before release: require support
  in at least three latitude bands and six longitude sectors, with no single
  band or compact cap dominating the consensus.
- Reject pairs whose matches lie primarily on one pipe, railing, horizon,
  corridor edge, or small angular cap, even when their count is high.

In the frozen sweep, 160 matches confined to a 15-degree cap produced zero
accepted poses and median translation errors near 56 degrees. A narrow
equatorial band could occasionally be accepted with an incorrect translation,
so the current scalar entropy check is necessary but not sufficient.

### 1.5 Image and motion quality

- The 2-degree parallax target remained strictly correct and accepted in 3/3
  trials for independent bearing noise from 0.02 through 0.50 degree.
- At 1 degree, acceptance degraded at 0.25 degree noise and disappeared at
  0.50 degree noise.
- Capture should therefore avoid blur, rolling motion during panorama
  formation, moving foreground objects, exposure discontinuities and weak
  calibration. These effects are correlated and were not simulated by the
  independent-noise model.

## 2. Pair acceptance record

Every accepted capture pair should store:

1. bidirectional spatial cloud overlap;
2. total valid correspondences and robust inlier count/ratio;
3. occupied equal-area cells and coverage entropy in both panoramas;
4. longitude- and latitude-spread diagnostics;
5. median and P90 angular residual;
6. median triangulation parallax;
7. cheirality ratio and translation-orientation margin;
8. rotation and translation stability under deterministic resampling;
9. Essential versus rotation-only and spherical-homography competition;
10. the final acceptance decision and every rejection reason.

Validity masks must remain explicit throughout. Black pixels are not an
invalidity oracle.

## 3. Capture decision

### Accept as nominal

- overlap >= 70%;
- median inlier parallax >= 2 degrees;
- at least 160 verified candidates and approximately 88 or more inliers;
- inlier ratio >= 55%;
- broad two-dimensional angular support;
- public quality policy accepts the pose;
- no neighboring-view or three-view consistency contradiction.

### Accept only as provisional

- overlap from 50% to 70%, or parallax from 1 to 2 degrees;
- all remaining evidence is nominal;
- the pair is confirmed by at least one neighboring view.

### Recapture

- overlap < 50%;
- parallax < 1 degree or rotation-only motion;
- fewer than 160 candidates or fewer than about 88 supported inliers;
- matches concentrated in a compact cap or one narrow scene band;
- the estimator returns no pose, rejects the pose, prefers rotation-only or a
  spherical homography, or reports unstable translation.

## 4. What remains before release

The thresholds above must be treated as a preregistered candidate standard and
validated without retuning on:

- a larger held-out synthetic confirmation set at the joint nominal boundary;
- frozen P74 pairs selected independently by cloud overlap;
- at least one sequential-video capture with known or externally measured
  trajectory;
- repeated texture, dynamic objects, exposure change, blur, calibration bias
  and non-central panorama failure sets;
- three-view cycle consistency.

A release claim should report the lower confidence bound for strict precision
among accepted poses, not only average error or return rate.
