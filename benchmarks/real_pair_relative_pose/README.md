# Real calibrated-pair relative-pose benchmark

This source-checkout benchmark evaluates PanorAi relative pose on real feature
correspondences from the calibrated `cam0` lens of the
Hilti-Trimble-Oxford `floor_2_2025-10-28_run_2` sequence. Original dataset
bytes remain ignored under `.datasets/` and are never copied into this
directory, a source commit, wheel, or sdist.

The workflow has two separate commands:

1. `estimate` opens only the ROS bag and camera calibration, extracts SIFT
   correspondences, estimates each pose, and atomically freezes a read-only
   prediction;
2. `evaluate` refuses writable predictions, validates the prediction hash and
   only then opens the LiDAR-derived `cam0 -> map` trajectory.

Hilti/Kalibr camera poses use the OpenCV optical basis (`+Y` down), while
`EquidistantFisheyeCamera` returns PanorAi bearings (`+Y` up). Evaluation
therefore applies the explicit orthogonal basis change
`P = diag(1, -1, 1)`: `R_P = P R_CV P` and `t_P = P t_CV`. An analytic point
transport test independently checks this composition.

## Evidence phases

The first descriptive run used offsets `6,20,34,48,62,76` seconds, four
frames per window at 5 Hz, and 18 adjacent pairs. It compared count-first,
count-first+refit, MSAC-first, MSAC-first+refit, and the guarded decoupled
method. That run selected `MSAC-first+refit` as the sole held-out candidate;
its results must not be used as held-out evidence.

`HELDOUT_PROTOCOL.md` freezes the disjoint second run before its prediction is
generated or its reference poses are opened. No candidate or threshold may be
changed after that point.

The dataset is CC BY-NC-SA 3.0. Tracked output contains only compact derived
numeric measurements, hashes, configuration, and conclusions; it contains no
image, descriptor, calibration, trajectory, or ROS message bytes.
