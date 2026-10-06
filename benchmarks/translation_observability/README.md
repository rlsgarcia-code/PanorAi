# Fixed-rotation translation observability study

GEO-017 tested a robust unit-translation refit after the GEO-016
MSAC-first/all-inlier Essential estimate. The experiment held the selected
rotation fixed, fitted `t` from epipolar-plane normals with IRLS, and measured
translation stability through deterministic subset re-estimation that allowed
rotation/translation coupling.

The estimator processed only the Hilti cam0 bag and calibration. Its prediction
was frozen read-only before a separate evaluator opened the LiDAR-derived
trajectory. Original dataset bytes remain ignored under `.datasets/`; this
directory contains only first-party protocol text and compact derived numeric
results.

The frozen held-out result was negative. See `CONCLUSIONS.md`. The proposed
public implementation was not retained. The established MSAC+all-inlier refit
is the default joint estimator; the overall relative-pose API remains
Experimental.
