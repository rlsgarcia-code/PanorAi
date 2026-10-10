# Multi-scene metric landmark BA

This development benchmark prospectively replays one high-overlap P74 pair
from each of the W, G, and M families. Pair eligibility comes from VAL-013;
the three targets already have frozen no-resize VAL-020 monocular priors.

The runner compares four values only at geometrically verified matched rays:
the monocular prior, initial metric triangulation, joint second-camera plus
landmark BA, and landmark-only BA with registered poses fixed. Registered
organized XYZ is opened only after prediction arrays are saved and hashed.

```bash
python benchmarks/metric_landmark_ba/run_p74_multiscene.py \
  --p74-root /path/to/P74/eq \
  --prior-root /private/tmp/panorai-val020-native-noresize \
  --output /private/tmp/panorai-val048-multiscene-landmark-ba
```

The benchmark does not densify a panorama. Its metric and scale-invariant
scores apply only to retained sparse landmarks and must not be described as
full-image depth accuracy.
