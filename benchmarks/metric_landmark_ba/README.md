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

## Anchor-preserving densification diagnostic

`run_anchor_densification.py` consumes the frozen G and M prediction files
from the benchmark above. It never resizes the native ERP prior. It compares:

- one global robust scale;
- a compact Wendland-C2 interpolation around the sparse landmarks;
- a degree-0/1/2 spherical-harmonic log-range correction whose degree is
  selected only by leave-one-landmark-out error; and
- the harmonic field followed by a compact residual that overwrites every
  unique landmark cell exactly.

The dense arrays and their hashes are written before registered depth is
opened. The final composition therefore uses sparse geometry alone: the
harmonic term repairs low-frequency scale drift, while the compact residual
keeps the optimized landmark values immutable. Pixels outside the compact
residual support remain byte-identical to the harmonic field. Longitude wraps
at the ERP seam and latitude does not.

```bash
python benchmarks/metric_landmark_ba/run_anchor_densification.py \
  --val048-root /private/tmp/panorai-val048-multiscene-landmark-ba \
  --val020-root /private/tmp/panorai-val020-native-noresize \
  --output /private/tmp/panorai-val049g-anchor-densification \
  --families G M
```

On the frozen P74 G and M targets, the exact-anchor harmonic composition
improved full-map AbsRel from `0.78118` to `0.72532` and from `0.55082` to
`0.46540`. Scale-invariant log RMSE improved from `0.81365` to `0.68674` and
from `0.46718` to `0.45719`. Delta-1 increased from `0.01279` to `0.28169`
and from `0.03895` to `0.35344`. Mean normal error changed from `52.54` to
`54.26` degrees on G and from `62.04` to `62.28` degrees on M. Consequently,
this is evidence for better dense radial structure with exact sparse
preservation, not for better surface normals. The compact-only interpolation
is retained as a negative control: it preserves landmarks but does not
materially repair the full image.
