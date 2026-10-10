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

### DA3 prior and unclipped correction follow-up

The historical result above used the VAL-020 ConvNeXt-Tiny prior and bounded
the harmonic log correction to `[-log(8), +log(8)]`. On G046, 15,769,758
pixels reached the positive correction boundary. Because 13,175,219 output
pixels also inherited the Tiny prior's `0.3 m` floor, the composition created
an artificial shell at exactly `0.3 * 8 = 2.4 m`. That visualization is not
valid evidence about harmonic densification.

VAL-051 repeats G046 and M014 with the external Apache-2.0
`DA3METRIC-LARGE` checkpoint. The source ERP is not resized. G uses forty-two
`812x1400` tangent views over a `4128x8256` ERP. M uses forty-two
`1008x1764` views over a `5184x10368` ERP so the W/G learned angular support
is retained at M's denser native lattice. The model/projection validity mask
is an explicit input. The harmonic field and compact anchor residual have no
hard amplitude clip; non-finite output fails instead of being silently
clamped.

```bash
python benchmarks/metric_landmark_ba/run_anchor_densification.py \
  --val048-root /private/tmp/panorai-val048-multiscene-landmark-ba \
  --val020-root /private/tmp/panorai-val020-native-noresize \
  --prior-root /private/tmp/panorai-val051-da3-priors \
  --prior-suffix=-da3metric-large-radial-m.npy \
  --prior-validity-suffix=-da3metric-large-validity.npy \
  --output /private/tmp/panorai-val051-da3-unclipped-densification-final \
  --families G M
```

| target | metric | DA3 prior | harmonic + exact anchors |
| --- | --- | ---: | ---: |
| G046 | AbsRel | 0.25501 | **0.21552** |
| G046 | delta-1 | 0.41761 | **0.63158** |
| G046 | scale-aligned relative 3D RMSE | **0.31063** | 0.32288 |
| G046 | mean normal error | **41.45 deg** | 43.49 deg |
| M014 | AbsRel | 0.24484 | **0.12522** |
| M014 | delta-1 | 0.49204 | **0.89573** |
| M014 | scale-aligned relative 3D RMSE | 0.25476 | **0.19454** |
| M014 | mean normal error | **29.37 deg** | 30.53 deg |

The corrected output contains zero exact `2.4 m` pixels on both targets,
versus 13,175,219 on historical G046. The degree-two correction stays within
`[-0.262, 0.700]` log-scale on G and `[-0.639, 0.355]` on M
without clipping. It preserves 42/42 G anchors and 201/201 M anchors within
`4.5e-7 m`. The result supports improved metric depth on both targets and
improved scale-invariant structure on M, but not a general local-surface gain:
normal error worsens on both targets and scale-invariant structure worsens on
G. Arrays, PLYs, viewers, P74 data, upstream source and checkpoint remain
external development artifacts.
