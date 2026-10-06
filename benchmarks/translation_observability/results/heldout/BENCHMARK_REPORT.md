# Real calibrated-pair relative-pose benchmark

| Variant | Returned | Accepted | Strict | High precision | R median / P95 (deg) | t median / P95 (deg) | Axis median / P95 (deg) | Runtime median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| msac-first-refit | 10/12 | 8/12 | 8/12 | 4/12 | 0.860 / 25.000 | 5.425 / 142.098 | 5.425 / 62.467 | 2866.6 |
| fixed-rotation-translation-gated | 10/12 | 7/12 | 8/12 | 3/12 | 0.860 / 25.000 | 5.981 / 143.763 | 5.981 / 63.828 | 2336.0 |

Strict means `R < 5 deg` and oriented `t < 10 deg`. High precision means
`R < 1 deg` and oriented `t < 5 deg`. Axis error ignores the sign of `t`.

## Evidence separation

The estimate command had no ground-truth argument and froze its output
read-only. The evaluate command opened that prediction first and only then
loaded the LiDAR-derived cam0 reference trajectory.

## Reproducibility

- Prediction SHA-256: `2e68b53fe8f0946b860f3c310f41d13fdca976a41afa4585764b26278f4ee99c`
- Ground-truth SHA-256: `33ea895a4e1bf9434febfc617a5a4baa66971a38ea4ee52b6c8bf7d91c417f0f`
- Raw CSV SHA-256: `4a9c167586d64874556e42da61381599f2a99a93d8c0286c8f5c99965e39888c`
- Source commit: `2503319d5024dfad6d2b9da441700dbea1a5092b` (dirty: `True`)
- Pairs: `12`

## Scope

This is one real sequence from one calibrated central fisheye lens. Pairs
within a window are temporally correlated. The result is descriptive and
does not justify a default-policy change by itself.
