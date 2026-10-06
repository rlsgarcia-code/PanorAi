# Real calibrated-pair relative-pose benchmark

| Variant | Returned | Accepted | Strict | High precision | R median / P95 (deg) | t median / P95 (deg) | Axis median / P95 (deg) | Runtime median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count-first | 12/12 | 4/12 | 4/12 | 2/12 | 1.014 / 10.172 | 12.933 / 65.903 | 12.933 / 65.903 | 4129.7 |
| msac-first-refit | 11/12 | 7/12 | 6/12 | 3/12 | 0.901 / 9.460 | 5.933 / 58.451 | 5.933 / 58.451 | 5025.0 |
| decoupled-guarded | 12/12 | 2/12 | 4/12 | 1/12 | 1.102 / 10.172 | 12.933 / 65.903 | 12.933 / 65.903 | 4873.3 |

Strict means `R < 5 deg` and oriented `t < 10 deg`. High precision means
`R < 1 deg` and oriented `t < 5 deg`. Axis error ignores the sign of `t`.

## Evidence separation

The estimate command had no ground-truth argument and froze its output
read-only. The evaluate command opened that prediction first and only then
loaded the LiDAR-derived cam0 reference trajectory.

## Reproducibility

- Prediction SHA-256: `4a6e44c027bd54192eeef4703cb220c5600c17a3b6724be617a6c80bd2d9d582`
- Ground-truth SHA-256: `33ea895a4e1bf9434febfc617a5a4baa66971a38ea4ee52b6c8bf7d91c417f0f`
- Raw CSV SHA-256: `c9572b2efeb85fbaace15d0615de605e45b7480af580a0cf01c54bfae795745f`
- Source commit: `7cea26402662c83faaa73105d7dab2b565435291` (dirty: `True`)
- Pairs: `12`

## Scope

This is one real sequence from one calibrated central fisheye lens. Pairs
within a window are temporally correlated. The result is descriptive and
does not justify a default-policy change by itself.
