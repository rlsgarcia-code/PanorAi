# Real calibrated-pair relative-pose benchmark

| Variant | Returned | Accepted | Strict | High precision | R median / P95 (deg) | t median / P95 (deg) | Axis median / P95 (deg) | Runtime median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| count-first | 18/18 | 13/18 | 13/18 | 6/18 | 0.658 / 1.596 | 6.852 / 82.861 | 6.852 / 82.861 | 16748.6 |
| count-first-refit | 18/18 | 13/18 | 13/18 | 6/18 | 0.687 / 1.596 | 7.975 / 82.861 | 7.975 / 82.861 | 6279.7 |
| msac-first | 18/18 | 16/18 | 13/18 | 7/18 | 0.447 / 1.773 | 5.802 / 20.554 | 5.802 / 20.554 | 7888.8 |
| msac-first-refit | 18/18 | 15/18 | 14/18 | 9/18 | 0.351 / 1.773 | 5.110 / 20.554 | 5.110 / 20.554 | 4529.2 |
| decoupled-guarded | 18/18 | 5/18 | 12/18 | 5/18 | 0.756 / 1.887 | 7.377 / 82.861 | 7.377 / 82.861 | 21531.2 |

Strict means `R < 5 deg` and oriented `t < 10 deg`. High precision means
`R < 1 deg` and oriented `t < 5 deg`. Axis error ignores the sign of `t`.

## Evidence separation

The estimate command had no ground-truth argument and froze its output
read-only. The evaluate command opened that prediction first and only then
loaded the LiDAR-derived cam0 reference trajectory.

## Reproducibility

- Prediction SHA-256: `852e19e63cb07938f1be6890efa810b1e5449ce7d31c74d40f7293f71e6484d2`
- Ground-truth SHA-256: `33ea895a4e1bf9434febfc617a5a4baa66971a38ea4ee52b6c8bf7d91c417f0f`
- Raw CSV SHA-256: `68fdfaa2f6d776c7679774efe2b2e7d1775bf41c4880f001e4ff2662edb25f88`
- Source commit: `7cea26402662c83faaa73105d7dab2b565435291` (dirty: `True`)
- Pairs: `18`

## Scope

This is one real sequence from one calibrated central fisheye lens. Pairs
within a window are temporally correlated. The result is descriptive and
does not justify a default-policy change by itself.
