# Essential scoring and all-inlier refit ablation

The table compares identical deterministic five-point proposal streams.
Refit variants use an unweighted all-inlier SVD, calibrated Essential
projection and a 100-step stabilization cap before bounded nonlinear
pose refinement.

| Condition | Variant | Strict | R median / P95 (deg) | t median / P95 (deg) | Refit median / max | Runtime median (ms) | Gains / losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sub-meter-mixed-depth | count-first-no-refit | 100.0% | 0.015 / 0.039 | 0.297 / 1.291 | 0.0 / 0 | 1062.4 | 0 / 0 |
| sub-meter-mixed-depth | count-first-refit | 100.0% | 0.016 / 0.041 | 0.196 / 0.781 | 5.0 / 7 | 1069.1 | 0 / 0 |
| sub-meter-mixed-depth | msac-first-no-refit | 100.0% | 0.013 / 0.029 | 0.150 / 0.310 | 0.0 / 0 | 1000.8 | 0 / 0 |
| sub-meter-mixed-depth | msac-first-refit | 100.0% | 0.012 / 0.031 | 0.155 / 0.306 | 5.0 / 7 | 1349.0 | 0 / 0 |
| sub-meter-mixed-depth | scale-marginal-first-no-refit | 100.0% | 0.013 / 0.031 | 0.197 / 0.314 | 0.0 / 0 | 1095.0 | 0 / 0 |
| sub-meter-mixed-depth | scale-marginal-first-refit | 100.0% | 0.014 / 0.041 | 0.181 / 0.318 | 5.0 / 7 | 1196.5 | 0 / 0 |
| unobservable | count-first-no-refit | 0.0% | 0.086 / 0.186 | 42.387 / 117.840 | 0.0 / 0 | 1469.3 | 0 / 0 |
| unobservable | count-first-refit | 5.0% | 0.057 / 0.180 | 41.469 / 113.876 | 23.0 / 100 | 2142.4 | 1 / 0 |
| unobservable | msac-first-no-refit | 0.0% | 0.035 / 0.082 | 62.027 / 151.327 | 0.0 / 0 | 1435.3 | 0 / 0 |
| unobservable | msac-first-refit | 0.0% | 0.031 / 0.059 | 56.766 / 97.165 | 24.0 / 93 | 1991.0 | 0 / 0 |
| unobservable | scale-marginal-first-no-refit | 0.0% | 0.046 / 0.122 | 49.748 / 117.347 | 0.0 / 0 | 1435.8 | 0 / 0 |
| unobservable | scale-marginal-first-refit | 0.0% | 0.031 / 0.063 | 48.306 / 102.496 | 24.0 / 93 | 2699.7 | 0 / 0 |
| weak-parallax-stress | count-first-no-refit | 10.0% | 0.096 / 0.181 | 40.232 / 142.713 | 0.0 / 0 | 1554.4 | 0 / 0 |
| weak-parallax-stress | count-first-refit | 20.0% | 0.054 / 0.181 | 38.901 / 142.713 | 14.0 / 86 | 2095.1 | 2 / 0 |
| weak-parallax-stress | msac-first-no-refit | 5.0% | 0.027 / 0.063 | 55.987 / 130.360 | 0.0 / 0 | 1223.3 | 1 / 2 |
| weak-parallax-stress | msac-first-refit | 5.0% | 0.033 / 0.048 | 59.905 / 124.852 | 20.0 / 93 | 3036.8 | 1 / 2 |
| weak-parallax-stress | scale-marginal-first-no-refit | 10.0% | 0.048 / 0.177 | 59.891 / 127.316 | 0.0 / 0 | 1654.9 | 2 / 2 |
| weak-parallax-stress | scale-marginal-first-refit | 10.0% | 0.032 / 0.079 | 53.963 / 129.081 | 19.0 / 85 | 2973.1 | 2 / 2 |
| weak-parallax-with-outliers | count-first-no-refit | 5.0% | 0.092 / 0.185 | 54.603 / 151.702 | 0.0 / 0 | 1291.1 | 0 / 0 |
| weak-parallax-with-outliers | count-first-refit | 10.0% | 0.037 / 0.185 | 39.355 / 146.966 | 20.5 / 77 | 1917.2 | 2 / 1 |
| weak-parallax-with-outliers | msac-first-no-refit | 0.0% | 0.027 / 0.067 | 78.829 / 153.449 | 0.0 / 0 | 1172.8 | 0 / 1 |
| weak-parallax-with-outliers | msac-first-refit | 10.0% | 0.032 / 0.042 | 58.005 / 145.636 | 18.5 / 91 | 1076.5 | 2 / 1 |
| weak-parallax-with-outliers | scale-marginal-first-no-refit | 5.0% | 0.041 / 0.101 | 65.062 / 156.764 | 0.0 / 0 | 989.3 | 0 / 0 |
| weak-parallax-with-outliers | scale-marginal-first-refit | 10.0% | 0.036 / 0.101 | 43.094 / 148.394 | 19.5 / 82 | 1137.0 | 2 / 1 |
| well-conditioned | count-first-no-refit | 100.0% | 0.017 / 0.058 | 0.040 / 0.425 | 0.0 / 0 | 899.3 | 0 / 0 |
| well-conditioned | count-first-refit | 100.0% | 0.017 / 0.037 | 0.040 / 0.388 | 3.0 / 4 | 988.0 | 0 / 0 |
| well-conditioned | msac-first-no-refit | 100.0% | 0.016 / 0.031 | 0.041 / 0.086 | 0.0 / 0 | 923.5 | 0 / 0 |
| well-conditioned | msac-first-refit | 100.0% | 0.017 / 0.032 | 0.040 / 0.085 | 3.0 / 4 | 738.8 | 0 / 0 |
| well-conditioned | scale-marginal-first-no-refit | 100.0% | 0.016 / 0.032 | 0.040 / 0.085 | 0.0 / 0 | 836.7 | 0 / 0 |
| well-conditioned | scale-marginal-first-refit | 100.0% | 0.016 / 0.032 | 0.040 / 0.088 | 3.0 / 4 | 1198.1 | 0 / 0 |

## Reproducibility

- Schema: `panorai-essential-scoring-refit-ablation/v1`
- Commit: `fc00acf3bb8fd6b2307a109b754e7f3b685b63d0`
- Dirty source: `True`
- Seed base: `1100000`
- Cases per condition: `20`
- Python: `3.12.4`
- NumPy: `1.26.4`
- SciPy: `1.14.0`
- Platform: `macOS-26.6.2-arm64-arm-64bit`
- Peak process RSS: `92422144` bytes
- Raw CSV SHA-256: `ffe2611b26901f8a9572c93a02c4297ae7f163ae35a868a0da0c97cf0228a11e`

## Limitations

This is synthetic source-checkout evidence. The ranking and refit
controls preserve the default behavior but are Experimental. A lower
internal score is not a calibrated accuracy probability, and no result
here establishes performance on real feature correspondences.
