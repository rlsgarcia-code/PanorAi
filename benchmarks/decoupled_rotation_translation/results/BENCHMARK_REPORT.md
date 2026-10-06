# Decoupled rotation/translation benchmark

| Condition | Variant | Accepted | Strict | High precision | Accepted R med / P95 | Accepted t med / P95 | False accept | Runtime med (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sub-meter-mixed-depth | count-first | 100.0% | 100.0% | 100.0% | 0.014 / 0.098 | 0.245 / 1.793 | 0.0% | 896.3 |
| sub-meter-mixed-depth | msac-first | 100.0% | 100.0% | 100.0% | 0.012 / 0.029 | 0.197 / 1.793 | 0.0% | 771.5 |
| sub-meter-mixed-depth | decoupled-unguarded | 100.0% | 100.0% | 100.0% | 0.023 / 0.031 | 0.160 / 0.489 | 0.0% | 881.4 |
| sub-meter-mixed-depth | decoupled-guarded | 100.0% | 100.0% | 100.0% | 0.023 / 0.031 | 0.160 / 0.489 | 0.0% | 995.1 |
| unobservable | count-first | 100.0% | 0.0% | 0.0% | 0.090 / 0.206 | 57.655 / 124.297 | 100.0% | 1008.5 |
| unobservable | msac-first | 100.0% | 0.0% | 0.0% | 0.037 / 0.064 | 61.897 / 142.085 | 100.0% | 1085.8 |
| unobservable | decoupled-unguarded | 0.0% | 0.0% | 0.0% | nan / nan | nan / nan | 0.0% | 1087.8 |
| unobservable | decoupled-guarded | 0.0% | 0.0% | 0.0% | nan / nan | nan / nan | 0.0% | 1191.5 |
| weak-parallax-stress | count-first | 100.0% | 10.0% | 5.0% | 0.103 / 0.209 | 36.110 / 106.898 | 90.0% | 901.0 |
| weak-parallax-stress | msac-first | 100.0% | 0.0% | 0.0% | 0.042 / 0.091 | 61.935 / 136.884 | 100.0% | 763.6 |
| weak-parallax-stress | decoupled-unguarded | 100.0% | 100.0% | 95.0% | 0.015 / 0.028 | 0.954 / 5.919 | 0.0% | 860.1 |
| weak-parallax-stress | decoupled-guarded | 100.0% | 100.0% | 95.0% | 0.015 / 0.028 | 0.954 / 5.919 | 0.0% | 878.6 |
| weak-parallax-with-outliers | count-first | 100.0% | 0.0% | 0.0% | 0.137 / 0.278 | 62.723 / 117.654 | 100.0% | 993.8 |
| weak-parallax-with-outliers | msac-first | 100.0% | 0.0% | 0.0% | 0.044 / 0.084 | 53.505 / 135.716 | 100.0% | 1358.3 |
| weak-parallax-with-outliers | decoupled-unguarded | 100.0% | 90.0% | 90.0% | 0.016 / 0.030 | 0.809 / 71.467 | 10.0% | 992.8 |
| weak-parallax-with-outliers | decoupled-guarded | 85.0% | 85.0% | 85.0% | 0.016 / 0.030 | 0.794 / 2.580 | 0.0% | 970.8 |
| well-conditioned | count-first | 100.0% | 100.0% | 95.0% | 0.011 / 0.111 | 0.039 / 0.310 | 0.0% | 2184.8 |
| well-conditioned | msac-first | 100.0% | 100.0% | 100.0% | 0.012 / 0.025 | 0.031 / 0.119 | 0.0% | 1338.2 |
| well-conditioned | decoupled-unguarded | 100.0% | 100.0% | 95.0% | 0.011 / 0.111 | 0.039 / 0.310 | 0.0% | 988.9 |
| well-conditioned | decoupled-guarded | 100.0% | 100.0% | 95.0% | 0.011 / 0.111 | 0.039 / 0.310 | 0.0% | 1087.1 |

## Reproducibility

- Schema: `panorai-decoupled-rotation-translation-benchmark/v1`
- Commit: `fc00acf3bb8fd6b2307a109b754e7f3b685b63d0`
- Dirty source: `True`
- Seed base: `1400000`
- Cases per condition: `20`
- Raw CSV SHA-256: `8e0565029fc452cada3dbaf17712bd6e52831525e819e4824f63aa1bbde7430e`
- Peak process RSS: `89145344` bytes

## Limitations

Synthetic source-checkout evidence only. The permissive benchmark policy
isolates GEO-015 abstention; it is not the default public acceptance policy.
No result establishes performance on real feature correspondences.
