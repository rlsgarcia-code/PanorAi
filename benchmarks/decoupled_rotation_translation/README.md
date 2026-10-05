# Decoupled rotation/translation benchmark

This source-checkout experiment compares the compatibility baseline and the
GEO-014 MSAC rotation candidate with the GEO-015 two-stage estimator.  The
two-stage path first estimates a rotation-only Wahba consensus, then estimates
translation from correspondences not explained by that rotation.

Two decoupled variants are reported:

- `decoupled-unguarded` bypasses the margin gate when translation is
  observable;
- `decoupled-guarded` requires a 0.15 relative score margin between distinct
  translation-direction clusters and otherwise abstains.

The benchmark uses a deliberately permissive acceptance policy so that the
guard's coverage and false-accept behavior are measured directly.  The normal
public quality policy remains unchanged.

Development used seeds below `1300000`.  The primary evidence must use a new
seed base and must not be inspected until the algorithm and margin are frozen.

```console
python benchmarks/decoupled_rotation_translation/run_benchmark.py \
  --output benchmarks/decoupled_rotation_translation/results \
  --cases 20 \
  --seed-base 1400000
```

This is synthetic source-checkout evidence, not an installed-wheel or
real-feature accuracy claim.
