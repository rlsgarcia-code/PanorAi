# RGB-match overlap proxy and post-pose policy

This benchmark materializes the two probability layers used by
`panorai.experimental.two_view_probability`.

The overlap proxy is supervised by `registered_cloud_overlap_min`, measured
offline after registering each dataset's depth clouds. The model never receives
depth or a point cloud at runtime: its features are produced by the optimized
two-image spherical frontend before `R,t` estimation. Development, calibration,
and evaluation splits must be disjoint by `independence_component_id`; the fit
script fails if this contract is violated.

```bash
python benchmarks/two_view_overlap_proxy/fit_overlap_proxy.py \
  --features /path/to/aligned-table/features.jsonl \
  --output panorai/experimental/two_view_probability/data/rgb-overlap-proxy-v1.json \
  --frozen-bundle /path/to/panorai-3.5.0-authorized-v1.json \
  --bundle-output panorai/experimental/two_view_probability/data/panorai-3.5.0-authorized-v1.json

python benchmarks/two_view_overlap_proxy/evaluate_post_policy.py \
  --outcomes /path/to/aligned-table/outcomes-evaluation.jsonl \
  --predictions /path/to/models/predictions.jsonl \
  --output benchmarks/two_view_overlap_proxy/results/post-policy-evaluation.json
```

The evaluator requires exactly one selected-model evaluation prediction for
every outcome key. Missing, duplicate, or unexpected pair predictions abort
the run instead of being counted as implicit abstentions.

The tracked artifacts were produced from 2,385 two-view pairs (4,770 image
observations) and 69 independent spatial components across three evaluation
domains. Public artifacts deliberately use anonymous domain identifiers. The
aligned feature table does not retain both source panorama IDs, so it cannot
establish the number of unique panoramas after image reuse; the model artifact
records that count as unavailable. The held-out overlap proxy metrics are MAE
0.139, Brier 0.099, precision 0.811, and recall 0.370 for the event
`overlap >= 0.5` at a 0.5 probability threshold.

For the deployed post rule (`public quality accepted` and
`p_precise_post >= 0.90`), retrospective held-out pair-level precision is
105/107 = 0.981 and recall is 105/141 = 0.745. Its pooled exact one-sided 95%
precision lower bound is 0.942. Domain-specific lower bounds do not all clear
0.90 because two domains have limited selected support. Consequently, neither
artifact is a release reliability claim; prospective, independent domain
confirmation remains required.
