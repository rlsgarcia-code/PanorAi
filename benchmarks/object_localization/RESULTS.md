# Stanford2D3D office practical result

Status: **PASS** (development evidence, not release qualification)

The frozen `area_2/office_3` pair contains two separate desk instances and one
bookcase visible in both panoramas. Prediction used RGB, frozen manual region
rectangles, spherical SIFT/FLANN features and matches, and an estimated
relative pose. Reference pose and radial depth were opened only after the
prediction file had been made read-only.

## Result

| Check | Result |
| --- | ---: |
| Correct object links | 3 / 3 |
| Incorrect object links | 0 |
| Identity precision / recall / F1 | 1.0 / 1.0 / 1.0 |
| Repeated unique IDs | 3 / 3 identical |
| Relative rotation error | 0.268 degrees |
| Translation-direction error | 0.283 degrees |
| Median regional 3D-center error | 0.0067 m |
| Maximum regional 3D-center error | 0.0139 m |
| Exact identity set after ±16 px boundary changes | 2 / 2 |
| Incorrect links under ±32 px boundary stress | 0 |

The same-class cross-desk alternatives were rejected by their geometric
support. Under 32-pixel contraction, smaller regions lost enough supporting
features to be rejected. Under 32-pixel expansion, the two desk regions
overlapped enough evidence to be marked ambiguous. Both are fail-closed
outcomes: recall falls under extreme boundary error, but no false identity is
created.

## Reproduce

```bash
python benchmarks/object_localization/run_practical_validation.py --preflight
python benchmarks/object_localization/run_practical_validation.py \
  --output-dir /private/tmp/panorai-object-localization-practical-final
```

The runner verifies exact RGB and depth checksums. It writes
`prediction.json`, `summary.json`, `hypotheses.csv`, and
`annotated-regions.png` without copying dataset bytes into this repository.

## Limits

- One image pair and one indoor room are covered.
- Regions are manual post-hoc RGB annotations; detector/CAM quality is not
  evaluated.
- Metric scoring applies the reference baseline only after scale-free
  prediction is frozen.
- The spatial target is the median of supporting feature points, not the
  physical centroid of the object.
