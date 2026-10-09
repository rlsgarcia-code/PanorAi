# Semi-dense spherical tangent-grid result

## Frozen protocol

The P74 W121 experiment used the frozen native 4128x8256 ConvNeXt-L Metric3D
radial prior and registered metric poses to W119 and W124. No image or depth
resize was performed. Ground truth was opened only after all predictions and
their hashes were written to `FROZEN-BEFORE-GT.json`.

The grid contained 33,282 locations at 32-pixel spacing. Each location used 49
tangent samples over a one-degree angular radius and 17 log-inverse-range
hypotheses from `prior / 1.5` to `prior * 1.5`. A proposal required sufficient
support/texture, ZNCC cost at most 0.4, an interior range minimum, and a positive
best-versus-second cost margin. W119 and W124 were processed independently.

## Grid proposals

| Route | Accepted | GT-valid | Prior AbsRel | Proposed AbsRel | Improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| W119 | 7,084 | 5,975 | 0.2812 | 0.2581 | 58.19% |
| W124 | 6,616 | 5,741 | 0.2735 | 0.2433 | 61.47% |
| Compatible union | 7,832 | 6,713 | 0.2911 | 0.2476 | 64.50% |
| Strict two-source consensus | 1,320 | 1,169 | 0.2238 | **0.1352** | **76.30%** |

The median consensus error fell from 0.1713 to 0.0382. This is much stronger
coverage than the eleven DoG-derived VAL-030 seeds and confirms that regular
sampling is useful when pose and the prior constrain the search.

## Dense maps

The same bounded spherical color-aware splat used by VAL-030 propagated each
accepted grid residual over its one-degree support.

| Full native map | CNN seed | Grid consensus | Grid union |
| --- | ---: | ---: | ---: |
| AbsRel | 0.310326 | 0.304179 | **0.292480** |
| delta-1 | 0.509802 | 0.522725 | **0.561290** |
| Scale-aligned relative 3D RMSE | 0.400933 | 0.398947 | **0.398899** |
| Scale-invariant log RMSE | 0.374323 | **0.372321** | 0.372598 |
| Log-depth correlation | 0.718195 | 0.720610 | **0.720862** |
| Mean normal error | **48.2036 deg** | 50.8777 deg | 60.7943 deg |
| ERP seam MAE | **0.08940 m** | 0.09238 m | 0.10328 m |

Within changed evaluation support, the union improved 61.09% of 9,517,729
pixels and reduced AbsRel from 0.2950 to 0.2598. Consensus improved 68.29% of
2,034,664 pixels and reduced AbsRel from 0.2295 to 0.1722.

The result is therefore mixed but informative:

- the grid solves the coverage problem and improves metric depth and global
  scale-invariant structure;
- two-source consensus is substantially more reliable per seed;
- the current independent one-degree splats introduce patch boundaries and
  high-frequency surface noise, causing a severe normal regression;
- union maximizes global depth improvement, while consensus is the safer basis
  for a later continuous residual solve.

The next step should keep the frozen grid proposals and replace only the splat
with a smooth edge-aware residual field. Re-running or loosening matching gates
is not justified by this evidence.

## Runtime and artifacts

The two native grid searches took 2.28 s and 1.95 s; propagation plus artifact
preparation brought the pre-evaluation experiment to 8.51 s on the recorded
CPU environment. Outputs are under
`/private/tmp/panorai-val031-grid-tangent-depth`, including native NPY/PNG maps,
proposal NPZ, three verified binary PLY clouds, comparison panel,
`results.json`, and a self-contained `viewer.html`.

P74/model bytes remain external. This is development-only source-checkout
evidence, not an installed-wheel, public API, or publication claim.
