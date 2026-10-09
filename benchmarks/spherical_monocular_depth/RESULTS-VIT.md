# P74 Metric3Dv2 and Depth Anything 3 native-density tangent controls

## Frozen protocol

VAL-034 evaluates the P74 `W_121` panorama at the audited native angular ERP
shape, 4128x8256. The source ERP is never resized and no image pyramid or
prefilter is used. Instead, 42 overlapping gnomonic windows are sampled
directly from that lattice at 812x1400. Their focal length is the ERP's
equatorial density, 1313.99 px/rad, so their 56.09 by 34.34 degree field of
view matches Metric3Dv2's official 616x1064, 1000 px canonical camera while
retaining the native angular sampling density.

The unchanged official Metric3Dv2 ViT-Small/RAFT-4iter and
ViT-Large/RAFT-8iter weights run on every window. Axial depth is converted to
radial range with the exact window rays and the official focal scaling, then
PanorAi's gnomonic Gaussian workflow reconstructs one native ERP. Predictions
and the ConvNeXt-Large control are capped at the same 15 m evaluation range.
Ground truth was opened only after each prediction, mask, and protocol were
written to `FROZEN-BEFORE-GT.json`.

This is the perspective/tangent multiface ViT control. It is not yet a port of
ViT attention or positional encoding to spherical tokens.

## Result

All rows use the identical 22,991,801-pixel common mask.

| Native W121 map | ConvNeXt-Large | ViT-Small | ViT-Large |
| --- | ---: | ---: | ---: |
| Parameters | 203.2 M | 37.5 M | 411.9 M |
| AbsRel | 0.309563 | 0.240968 | **0.190422** |
| delta-1 | 0.509827 | **0.623720** | 0.601461 |
| RMSE (m) | 1.488742 | 1.173462 | **1.171125** |
| Scale-aligned relative 3D RMSE | 0.400226 | 0.315455 | **0.263836** |
| Scale-invariant log RMSE | 0.373701 | 0.288469 | **0.243768** |
| Log-depth correlation | 0.719605 | 0.814461 | **0.870491** |
| Optimal evaluation-only scale | 0.928899 | 0.944871 | 1.230119 |
| Mean normal error (deg) | 48.0660 | 38.3013 | **36.8352** |
| ERP seam MAE (m) | 0.083342 | **0.007925** | 0.011671 |
| Inference time (CPU s) | prior artifact | **88.87** | 453.81 |

The ViT result is materially better, not merely smoother. ViT-Large reduces
AbsRel by 38.5%, scale-aligned 3D error by 34.1%, and mean normal error by
11.23 degrees relative to the spherical ConvNeXt-Large control. ViT-Small has
the best delta-1 and seam, while ViT-Large recovers substantially better
scale-invariant structure. Large remains globally subscaled: its optimal
evaluation-only multiplier is 1.2301. That scale is reported, never applied to
the frozen prediction.

Angular context is a necessary part of the result. A preliminary Large run
kept native sampling but used only 280x504 windows (21.71 by 12.16 degrees).
It regressed to AbsRel 0.320210, scale-aligned 3D RMSE 0.408152, and 47.44-degree
normal error. Increasing capacity without preserving the learned field of view
therefore fails; native density and learned angular support must be preserved
together.

## Artifacts and limitations

Primary external artifact directory:
`/private/tmp/panorai-val034e-vit-large-native-trained-fov`.
It contains raw and 15 m-capped native NPY maps, explicit validity, two binary
PLYs, the freeze manifest, panel, results JSON, and an interactive viewer that
also includes GT and ViT-Small. The ViT-Small counterpart is under
`/private/tmp/panorai-val034d-vit-small-native-trained-fov`.

This is one development panorama, not a dataset aggregate. The transformer is
still evaluated on a tangent atlas; attention, learned position, and decoder
convolutions remain planar inside each window. The official checkpoints remain
external and their repository provides no separate checkpoint/model-card
license. No weights, P74 bytes, maps, or point clouds are committed.

## Depth Anything 3 Metric Large follow-up

VAL-035 applies the identical 42-view 812x1400 atlas to the newer official
Depth Anything 3 Metric Large model (334.2 M parameters). The standard DA3 API
was deliberately bypassed because it resizes inputs to a processing resolution;
the unchanged network receives each native-density tangent tensor directly.
Its axial output is converted to metres with the official `focal / 300` rule,
then to radial range with the exact tangent rays. No fitted scale, mean/std
alignment, source resize, prediction resize, pyramid, or prefilter is used.

All three rows below use the same 22,991,801 pixels and 15 m cap.

| Native W121 map | ConvNeXt-Large | Metric3Dv2 ViT-Large | DA3 Metric Large |
| --- | ---: | ---: | ---: |
| AbsRel | 0.309563 | **0.190422** | 0.196638 |
| delta-1 | 0.509827 | 0.601461 | **0.710510** |
| RMSE (m) | 1.488742 | **1.171125** | 1.261193 |
| Scale-aligned relative 3D RMSE | 0.400226 | **0.263836** | 0.338527 |
| Scale-invariant log RMSE | 0.373701 | **0.243768** | 0.264366 |
| Log-depth correlation | 0.719605 | **0.870491** | 0.853844 |
| Optimal evaluation-only scale | 0.928899 | 1.230119 | **1.070964** |
| Mean normal error (deg) | 48.0660 | 36.8352 | **34.2932** |
| ERP seam MAE (m) | 0.083342 | 0.011671 | **0.008430** |
| Inference time (CPU s) | prior artifact | 453.81 | **335.51** |

DA3 is the strongest local-surface result: it improves delta-1 by 10.90
percentage points, mean normal error by 2.54 degrees, seam continuity, and
unscaled metric calibration relative to Metric3Dv2 ViT-Large. It does not win
every structural measure. Metric3Dv2 retains 3.3% better AbsRel and 28.3%
better scale-aligned 3D RMSE, with higher log-depth correlation. The band
breakdown localizes most of DA3's loss to the equatorial band (AbsRel 0.2976
versus 0.2607); DA3 is better at the P74 south-support boundary (0.0958 versus
0.1183). The evidence therefore favors DA3 when local surface orientation and
metric hit rate matter, while Metric3Dv2 better preserves the scene's global
scale-invariant layout on this panorama.

The external VAL-035 artifact root is
`/private/tmp/panorai-val035-da3metric-large-native-trained-fov`. It contains
the native prediction/mask, freeze manifest, four PLYs, panel, static preview,
results JSON, and a self-contained interactive viewer. The official checkpoint
is 1,336,734,448 bytes with SHA-256
`bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776`.
The DA3 Metric Large model card declares Apache-2.0; source and weights remain
external and are not included in PanorAi artifacts.

## Spherical overlap feature sharing

VAL-036 isolates one proposed adaptation before changing attention or the DPT.
The frozen DA3 backbone produces four `58x100x1024` feature lattices per view.
For each target lattice, at most six overlapping source views are sampled at
the same panorama-frame rays. A deterministic center-weighted consensus is
then decoded by the unchanged official DPT. All 334.2 M learned parameters are
identical to VAL-035; there is no fitting or scale alignment.

| Native W121 map | DA3 Metric Large | Shared spherical features | Relative change |
| --- | ---: | ---: | ---: |
| AbsRel | **0.196638** | 0.199603 | +1.51% |
| delta-1 | **0.710510** | 0.689916 | -2.90% |
| RMSE (m) | 1.261193 | **1.260766** | -0.03% |
| Scale-aligned relative 3D RMSE | 0.338527 | **0.336350** | -0.64% |
| Scale-invariant log RMSE | 0.264366 | **0.264298** | -0.03% |
| Log-depth correlation | **0.853844** | 0.853146 | -0.08% |
| Mean normal error (deg) | **34.2932** | 36.0553 | +5.14% |
| ERP seam MAE (m) | 0.008430 | **0.006461** | -23.36% |

The communicated map differs from the control by 0.129 m mean absolute depth
over its valid support, with a 0.036 m median and 0.578 m 95th percentile.
Visually, the global layout remains essentially the same. The large seam gain
and small scale-invariant 3D gain show that panorama-ray correspondence is
working; the worse hit rate and normals show that unconditional averaging
mixes view-conditioned representations and softens local geometry.

This is deliberately not described as a spherical DA3 decoder. The DPT remains
planar and per-view. A native dense ERP DPT is also not a practical direct
port: a single 4128x8256x256 float32 activation is about 32.5 GiB. The next
architecturally valid step is a sparse/streaming atlas DPT with spherical halo
exchange, followed by attention inside the backbone using spherical relative
positions. The external VAL-036 artifact root is
`/private/tmp/panorai-val036-da3-shared-features`; it contains the native map,
freeze manifest, three PLYs, panel, preview, results JSON, and interactive
viewer. Model weights and P74 data remain external.
