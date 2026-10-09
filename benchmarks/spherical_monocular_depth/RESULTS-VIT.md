# P74 Metric3Dv2 ViT native-density tangent control

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
