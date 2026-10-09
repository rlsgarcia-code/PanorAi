# Estimated-pose versus oracle dense-depth control

## Frozen protocol

The experiment used the native 4128x8256 spherical Metric3D-v1
ConvNeXt-Large/Hourglass W121 depth prior from VAL-026. It is not the Tiny
model. No RGB or depth observation was resized.

Only W119 was used. Its image-estimated pose passed the frozen VAL-030 quality
gate with 21/32 inliers, 0.16749-degree rotation error, and 0.49718-degree
translation-direction error. Because two-view pose has no absolute translation
scale, the primary estimated route inferred baseline magnitude from the frozen
ConvNeXt-Large target ranges and 32 tangent-feature bearing correspondences. A
robust median retained 19/20 candidates and estimated 2.30453 m, 17.26% below
the registered 2.78519 m baseline. A hybrid route attached the oracle baseline
norm to the estimated direction, and the full control used registered metric
`R,t`. W124 was excluded because its estimated pose failed the gate.

All three routes used the same 33,282-location native tangent grid and VAL-032
continuous residual weights. The predictions and their hashes were frozen
before P74 ground truth was opened.

## Proposal sensitivity

| Grid proposal route | Accepted | GT-valid | Prior AbsRel | Proposed AbsRel | Improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full oracle `R,t` | 7,084 | 5,975 | 0.28121 | **0.25809** | 58.19% |
| Estimated direction + oracle norm | 5,350 | 4,498 | 0.29880 | 0.31357 | 50.51% |
| Fully estimated `R,t` from CNN prior | 5,312 | 4,465 | 0.29673 | 0.34312 | 41.66% |

The hybrid/oracle routes shared 4,113 accepted nodes, Jaccard 0.494. Their
median/P90 absolute log-range disagreement was 0.03999/0.33013. The fully
estimated/oracle routes shared 3,721 nodes, Jaccard 0.429, and disagreement rose
to 0.19312/0.35478. Thus sub-degree direction error changes which local ZNCC
minima pass, while a 17% baseline error shifts the selected range basin enough
to make most proposals harmful.

## Dense result

| Full native map | CNN seed | Est. prior scale | Est. oracle scale | Oracle pose |
| --- | ---: | ---: | ---: | ---: |
| AbsRel | 0.310326 | 0.314074 | 0.298312 | **0.279078** |
| delta-1 | 0.509802 | 0.519885 | 0.561892 | **0.598841** |
| RMSE (m) | 1.490122 | 1.498689 | 1.463806 | **1.412988** |
| Scale-aligned relative 3D RMSE | 0.400933 | 0.404089 | 0.392821 | **0.381656** |
| Scale-invariant log RMSE | 0.374323 | 0.378838 | 0.370457 | **0.357784** |
| Log-depth correlation | 0.718195 | 0.711289 | 0.720958 | **0.737647** |
| Mean normal error | **48.2036 deg** | 59.3651 deg | 59.2565 deg | 60.2651 deg |
| ERP seam MAE | **0.089395 m** | 0.091926 m | 0.093865 m | 0.090553 m |

The hybrid route still improves global metric and scale-invariant depth, but
recovers less than half the oracle AbsRel gain: 0.01201 versus 0.03125. The
fully estimated route is negative: AbsRel, metric RMSE, scale-invariant 3D
RMSE, log correlation, and normals are all worse than the CNN prior. At an
absolute log-residual threshold of 0.005 it changes 21,631,528 evaluation
pixels; only 49.52% improve and local AbsRel increases 0.31050 to 0.31467.

The fully estimated and oracle dense maps differ by mean/median/P90 absolute
log depth 0.06934/0.04814/0.16811. For the hybrid these values are
0.04816/0.02436/0.13292. The control proves that both sub-degree direction
accuracy and metric translation scale materially affect dense depth.

Both single-source routes substantially worsen normals. This is not evidence
against the estimated pose specifically: the oracle route is slightly worse.
It confirms that a dense field driven by thousands of single-source proposals
needs source consensus or a robust piecewise-constant model before it can be
treated as a surface reconstruction.

## Conclusion and scope

The requested fully estimated `R,t` route is not yet a depth-improving signal.
The hybrid result shows useful photometric information remains, but it depends
on an oracle metric scale. The current evidence supports this ordering:

1. use the strongest available ConvNeXt-Large depth prior;
2. quality-gate relative pose before depth search;
3. estimate translation scale independently rather than recycling the same
   biased monocular depth prior;
4. preserve hybrid and full-oracle controls to separate scale/direction error;
5. require multiple accepted source poses or robust proposal rejection;
6. avoid claiming geometric improvement from depth metrics alone when normals
   regress.

Outputs remain external under
`/private/tmp/panorai-val033b-estimated-metric-pose-depth-control`. P74, model,
predictions, and PLYs are not redistributed. This is source-checkout research
evidence, not an installed-wheel, stable API, release, or publication claim.
