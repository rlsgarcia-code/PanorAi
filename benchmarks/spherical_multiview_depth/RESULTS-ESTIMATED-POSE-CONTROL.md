# Estimated-pose versus oracle dense-depth control

## Frozen protocol

The experiment used the native 4128x8256 spherical Metric3D-v1
ConvNeXt-Large/Hourglass W121 depth prior from VAL-026. It is not the Tiny
model. No RGB or depth observation was resized.

Only W119 was used. Its image-estimated pose passed the frozen VAL-030 quality
gate with 21/32 inliers, 0.16749-degree rotation error, and 0.49718-degree
translation-direction error. The estimated route used that `R` and translation
direction, attaching only the 2.78519 m registered baseline norm because
two-view pose has no absolute translation scale. The oracle control used the
complete registered metric `R,t`. W124 was excluded because its estimated pose
failed the gate.

Both routes used the same 33,282-location native tangent grid and VAL-032
continuous residual weights. The two predictions and their hashes were frozen
before P74 ground truth was opened.

## Proposal sensitivity

| Grid proposal route | Accepted | GT-valid | Prior AbsRel | Proposed AbsRel | Improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full oracle `R,t` | 7,084 | 5,975 | 0.28121 | **0.25809** | 58.19% |
| Estimated `R,t` direction + oracle norm | 5,350 | 4,498 | 0.29880 | 0.31357 | 50.51% |

The routes shared 4,113 accepted nodes, a Jaccard overlap of only 0.494. At
shared nodes, median absolute log-range disagreement was 0.03999 and P90 was
0.33013. Thus sub-degree pose error changes both which local ZNCC minima pass
and which depth basin is selected. The estimated proposals improve the median
error but not their mean, indicating a consequential outlier tail.

## Dense result

| Full native map | ConvNeXt-Large seed | Estimated pose | Oracle pose |
| --- | ---: | ---: | ---: |
| AbsRel | 0.310326 | 0.298312 | **0.279078** |
| delta-1 | 0.509802 | 0.561892 | **0.598841** |
| RMSE (m) | 1.490122 | 1.463806 | **1.412988** |
| Scale-aligned relative 3D RMSE | 0.400933 | 0.392821 | **0.381656** |
| Scale-invariant log RMSE | 0.374323 | 0.370457 | **0.357784** |
| Log-depth correlation | 0.718195 | 0.720958 | **0.737647** |
| Mean normal error | **48.2036 deg** | 59.2565 deg | 60.2651 deg |
| ERP seam MAE | **0.089395 m** | 0.093865 m | 0.090553 m |

The estimated-pose route still improves global metric and scale-invariant
depth over the CNN seed. However, it recovers less than half of the oracle
AbsRel gain: 0.01201 versus 0.03125. At an absolute log-residual threshold of
0.005 it changes 21,560,118 evaluation pixels, of which 57.41% improve. The
oracle changes 22,018,882 pixels, of which 64.11% improve.

The resulting estimated and oracle dense maps differ by mean absolute log
depth 0.04816, median 0.02436, and P90 0.13292. The control proves that even a
high-quality sub-degree image pose materially affects dense depth.

Both single-source routes substantially worsen normals. This is not evidence
against the estimated pose specifically: the oracle route is slightly worse.
It confirms that a dense field driven by thousands of single-source proposals
needs source consensus or a robust piecewise-constant model before it can be
treated as a surface reconstruction.

## Conclusion and scope

The requested estimated-pose route is viable as a depth-improving signal, but
not interchangeable with the registered oracle. The current evidence supports
this ordering:

1. use the strongest available ConvNeXt-Large depth prior;
2. quality-gate relative pose before depth search;
3. preserve an oracle-pose control to measure pose sensitivity;
4. require multiple accepted source poses or robust proposal rejection;
5. avoid claiming geometric improvement from depth metrics alone when normals
   regress.

Outputs remain external under
`/private/tmp/panorai-val033-estimated-pose-depth-control`. P74, model,
predictions, and PLYs are not redistributed. This is source-checkout research
evidence, not an installed-wheel, stable API, release, or publication claim.
