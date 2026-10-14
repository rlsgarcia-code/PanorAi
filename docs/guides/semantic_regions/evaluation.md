# Evaluate semantic regions and hypotheses

**Level:** Evaluate · **Evidence:** directional, regional and cross-view

Evaluate each stage separately:

1. **Classifier/FCN:** planar parity on the canonical crop and complete layer
   replacement.
2. **Spherical CAM:** seam continuity, supported solid-angle concentration and
   comparison with controlled tangent-view evidence.
3. **Proposal:** object coverage, excess FOV, centre stability and duplicate
   suppression.
4. **Association:** transferred-region overlap, track support, appearance
   consistency and agreement across views.
5. **Spatial hypothesis:** reprojection residual, number of independent views,
   position/scale uncertainty and rejection state.
6. **Optional mask:** spherical IoU and continuity only when a legitimate
   reference mask exists.

Do not report CAM as segmentation accuracy or use a selected attractive
overlay as the evaluation. Fix proxy vocabularies and thresholds before the
evaluation set, include failed channels and retain model, checkpoint, support,
pose and proposal provenance.

## Stage-specific measures

| Stage | Useful measures | Misleading substitute |
| --- | --- | --- |
| ported classifier | global-logit cosine, top-k retention, planar-layer audit | attractive CAM colour |
| CAM | angular peak error, solid-angle enrichment, rotation stability | upsampled pixel sharpness |
| proposal | target coverage, excess solid angle, centre/FOV stability, recall | centre-point hit alone |
| association | transferred IoU, track support, embedding agreement, ID precision/recall | nearest CAM peak |
| spatial estimate | angular reprojection, view count, uncertainty, metric error when scale exists | arbitrary-scale distance |
| optional mask | solid-angle IoU, boundary continuity, alternative-mask spread | planar ERP pixel IoU alone |

Report proposal recall together with proposal count and area. A method that
covers every target by proposing most of the sphere is not useful. For
association, split true repeated instances across views so that semantic label
agreement alone cannot solve the task.

## Minimum reproducibility record

Freeze images and splits, proxy labels and weights, checkpoint hashes, ERP
resolution, CAM layer/stride, support masks, proposal thresholds, anchor
scales/FOVs, pose source, matching configuration and random seeds. Publish
negative cases and all compared variants under the same visualization scale.

Use [benchmarking](../tutorials/08_benchmarking.md) for run records, the
[CAM tutorial](../tutorials/09_spherical_fcn_cam.md) for the CAM protocol
and the [object-hypothesis tutorial](../tutorials/11_cam_guided_object_hypotheses.md)
for association metrics.
