# Localize and associate semantic regions

**Outcome:** turn directional learned evidence into auditable angular regions
and, when multiple registered views agree, spatial object hypotheses.

![Semantic directional evidence in an industrial panorama](../../_static/tutorials/application-semantics-boiler-room.jpg)

```text
pretrained CNN -> spherical FCN -> CAM -> angular proposal
                                         |
                      registered views --+-> spatial hypothesis
                                         |
                                         +-> optional mask refinement
```

## Start

1. [Generate a spherical CAM](spherical_cam.md) with a pretrained classifier.
2. [Interpret its resolution and support](interpreting_cam.md) without treating
   it as a segmentation mask.

## Build proposals

3. Convert evidence into finite
   [angular and FOV proposals](angular_proposals.md).

## Associate views

4. [Associate proposals across registered views](multiview_association.md) and
   create a spatial hypothesis only when the evidence supports it.

## Extend and evaluate

5. Optionally [refine an accepted region](mask_refinement.md) with local masks.
6. [Evaluate every stage](evaluation.md) without claiming ground-truth
   segmentation.
