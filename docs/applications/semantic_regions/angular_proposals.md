# Convert CAM evidence into angular proposals

**Level:** Compose · **Output:** finite spherical centre and FOV candidates

The useful analogy to the R-CNN family is proposal handling, not rectangular
ERP boxes:

| Detector pipeline | Spherical semantic pipeline |
| --- | --- |
| objectness map | CAM/proxy evidence |
| image box proposal | angular centre + horizontal/vertical FOV |
| box IoU/NMS | solid-angle overlap/NMS |
| ROI crop | gnomonic view centred on the proposal |
| box refinement | centre and FOV refinement |

A proposal should cover enough of the candidate object for a local gnomonic
view without becoming an entire-room hypothesis. Generate several centre/FOV
anchors around stable CAM regions, score their supported evidence, reject
unsupported area and suppress near-duplicates with solid-angle overlap.

## From score map to candidate components

`semantic_regions_from_map()` provides the first deterministic split. It
thresholds a normalized score map, joins components with horizontal seam
wraparound, associates supported spherical features, and reports rejected,
truncated and seam-crossing counts.

```python
from panorai.object_localization import (
    SemanticRegionProposalConfig,
    semantic_regions_from_map,
)

components = semantic_regions_from_map(
    cam_hw,
    erp_shape_hw=erp_rgb.shape[:2],
    features=features,
    region_id_prefix="view-a/grid",
    class_id=class_id,
    class_name=class_name,
    config=SemanticRegionProposalConfig(
        threshold=0.50,
        min_component_cells=2,
        min_feature_count=2,
    ),
    source_id="resnet18-imagenet/cam",
)
```

The threshold is an experimental proposal parameter, not a calibrated
probability boundary. A component without enough accepted features remains in
the diagnostics but cannot support later geometric association.

## Anchor centre and extent

For each retained component, generate centre/FOV anchors at several angular
scales and aspect ratios. Score an anchor by supported semantic mass,
feature/track support and penalties for unsupported or excessive area. Use
minimum and maximum FOV bounds; otherwise a peak becomes a tiny crop or a weak
component expands to the whole room.

Compare anchors with solid-angle intersection-over-union. The merge rule must
distinguish duplicate observations of one region from incompatible concepts.
This is the spherical counterpart of anchor generation, box regression and
NMS—not an ERP bounding-box approximation.

Each proposal retains:

```text
concept and exact proxy labels
centre bearing
horizontal and vertical FOV
semantic support score
support/validity coverage
source model, CAM and view provenance
```

The proposal is intentionally uncertain. Cross-view evidence decides whether
it should be rejected, merged, kept ambiguous or promoted.

Next: [associate proposals across views](multiview_association.md). The full
method and ablations are in the
[object-hypothesis tutorial](../../tutorials/11_cam_guided_object_hypotheses.md).
