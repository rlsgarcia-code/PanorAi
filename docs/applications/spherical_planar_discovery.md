# Spherical planar discovery

`panorai.reconstruction.planar.SphericalPlanarDiscovery` is an Experimental,
CPU-first application for finding reviewable planar orientation evidence in one
equirectangular panorama. It does not estimate metric plane distance.

The public pipeline is:

```text
ValidatePlanarDiscoveryRequest → BuildOverlappingGnomonicAtlas
→ DetectAndFitGnomonicLines → MergeSphericalLineObservations
→ EstimateVanishingBundles → SuggestVanishingBundleLines
→ ProposePlanarOrientations
→ AssemblePlanarDiscoveryResult
```

Edges and finite segments are detected and robustly fitted only in overlapping
gnomonic charts. Endpoints return to the original ERP pixel-centre grid. Chart
duplicates are merged as great-circle observations, then line-plane
intersections are clustered in projective direction space, where `v` and `-v`
represent the same vanishing direction.

After each direction is estimated, `SuggestVanishingBundleLines` evaluates
the remaining great-circle observations against that fixed projective
direction. It reports a ranked, review-only set using the antipodal angular
residual `asin(abs(n @ v))`, where `n` is the line interpretation-plane normal
and `v` is the vanishing direction. Suggestions are never inserted silently.
Accepting one means adding its deterministic line ID to `bundle_memberships`
and rerunning the robust direction fit; direction-specific rejections are
preserved in `bundle_line_decisions` without rejecting that line globally.

The estimator does not impose a Manhattan world or a fixed number of vanishing
directions. Two or more non-degenerate direction bundles can define an
orientation-only plane proposal. Its public `PlaneEstimate` has
`normal_world=None` and `offset_m=None`; the proposal reports `normal_camera`
separately. A mask, metric anchor and reviewed geometric relation are still
required before the known-height application can publish range.

```python
from panorai.reconstruction.planar import (
    PlanarDiscoveryRequest,
    SphericalPlanarDiscovery,
)

request = PlanarDiscoveryRequest(panorama, revision="room-001")
result = SphericalPlanarDiscovery()(request)
print(len(result.lines), len(result.bundles), len(result.proposals))
```

For low-contrast architectural joints, callers may opt into chart-local CLAHE
before Canny with `local_contrast_clip_limit` (zero, the public default,
disables it). The Workbench's **Balanceado** preset uses a clip limit of `2`,
Canny `25–105`, 384-pixel charts and a 14-pixel minimum segment. These are
working-chart controls only: observations are still reported on the original
ERP pixel-centre grid, and contrast enhancement never changes the input image
or accepts a line automatically.

Human review is replayable. `SphericalLineSeed` adds a reviewed manual line on
the original ERP grid; its scientific great-circle fit still occurs in the
public pipeline. `line_decisions` accepts deterministic line IDs,
`bundle_memberships` replaces automatic membership with reviewed line groups,
`bundle_line_decisions` preserves rejected expansion candidates per direction,
`plane_bundle_groups` names bundles that the user considers coplanar, and
`plane_line_groups` associates automatic or manual lines with a physical plane
instance. Optional `PlanarRegion` masks provide nearest-neighbour regional
support without conflating image intensity, support or validity.
Unknown IDs and degenerate groups fail explicitly. Automatic proposals remain
`draft`; reviewed groups become accepted *orientation proposals*, never metric
planes.

The Workbench exposes this application as **Descoberta assistida** in the
known-height alignment step. Its guided flow keeps three entities explicit:

1. selected observations are associated with the active physical plane;
2. a parallel subset of those observations estimates one projective direction
   (a vanishing bundle);
3. two or more non-parallel directions estimate the plane orientation.

Global automatic bundles remain technical suggestions: the primary list shows
only directions supported by lines already associated with the active plane.
Each resulting direction offers **Expandir direção**, to rank other compatible
lines, and **Usar na orientação do plano**. Nothing is accepted implicitly.
Users can reject lines, create/replace directions, combine directions into a
plane, and explicitly apply one proposal as the floor's two line families. A
missing line can be added directly from this assisted flow with **Desenhar reta
ausente**. It is drawn in a gnomonic chart, passed
through the same local edge-refinement diagnostic, and remains a valid human
observation when no trustworthy edge is found. Human lines are retained in
addition to the bounded automatic line budget. **Marcação manual** remains
available for the legacy two-family workflow without discarding automatic
evidence.

## Limitations

- Canny/LSD evidence is appearance dependent and may fragment low-contrast or
  curved physical boundaries.
- Spatial co-occurrence ranks automatic plane pairs but does not prove that
  bundles belong to the same physical surface.
- Direction estimates are camera-frame, antipodal and orientation-only.
- Confidence is a deterministic support score, not calibrated probability.
- No model, checkpoint, download or semantic plane label is used.
