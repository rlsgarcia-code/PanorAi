# Known-height planar mapping

`panorai.reconstruction.known_height_planar` is an Experimental public
application for a single equirectangular panorama. It maps reviewed planar
regions into a local metric frame from the known optical-centre height above
the floor. It does not infer metric geometry from a mask or colour alone.

The application owns a real, deterministic public `Pipeline`:

```text
ValidateKnownHeightPlanarScene → ResolvePlanarReferenceFrame
→ SegmentKnownHeightRegions → AnchorFloorPlane
→ ResolveConnectedPlanes → ResolvePlanarPreviews → RasterizeKnownHeightRanges
→ AssembleKnownHeightPlanarMap
```

Only accepted final masks and accepted `PlanarConstraint` relations participate
in geometry. A segmentation proposal remains `draft`; an unconfirmed relation
remains `proposed`. Free planes, short contacts, cycles, and inconsistent
orientation observations are reported as unresolved rather than guessed.

## Coordinate and range contract

The metric height is a required user measurement; the application never
supplies or estimates it from the panorama. Before segmentation, the image must
be levelled from reviewed horizontal evidence. The preferred observation is
two direction families of floor lines, with at least two line segments in each
family. Each family determines a spherical vanishing direction from the
intersection of its interpretation planes; the two directions determine the
horizontal great circle. Two directly marked horizontal vanishing points are
also accepted. The down-side observation chooses the sign of gravity.

Existing revisions may continue to provide explicit horizon points. Those
points fit a great circle by TLS; disagreement above three degrees produces
`needs-review`. Incomplete or degenerate line families never fall back to an
invented orientation. The local frame is `+Y` up, the camera centre is
`[0, camera_height_m, 0]`, and the floor is `Y=0`. Heading follows the camera
forward direction projected onto the horizontal plane.

Output range is radial distance in metres. Support, image validity, range
validity, conflicts, values and uncertainty are separate arrays. Internal black
pixels remain valid. Masks and labels use nearest-neighbour sampling. When
`camera_height_sigma_m` is absent, metric uncertainty is explicitly
unavailable (`NaN`), never zero.

## Public usage

```python
from panorai.reconstruction.known_height_planar import (
    KnownHeightPlanarMapping,
    KnownHeightPlanarOptions,
    KnownHeightPlanarScene,
    PlanarConstraint,
)

scene = KnownHeightPlanarScene(
    panorama=panorama,                 # EquirectangularImage, RGB uint8
    camera_height_m=1.60,
    camera_height_sigma_m=0.02,
    down_erp_xy=(511.5, 510.0),
    floor_line_families_erp_xy=(floor_lines_a, floor_lines_b),
    regions=reviewed_regions,          # public PlanarRegion values
    constraints=reviewed_constraints,
    revision="survey-004",
)
application = KnownHeightPlanarMapping(
    KnownHeightPlanarOptions(segmentation="manual")
)
run = application.run(scene)
result = run.output
print(result.state, result.unresolved)
```

`floor-anchor` fixes the metric floor. The generic reviewed relations are
`orthogonal-to`, `parallel-to` and `coplanar-with`. A reviewed orientation plus
one contact positions an orthogonal or parallel plane; additional contacts
refine the offset robustly. Without orientation, an orthogonal relation needs a
two-point contact line with at least 10 cm span. The older
`vertical-to-floor` and `horizontal-to-plane` names remain compatible aliases.

`PlanarPreview` keeps `orientation-only`, `positioned-draft` and `resolved`
geometry separate. Only `resolved` previews are publishable. Draft previews
never enter `KnownHeightPlanarMap.planes` or range rasterization. Interactive
clients may call `preview_planar_relation` to project a cursor contact without
reimplementing spherical geometry.

## Workbench

Open `/known-height-planar.html`, or choose **Mapeamento planar por altura
conhecida** in Single View. A real CC0 ERP loads by default, but no metric
height or orientation is assumed. The user first enters the measured optical
centre height, then uses **Descoberta assistida** to review automatically fitted
great-circle lines, vanishing bundles and orientation-only proposals for the
active plane. The user first associates line observations with that physical
plane, then chooses parallel subsets to estimate directions, and finally
combines two or more non-parallel directions into its orientation. A direction
is therefore never presented as a plane. Automatic and manually refined lines
share the same public contract and can be associated with any plane instance.
If a visible boundary is absent from the automatic overlay, **Desenhar reta
ausente** activates the gnomonic line tool in-place; the user explicitly keeps
the manual observation or accepts the local robust-fit recommendation before
it joins discovery.
The same step can switch to **Marcação manual** for two floor-line families or
two horizontal vanishing points. The selected evidence is validated before
floor segmentation. **Solução
analítica** loads a closed-form floor, wall and ceiling fixture for a complete
metric smoke test. Each human action creates a new in-memory revision; every execution
persists the processed revision, exact arrays, pipeline trace, presentation
rasters and a metric point cloud.

CPU segmentation is available without models. SAM is optional, is never
downloaded implicitly, and requires explicit acceptance of the upstream terms.
The Workbench only translates user revisions and renders public results; it does
not reimplement plane or range geometry.

Once the reference frame is accepted, the measured camera height already fixes
the floor at `Y=0`. The public result therefore exposes it in `PlanarPreview`
as `positioned-draft`, including a draft footprint when a mask exists. This
yellow preview is metric but non-authoritative: it remains absent from
`KnownHeightPlanarMap.planes` and range rasterization until both the floor mask
and the `floor-anchor` relation are explicitly accepted. A resolved/publicable
floor is rendered only after that reviewed revision is executed.

## Limitations

The application is an anchored incremental solver, not unrestricted global
plane adjustment. It does not infer coplanarity from appearance and does not
resolve a disconnected plane with free orientation. Height uncertainty covers
only the supplied height observation; segmentation, camera calibration and
manual-contact uncertainty are not claimed to be calibrated.
