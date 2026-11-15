# Spherical selection and movement

The Workbench uses one internal interaction layer for catalog components that
need human observations on spherical data. A component does not receive a
custom canvas adapter. Its typed input semantics select an interaction
contract.

## Contract resolution

`src/spherical-interaction.ts` resolves explicit `semantic`, `kind`, and `grid`
metadata into four decisions:

- the selection primitive: point, line, polygon, brush, or correspondence;
- the editing projection;
- the persisted coordinate space;
- the interpolation policy and number of required views.

The resolver never uses array shape, dtype, colour, or a component name.
Ambiguous arrays therefore produce no interaction contract.

| Semantic family | Primitive | Editing view | Persistence | Interpolation |
| --- | --- | --- | --- | --- |
| ERP point, bearing, keypoint, seed | point | gnomonic | ERP pixel centre | none |
| ERP line, great circle, line family | line | gnomonic | ERP pixel centres | none |
| region, polygon, ROI | polygon | gnomonic | ERP pixel centres | nearest |
| mask, labels, support, validity | brush | gnomonic | native ERP mask | nearest |
| correspondence, match | paired point | two gnomonic views | ERP pixel centres | none |

Line evidence is straight only in its gnomonic chart. The persisted endpoints
are reconstructed through panorama-frame rays and use the geometry-v1
pixel-centre convention. An ERP overview may show the resulting finite
great-circle arc, but it is not an editing plane.

## Shared controls

The same navigation functions back annotation, planar reconstruction, and
known-height mapping:

- hand/orbit navigation with wrapped longitude and bounded latitude;
- drag or wheel zoom with a bounded FOV;
- selection handles in the active gnomonic chart;
- canonical chart-to-ray-to-ERP conversion;
- seam-safe ERP context rendering.

Application-specific code declares the required semantic observation and
stores the result. It must not reimplement spherical navigation or projection
geometry.

The visible control is a single compact rail attached to the left edge of the
image workspace in Single, Two, and Multi View. The rail owns only reusable
instruments: hand, zoom, one point selector, one line selector, and any
contextually valid region tools. It never owns scientific labels such as
"direction A", "floor side", or "contact with floor". The workflow panel
declares which typed field or plane context receives the next point or line.

Tools remain visible when a workspace cannot consume their primitive, but are
disabled with an explicit reason. The Workbench never records an untyped line
or point merely because a button exists.

Precision editing follows selection rather than a separate mode. With the
point tool active, dragging a visible point handle updates that observation in
the active gnomonic chart; clicking away from a handle creates a point for the
current workflow context. The line tool follows the same rule for endpoints:
dragging a handle refines the existing line, while dragging elsewhere creates
one new line. Coordinate feedback is shown in original ERP pixel-centre space.
Multi View applies the same point rule to human correspondence observations.

Selection review belongs to the shared rail, not to the scientific workflow
panel. After a point is moved, or after a line-assist proposal is available, a
narrow rail extension presents the manual and adjusted observations with
explicit **keep manual** and **accept adjustment** actions. The central image
shows both candidates at the same time. No adjusted coordinate is persisted
before that decision.

The rail also exposes a projection toggle. The gnomonic chart is the editing
view for points and straight line evidence; the ERP overview is a seam-safe
context view of the persisted point or great-circle arc. Choosing point or
line while the ERP overview is open returns to the gnomonic chart before any
coordinate can be edited.

## Reviewable line refinement

After a user draws a line, the Workbench may request an annotation-assist
proposal. The assist:

1. samples the image into the declared gnomonic chart through public
   `panorai.geometry` APIs;
2. crops a bounded neighbourhood around the manual segment and runs
   `opencv.Canny` in that straight-line editing domain;
3. limits candidates to a narrow band around the manual segment;
4. performs a deterministic Workbench-owned robust fit and TLS refinement;
5. returns a crop, edge evidence, residuals, the untouched manual line, and a
   separate recommendation.

If the requested thresholds do not support a fit, the backend performs one
bounded second crop-Canny pass at half thresholds. The selected thresholds
and whether that adaptive pass was used are returned in provenance. The crop,
manual line, and detected edge points remain visible even when neither pass
supports a recommendation.

Edge detection never scans the complete ERP: the crop shape and gnomonic
domain are recorded in provenance. Accepted endpoints are converted and
reported in original-ERP pixel-centre coordinates. Only one line review may be
active. A newer request aborts the older browser request, and an eight-second
UI guard always offers a safe explicit continuation with the untouched manual
line.

The robust fit remains an annotation convenience, not a PanorAi estimator.
It is recorded as `panorai-workbench-annotation-assist`. The recommendation is
never applied automatically: keeping the manual line and accepting the
recommendation both create explicit provenance entries in a new immutable
revision. No supported edge disables acceptance and preserves the manual
observation.

A pending line analysis or an actionable recommendation disables downstream
alignment validation. When a recommendation exists, the user must explicitly
keep the manual observation or accept the recommendation before the scientific
workflow can advance.

Only an actionable recommendation is a review gate. If Canny finds no
supported edge, the request times out, or the assist is unavailable, the
manual line remains the accepted observation and the diagnostic becomes
non-blocking. The user may immediately mark the next line, adjust an endpoint,
or validate once enough geometric evidence exists. The panel remains visible
only as evidence and may be dismissed explicitly.

## Orientation lifetime and feedback

The accepted orientation is stored independently from the latest segmentation
run. Seed, mask, constraint, and plane revisions invalidate downstream range
artifacts but preserve the accepted reference frame. Height changes, line
changes, endpoint adjustments, vanishing observations, and floor-side changes
invalidate that frame and require validation again.

The inspector presents display-only values derived from the public
`KnownHeightPlanarMap`: total tilt required to align the estimated camera
zenith with canonical `+Y`, the right-handed correction axis in camera
coordinates, the estimated zenith in original ERP pixel-centre coordinates,
and the complete `rotation_world_from_camera` matrix. The Workbench does not
refit or replace the public orientation result.

## Progressive planar state

Known-height mapping keeps accepted upstream evidence separate from the most
recent pipeline run. Accepting a segmentation, adding contact observations, or
creating another region invalidates downstream range artifacts, but it does
not invalidate the measured height or accepted orientation. Only an edit that
changes the metric reference frame sends the workflow back to alignment.

The plane list is monotonic across revisions: reviewed regions remain visible
while the floor is resolved first, followed by vertical planes connected to
the floor and then horizontal planes connected to a resolved vertical plane.
The latest published metrics remain inspectable and are labelled with their
source revision until the new revision is executed.

Plane metrics are presentation derived from public `PlaneEstimate` and
`PlanarRangeResult` values. They include the world normal, plane offset,
camera-to-plane distance, valid pixel count, radial-range statistics, and the
two-dimensional extent of the currently visible published samples. That last
quantity is never presented as physical area or as the complete dimensions of
the plane. Unresolved planes remain in the list with the public diagnostic
reason and receive no invented metric geometry.
