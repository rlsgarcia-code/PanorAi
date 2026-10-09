# Computer vision for spherical images: what PanorAi offers

PanorAi is a computer-vision library for central spherical images, not only a
projection utility. This guide starts from the result you want to obtain and
then explains how PanorAi preserves spherical geometry along the way.

The library uses two complementary strategies. Some operations evaluate rays,
angular neighbourhoods, or residuals directly on the sphere. Others create
gnomonic or cubemap images so mature planar algorithms can be reused, then map
their results back to panorama pixels, unit bearings, or an ERP image. Many
complete workflows are deliberately hybrid.

## Choose a computer-vision task

| Theme | What you can do | Typical input → result | Start here | Stability |
| --- | --- | --- | --- | --- |
| **Projection** | Convert among ERP, gnomonic views, cubemaps, and unit rays; process and reconstruct view sets | panorama or rays → projected images, rays, or reconstructed ERP | {doc}`02_projection_foundations` | Stable geometry; Compatibility samplers |
| **Image processing** | Convolve, smooth, equalize, transform, detect edges, and build pyramids without treating the ERP seam as an image border | ERP → enhanced or transformed ERP | {doc}`spherical_image_processing` | Experimental |
| **Deep learning depth** | Acquire one pinned external Metric3D CNN and port its learned spatial layers without resizing the ERP | one native 2:1 ERP → radial range on the same lattice | {doc}`10_spherical_monocular_depth` | Experimental |
| **Features and matching** | Detect SIFT/ORB/AKAZE features in overlapping views, or detect DoG keypoints directly on the sphere; match panoramas | one or two ERPs → descriptors and matched unit bearings | {doc}`03_features_and_matching` | Stable face-based core; spherical DoG Experimental |
| **Two-view geometry** | Estimate relative rotation and translation direction and triangulate correspondences | matched bearings → relative pose and sparse points | {doc}`04_two_view_geometry` | Experimental |
| **Multiview reconstruction** | Build pair graphs and tracks, initialize cameras and points, and run spherical bundle adjustment | 3+ panoramas/features → arbitrary-scale camera poses and sparse 3D map | {doc}`05_multiview_reconstruction` | Experimental |
| **Dense stereo** | Estimate radial range from two posed panoramas with spherical epipolar search and angular local evidence | two ERPs plus metric pose → dense radial range, validity, and confidence | {doc}`06_spherical_dense_stereo` | Experimental |
| **SLAM** | Track an ERP or calibrated central-fisheye sequence, create keyframes, map, relocalize, and correct loops | ordered frames → online trajectory, keyframes, and map | {doc}`07_spherical_slam` | Experimental |

These themes form an end-to-end path, but each can also be used independently:

![PanorAi computer-vision themes from spherical images to geometry and dense range](../_static/tutorials/computer-vision-themes.svg)

## How spherical images are handled

The execution domain is a mathematical choice; it is independent from whether
the implementation is Python, NumPy, OpenCV, or first-party C++.

| Execution domain | What happens | Best suited to | Output |
| --- | --- | --- | --- |
| **Sphere-native** | PanorAi works with unit rays, angular neighbourhoods, tangent frames, or spherical residuals without materializing a full planar view | convolution, gradients, equalization, rotations, bearing geometry, bundle adjustment | ERP-valued result, bearings, pose, or 3D structure |
| **Projection-domain** | PanorAi materializes gnomonic/cubemap images and applies an operation in those images | established planar detectors, descriptors, neural models, and view-local processing | projected images, or results mapped to ERP/bearings |
| **Hybrid** | A workflow combines sphere-native stages with small or full projected views | spherical DoG with tangent SIFT descriptors, face-based features followed by bearing geometry, projection/process/reconstruction | depends on the workflow |

Neither strategy is universally better. Sphere-native processing preserves a
constant angular interpretation across latitude and the longitude seam.
Projection-domain processing makes well-tested planar algorithms available in
locally low-distortion views. PanorAi's job is to make the chosen route and its
coordinate transformations explicit.

(capability-map-projection-foundations)=
## Projection

### What you can do with projection

Use canonical pixel-centre geometry to convert ERP pixels, gnomonic pixels,
cubemap pixels, and unit rays. Materialize one view or a batch, process views,
and reconstruct an ERP with explicit support and validity. The result is useful
on its own and is also the foundation for features, learned models, and
camera geometry.

**Typical input → result:** ERP/rays/projected images → views, bearings, or ERP.

**Start here:** {doc}`02_projection_foundations`; use {doc}`00_quick_start` for
the shortest path and {doc}`01_custom_pipeline` for batched processing.

| Capability | Public entry point | Execution domain | What PanorAi adds |
| --- | --- | --- | --- |
| ERP pixels ↔ rays | `erp_pixels_to_rays`, `rays_to_erp_pixels` | Sphere-native coordinates | Canonical `+X` right, `+Y` up, `+Z` forward frame, pixel centres, seam and pole rules |
| ERP ↔ one tangent view | `equirectangular_to_gnomonic`, `gnomonic_to_equirectangular`, `GnomonicProjector` | Projection-domain | Forward/inverse geometry, FOV and roll, seam-aware sampling, support and validity |
| ERP ↔ cubemap | `equirectangular_to_cubemap`, `cubemap_to_equirectangular` | Projection-domain | Fixed face order/bases and deterministic edge/vertex ties |
| ERP ↔ arbitrary view set | `EquirectangularImage.views`, `GnomonicFaceSet.reconstruct` | Projection-domain | Face provenance, modality-aware sampling, inverse projection, explicit-support blending |
| Process and reconstruct | `EquirectangularImage.process_views` | Projection-domain | ERP → views → caller operation → backprojection → blend orchestration; the caller owns the view-local operation |
| Batched differentiable projection | Torch gnomonic/cubemap APIs | Projection-domain | `NCHW` batch/channel/dtype/device preservation and gradients with respect to input values |
| Sphere sampling layouts | cube, icosahedron, Fibonacci, and Compatibility samplers | Sphere-native planning | View centres/FOVs in the canonical spherical frame; a raster exists only when projection is requested |
| Blending | average, feather, Gaussian, closest, Huber, and research blenders | Hybrid reconstruction | Combines already backprojected ERP layers using explicit masks; zero/black remains valid data |
| RGB, range, labels, masks | modality-aware projection workflow | Projection-domain | Radial range stays radial; categories use nearest sampling; support and data validity remain separate |

(capability-map-quick-start)=
### Quick start and object workflow

The functional and object APIs share the same geometry. `project` materializes
a gnomonic image; `back_project` reverse-maps it onto explicit ERP support.
`process_views` is therefore a projection-domain workflow, even if the caller
uses a filter whose name resembles a spherical operation. Use a
`panorai.image_processing` spherical function instead when the desired
neighbourhood must remain angularly constant everywhere.

(capability-map-cubemap-batch)=
### Cubemap, batch, and learned-model workflow

Cubemap and arbitrary-N views are first-class outputs. A model can operate on
those images and PanorAi can reconstruct its predictions into ERP. PanorAi
owns the projection, geometry, masks, and fusion; the caller or model owns the
semantics of the projected-image operation.

(capability-map-image-processing)=
## Image processing

### What you can do with image processing

Apply familiar OpenCV-style filtering concepts with angular rather than raw
ERP-pixel neighbourhoods. The current image-processing API is sphere-native:
it does not split the panorama into gnomonic tiles and stitch the result.

**Typical input → result:** ERP image → filtered, enhanced, transformed, edge,
or pyramid ERP data.

**Start here:** {doc}`spherical_image_processing`.

| Capability | Public entry point | Execution domain | What PanorAi adds |
| --- | --- | --- | --- |
| General 2D correlation | `spherical_filter2d` | Sphere-native spherical convolution | Kernel taps live in every ray's local east/north tangent frame, use constant angular spacing, and cross the ERP seam |
| Averaging and Gaussian smoothing | `spherical_box_blur`, `spherical_gaussian_blur` | Sphere-native spherical convolution | Angularly consistent support instead of latitude-dependent ERP-pixel support |
| First/second derivatives | `spherical_sobel`, `spherical_gradient`, `spherical_laplacian` | Sphere-native spherical convolution | East/north tangent derivatives, magnitude/orientation, and seam-consistent second derivatives |
| Median smoothing | `spherical_median_blur` | Sphere-native nonlinear neighbourhood | Median over a sampled geodesic tangent neighbourhood |
| Bilateral smoothing | `spherical_bilateral_filter` | Sphere-native nonlinear neighbourhood | Angular spatial weights plus radiometric weights |
| Edge detection | `spherical_canny` | Sphere-native hybrid of convolution and nonlinear stages | Spherical Gaussian/gradients followed by geodesic suppression and seam-connected hysteresis |
| Histogram equalization | `spherical_equalize_histogram` | Sphere-native global statistics | Optional solid-angle weighting avoids over-counting the dense ERP polar rows |
| Rotation | `spherical_rotate` | Sphere-native direct ray sampling | Inverse maps output rays through a proper 3×3 rotation, with no artificial image border |
| Resize | `spherical_resize` | Sphere-native direct ray sampling | Samples exact output ERP pixel-centre rays and treats longitude as periodic |
| Gaussian pyramid | spherical Gaussian pyramid API | Sphere-native convolution plus direct ray resizing | Angular prefiltering before each scale change |
| Laplacian pyramid | spherical Laplacian pyramid/reconstruction API | Sphere-native convolution plus direct ray resizing | Sphere-aware residual levels with tested reconstruction |

This distinction matters: linear convolution, nonlinear neighbourhood filters,
global histogram statistics, and geometric transforms are all sphere-native,
but only the linear kernel stages are *spherical convolution*.

(capability-map-features)=
## Features and matching

### What you can do with features and matching

Extract local features from a panorama, deduplicate overlap geometrically,
match descriptors, and obtain correspondence pairs as unit bearings. Choose a
stable projection-domain frontend or the Experimental sphere-native DoG
detector according to where keypoint selection must happen.

**Typical input → result:** one/two ERPs → spherical feature sets, descriptors,
matches, and paired unit bearings.

**Start here:** {doc}`03_features_and_matching`; see
{doc}`../how_to/spherical_features` for task-oriented recipes.

| Capability | Public entry point | Execution domain | What PanorAi adds |
| --- | --- | --- | --- |
| Stable SIFT/ORB/AKAZE frontend | spherical feature façade with `OpenCVFeatureBackend` | Projection-domain detection and description | Overlapping gnomonic views, masks, mapping back to ERP pixels/unit bearings, angular deduplication, provenance |
| Spherical DoG + SIFT | `SphericalDoGSIFTPipeline` | Hybrid: sphere-native Gaussian/DoG detection, one local tangent patch per descriptor | Seam-aware keypoint selection, angular scale, tangent orientation, and standard spherical result objects |
| Descriptor matching | BF/FLANN through the spherical matcher | Descriptor-domain, then sphere-native correspondence handling | Match provenance, angular deduplication, and aligned bearing pairs |
| Multiscale visual context | Experimental wide/local view routing | Projection-domain frontend, sphere-native deduplication | Explicit scale/view provenance and spherical NMS |
| COLMAP export | PyCOLMAP export adapter | Projection-domain virtual-camera representation | Consistent gnomonic camera rigs and feature/match export |

PanorAi does not reimplement the SIFT descriptor. In the Experimental route,
PanorAi detects DoG extrema on a spherical scale space and materializes only a
small tangent patch at each selected bearing; OpenCV computes the SIFT
descriptor on that patch. In the stable route, both detection and description
happen in overlapping gnomonic views.

(capability-map-two-view)=
## Two-view geometry

### What you can do with two views

Turn matched panorama features into a relative camera pose without pretending
that ERP pixel coordinates belong to a pinhole camera. The estimator consumes
unit bearings, scores hypotheses with spherical residuals, checks cheirality,
and reports degeneracy diagnostics.

**Typical input → result:** paired unit bearings → panorama-frame rotation,
unit translation direction, inliers, quality diagnostics, and arbitrary-scale
triangulated points.

**Start here:** {doc}`04_two_view_geometry`.

| Stage | Public entry point | Execution domain | What PanorAi adds |
| --- | --- | --- | --- |
| Correspondence adapter | matched spherical feature rows → bearing pairs | Sphere-native coordinates | Removes all virtual-face pixel frames before geometry |
| Essential pose | relative-pose estimator | Sphere-native bearing geometry | Five-ray hypotheses, tangent-Sampson scoring, refinement, cheirality, quality and degeneracy reporting |
| Triangulation example | tutorial helper, not public API | Sphere-native bearing geometry | Demonstrates the canonical frame, positive depth, and arbitrary-scale convention |

No projected image is required after the frontend has produced unit bearings.
If features came from gnomonic views, the complete workflow is hybrid: view
projection for local evidence, then sphere-native two-view geometry.

(capability-map-multiview)=
## Multiview reconstruction

### What you can do with multiple views

Connect pairwise matches into a view graph and conflict-free tracks, initialize
camera rotations/positions and 3D points, then jointly optimize spherical
reprojection residuals. Failures are represented explicitly instead of being
hidden behind partial maps.

**Typical input → result:** features and matches from 3+ panoramas → optimized
panorama poses, sparse 3D points, observations, and diagnostics at arbitrary
global scale.

**Start here:** {doc}`05_multiview_reconstruction`; see
{doc}`../how_to/spherical_reconstruction` for operational recipes.

| Stage | Public entry point | Execution domain | What PanorAi adds |
| --- | --- | --- | --- |
| Pair graph | reconstruction builder/mapper | Sphere-native bearing geometry | Quality-gated pose edges with accepted/rejected reasons and panorama identity |
| Tracks | multiview track construction | Sphere-native observation graph | Conflict-free tracks with at most one observation per panorama |
| Initialization | reconstruction mapper | Sphere-native bearing geometry | Panorama-frame pose initialization, point triangulation, and explicit gauge conventions |
| Bundle adjustment | spherical reconstruction options/mapper | Sphere-native optimization | Tangent-plane log-map residuals between predicted and measured bearings, filtering, Jacobians, and diagnostics |
| COLMAP alternative | PyCOLMAP export | Projection-domain virtual cameras | PanorAi owns export geometry; PyCOLMAP/COLMAP owns the downstream reconstruction |

The native PanorAi mapper never reconstructs ERP images during optimization.
Its output is geometry. The COLMAP alternative intentionally uses a projected
virtual-camera rig because that is the external tool's camera model.

(capability-map-dense-stereo)=
## Spherical dense stereo

### What you can do with two posed panoramas

Estimate radial range directly on the ERP lattice from two central panoramas
and a metric relative pose. Inverse-range hypotheses follow spherical
epipolar curves; no full gnomonic image set is materialized.

**Typical input → result:** two aligned ERP panoramas plus metric $R,t$ →
radial range, validity, confidence, cost, and hypothesis-index maps.

**Start here:** {doc}`06_spherical_dense_stereo`; see
{doc}`../explanation/spherical_dense_stereo` for the objective and limitations.

| Stage | Public entry point | Execution domain | What PanorAi adds |
| --- | --- | --- | --- |
| Plane sweep | `estimate_spherical_range` | Sphere-native bearing/range geometry | Pose-driven spherical epipolar hypotheses and seam-safe ERP sampling |
| Adaptive range pyramid | `SphericalStereoOptions.pyramid_levels` | Sphere-native coarse-to-fine range search | Broad absolute inverse-range sweep followed by uncertainty-aware normalized local offsets; supplied high-resolution $R,t$ is unchanged |
| Appearance normalization and gradients | `SphericalStereoOptions.filter_backend` | Sphere-native tangent neighbourhoods | Angular box support and east/north derivatives instead of rectangular ERP filters |
| Local cost filtering | dense stereo estimator | Sphere-native spherical convolution | Constant angular support across latitude; C++/NumPy parity through the image-processing contract |
| Path aggregation and consistency | dense stereo estimator | ERP optimization over spherical evidence | Four edge-aware paths, sub-hypothesis refinement, explicit validity and bidirectional checks |

OpenCV supplies the bilinear and nearest remapping primitive. The local filters
use PanorAi spherical convolution directly; they do not project to gnomonic
views and reconstruct an ERP.

(capability-map-slam)=
## SLAM

### What you can do with SLAM

Estimate an online trajectory from a central ERP or calibrated central-fisheye
sequence, select keyframes, create and optimize a local map, relocalize, and
apply loop corrections. PanorAi exposes status and diagnostics alongside the
pose so tracking failures are inspectable.

**Typical input → result:** ordered central-camera frames or caller-provided
spherical feature sets → per-frame pose/status, keyframes, map observations,
loop information, and an arbitrary-scale trajectory.

**Start here:** {doc}`07_spherical_slam`; see {doc}`../how_to/spherical_slam`
for configuration and lifecycle details.

| Workflow | Public entry point | Execution domain | What PanorAi adds |
| --- | --- | --- | --- |
| ERP incremental SLAM | `SphericalIncrementalSLAM.add_frame` | Hybrid: default gnomonic feature frontend, then sphere-native tracking/mapping | Central panorama bearings, immediate status/pose, keyframes, spherical 2D–3D refinement, local BA, relocalization and loop correction |
| Precomputed feature route | `SphericalIncrementalSLAM.add_features` | Sphere-native after caller-owned features | Skips image projection while retaining feature, match, pose, and map contracts |
| Calibrated central-fisheye SLAM | `SphericalVisualSLAM` | Hybrid: OpenCV features on the physical fisheye image, calibrated ray inversion, then sphere-native graph/reconstruction | Equidistant calibration, central-camera bearings, graph diagnostics, and arbitrary-scale trajectory |
| Dual-fisheye input | explicit validation | Not modeled as one central sphere | Rejects the invalid assumption that two physical camera centres are one central camera |

SLAM produces camera and map geometry, not a filtered or reconstructed ERP.
Its spherical specificity comes from calibrated unit bearings and tangent-space
residuals after the image frontend has produced observations.

## C++ acceleration is a separate axis

Sphere-native versus projection-domain describes the mathematics. C++ versus
NumPy/Python/OpenCV describes execution. A native kernel does not turn a
projected algorithm into a spherical one, and a NumPy implementation can still
be fully sphere-native.

| Area | Mathematical route | C++ acceleration | Reference/fallback | Effect on semantics |
| --- | --- | --- | --- | --- |
| Projection | Projection-domain | First-party fused arbitrary-N ERP → gnomonic sampling, selective bilinear cubemap → ERP, and compatible Gaussian view reconstruction | NumPy engine; Torch has its own differentiable route | None: same coordinates, interpolation, masks, and result contract |
| Image processing | Sphere-native | First-party `spherical_filter2d` kernel for compatible `float32`/`float64` NumPy `HW`/`HWC` inputs; linear filters and pyramids can reuse it | NumPy spherical sampler selected with `backend="numpy"` or when native is unavailable | None: same angular taps and output layout |
| Experimental deep learning | Sphere-native learned neighbourhoods | Differentiable Torch spherical `Conv2d`/`ConvTranspose2d` with bounded row chunks | Exact external pretrained parameters; planar/cubemap routes remain separate controls | Porting preserves parameters and output lattice, not perspective-domain calibration or spherical invariance |
| Dense stereo local evidence | Sphere-native | The same first-party spherical-convolution kernel accelerates local normalization, east/north gradients, and batched cost-volume filtering with hypotheses as channels | NumPy route via `filter_backend="numpy"`; OpenCV remains the remapping primitive | None: the range search, angular support, and result contract are unchanged |
| Features/descriptors | Projection-domain or hybrid | OpenCV's C++ implementation provides SIFT/ORB/AKAZE, BF, and FLANN; it is not a PanorAi native kernel | PanorAi orchestrates geometry and spherical metadata | OpenCV owns descriptor/detector semantics; PanorAi owns mapping and provenance |
| Two-view geometry | Sphere-native | First-party five-point polynomial coefficients and tangent-Sampson residual kernels | NumPy path via `compute_backend="numpy"` | None: same estimator policy and result model |
| Multiview reconstruction | Sphere-native | First-party spherical-BA residual and local Jacobian blocks | NumPy/SciPy path via `bundle_compute_backend="numpy"` | None: same tangent residual and optimization contract |
| SLAM | Hybrid | Reuses accelerated frontend, estimator, and bundle-adjustment components when their conditions apply; no separate native SLAM engine | Python orchestration over the same component contracts | None: backend choice does not redefine the SLAM model |

`"auto"` selects an available compatible first-party native path where that
public option exists. Explicit NumPy modes are useful for reference testing;
native and reference paths are expected to satisfy the same public numerical
contract.

## Responsibility and stability boundary

PanorAi owns spherical coordinates, projection, sampling, support/validity,
angular neighbourhoods, view orchestration, unit-bearing conversion,
spherical residuals, provenance, and result objects. OpenCV owns its detector,
descriptor, and matcher implementations. A caller-supplied model owns its
inference semantics. PyCOLMAP/COLMAP owns reconstruction after export.

The geometry and face-based feature core are Stable unless the API reference
says otherwise. Spherical image processing, spherical DoG, relative pose,
deep-learning portability, native multiview reconstruction, multiscale
routing, and SLAM are currently Experimental. Consult
{doc}`../reference/stability` before treating an
Experimental signature as a long-term compatibility contract.
