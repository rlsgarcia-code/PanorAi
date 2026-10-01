# PanorAi canonical spherical geometry

This document is the normative geometry contract introduced in PanorAi 3.1.
The public objects and functions in `panorai.geometry` remain compatible
throughout the 3.x series. The packaged machine-readable summary is
`panorai/geometry/geometry-v1.yaml`.

## Coordinate frame and units

- Cartesian axes are `+X` right, `+Y` up, and `+Z` forward.
- Image origin is the top-left. Array indices and public image coordinates
  address pixel centres: `(x, y) = (0, 0)` is the centre of the first pixel.
- Shapes are written `(H, W)`; NumPy image data is `HW` or `HWC` and Torch
  image data is `HW`, `CHW`, or `NCHW`.
- Angles stored by a specification are degrees. Equations on this page use
  radians unless they carry a degree suffix.
- Cartesian vector magnitude and canonical depth are radial range in the
  input distance unit. No API silently interprets radial range as camera
  z-depth.

## Equirectangular panorama (ERP)

For width `W` and height `H`, a pixel coordinate `(x, y)` maps to

```text
lon = 2*pi*(x + 0.5)/W - pi
lat = pi/2 - pi*(y + 0.5)/H
ray = (sin(lon)*cos(lat), sin(lat), cos(lon)*cos(lat))
```

Longitude is normalized to `[-pi, pi)` and the horizontal axis is periodic.
Latitude decreases north-to-south. The four equatorial cardinals are:

```text
periodic seam                  periodic seam
-Z          -X          +Z          +X          -Z
x=-0.5      W/4-0.5     W/2-0.5     3W/4-0.5    W-0.5
```

The north and south poles lie at `y=-0.5` and `y=H-0.5`; they are image
boundaries, not pixel centres. Longitude is mathematically undefined at an
exact pole. `rays_to_erp_pixels()` returns the deterministic representative
`x=W/2-0.5` for the exact vectors `(0,+1,0)` and `(0,-1,0)`. It returns
horizontal coordinates in `[0,W)` and represents the `-Z` seam at `x=W-0.5`.
Any horizontally congruent coordinate is accepted by `erp_pixels_to_rays()`.
Near a pole, where the horizontal components are non-zero, longitude remains
the value given by `atan2(X,Z)`.

Zero-length or non-finite Cartesian inputs have `valid=False` and `NaN` pixel
coordinates. Their reported range is the computed norm, so the three members
of `ERPPointProjection` remain semantically distinct.

## Rectangular gnomonic views

`GnomonicSpec` stores the centre latitude and longitude, independent
horizontal and vertical fields of view, roll, and output shape.

Specification angles must be finite real numbers, not booleans or numeric
strings. Latitude is restricted to `[-90, 90]` degrees. Longitude and roll are
periodic and are stored in the canonical interval `[-180, 180)`; periodically
equivalent inputs therefore create equal immutable specifications. Each field
of view is strictly between `0` and `180` degrees. Output dimensions are
exactly two positive integers and are never truncated from floating values.

Define the unit viewing basis

```text
forward = (sin(lon)*cos(lat),  sin(lat), cos(lon)*cos(lat))
right   = (cos(lon),           0,       -sin(lon))
up      = (-sin(lon)*sin(lat), cos(lat), -cos(lon)*sin(lat))
```

and tangent-plane limits

```text
x_limit = tan(hfov/2)
y_limit = tan(vfov/2)
```

`hfov` and `vfov` are full boundary-to-boundary angular extents through the
view centre. Output pixel centres subdivide the tangent-plane rectangle; the
outermost pixel centres are half a pixel inside its boundaries:

```text
x_j = (2*(j+0.5)/W - 1) * x_limit
y_i = (2*(i+0.5)/H - 1) * y_limit       # positive towards image bottom
```

For `roll=0`, a ray is proportional to `forward + x_j*right - y_i*up`.
Positive roll is a clockwise camera rotation for an observer at the camera
looking along `forward`. Operationally, with `roll=+90 deg`, the top-centre
output sample moves toward `+right`, and the right-centre sample moves toward
`-up`:

```text
plane_x = cos(roll)*x_j - sin(roll)*y_i
plane_y = sin(roll)*x_j + cos(roll)*y_i
ray      = normalize(forward + plane_x*right - plane_y*up)
```

Back-projection includes the closed rectangular boundary (`<=` at each
tangent-plane limit) and requires a positive forward denominator. Samples
outside that region have `support_mask=False`.

### Virtual-camera pixels, rays, and matrices

`gnomonic_pixels_to_rays()` accepts arrays whose last dimension is `(x, y)`
and returns unit rays in the panorama frame plus an explicit validity mask.
The closed raster footprint is `[-0.5, W-0.5] x [-0.5, H-0.5]`.
`rays_to_gnomonic_pixels()` is its inverse for finite, non-zero rays that are
in front of the virtual camera and inside that footprint; it also reports the
original vector ranges. Invalid output coordinates are `NaN`.

The equivalent pinhole intrinsics are

```text
fx = W / (2*tan(hfov/2))       cx = (W-1)/2
fy = H / (2*tan(vfov/2))       cy = (H-1)/2
K  = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

`R_panorama_from_face` has the face-camera right, raster-down, and forward
directions as its columns. Thus
`ray_panorama = normalize(R_panorama_from_face @ [u, v, 1])`, where
`u=(x-cx)/fx` and `v=(y-cy)/fy`. Because the image raster uses y-down while
the panorama frame uses `+Y` up, this direction transform is orthogonal with
determinant `-1`; it is not itself an `SO(3)` pose. The relative transform
between two face cameras is a proper rotation because the two reflections
cancel. Rig exporters must use those relative rotations and retain the
reference-face transform when results are converted back to panorama rays.

`gnomonic_pixel_map()` returns the exact face-to-ERP coordinate map used by
canonical sampling. Projection calls expose the same map through the optional
`ProjectionResult.source_pixels_xy`. `GnomonicFaceGeometry` groups the
resolved specification, `K`, direction transform, support mask, and optional
source map without inferring data validity from geometric support.

## Cubemap

The immutable face order is `front`, `right`, `back`, `left`, `up`, `down`.
Each face basis below is `(forward, right, up)` in the canonical Cartesian
frame. For normalized raster coordinates `u` rightward and `v` downward, a
face ray is `normalize(forward + u*right - v*up)`.

| face | forward | right | up |
|---|---|---|---|
| front | `+Z` | `+X` | `+Y` |
| right | `+X` | `-Z` | `+Y` |
| back | `-Z` | `-X` | `+Y` |
| left | `-X` | `+Z` | `+Y` |
| up | `+Y` | `+X` | `-Z` |
| down | `-Y` | `+X` | `+Z` |

The raster orientations form this net; the table below is authoritative when
an edge is rotated in the net.

```text
              +--------+
              |   up   |
     +--------+--------+--------+--------+
     |  left  | front  | right  |  back  |
     +--------+--------+--------+--------+
              |  down  |
              +--------+
```

Edge coordinates increase top-to-bottom for left/right edges and
left-to-right for top/bottom edges. `same` preserves that direction and
`reversed` flips it.

| face edge | adjacent edge | direction | face edge | adjacent edge | direction |
|---|---|---|---|---|---|
| front left | left right | same | front right | right left | same |
| front top | up bottom | same | front bottom | down top | same |
| right left | front right | same | right right | back left | same |
| right top | up right | reversed | right bottom | down right | same |
| back left | right right | same | back right | left left | same |
| back top | up top | reversed | back bottom | down bottom | reversed |
| left left | back right | same | left right | front left | same |
| left top | up left | same | left bottom | down left | reversed |
| up left | left top | same | up right | right top | reversed |
| up top | back top | reversed | up bottom | front top | same |
| down left | left bottom | reversed | down right | right bottom | same |
| down top | front bottom | same | down bottom | back bottom | reversed |

Cubemap-to-ERP assigns a ray to the face with the largest dot product against
the face forward vector. Scores within eight `float64` machine epsilons of
the maximum are treated as an exact edge or vertex tie, and the first face in
the canonical order wins. This prevents trigonometric round-off from changing
a mathematical tie. The rule is deterministic; it does not promise filtered
continuity across independently supplied face rasters. All six faces are
required and must use one backend and one spatial shape.

## Sampling at boundaries

### Nearest

Nearest sampling uses `floor(c + 0.5)`. An exact half-integer therefore goes
to the more positive integer index. ERP horizontal coordinates wrap modulo
`W`, including after rounding; vertical ERP coordinates clamp to the first or
last row. Gnomonic and cubemap coordinates clamp to their border pixel. Thus
the exact ERP seam coordinate `W-0.5` rounds and wraps to column `0`.

Integer labels and boolean masks require nearest sampling and retain their
exact values. Bilinear interpolation of integer or boolean data raises
`TypeError`.

### Bilinear

Bilinear weights are computed from `floor(c)` and the four neighbouring pixel
centres. ERP neighbours wrap horizontally and clamp vertically. Gnomonic and
cubemap neighbours clamp on both axes, which is border replication rather
than zero padding. At the ERP seam, the last and first columns are neighbours.

Geometry-v1 exposes two named invalid-data policies. `invalid_policy="propagate"`
is the default and uses ordinary IEEE arithmetic: if a weighted stencil
contains `NaN`, the result remains `NaN`, including when that neighbour has
zero mathematical weight. This preserves the strict 3.1 baseline.

`invalid_policy="renormalize"` is opt-in and requires both an explicit boolean
`validity_mask` and an explicit `min_valid_weight` in `(0, 1]`. For bilinear
weights `w_i`, mask values `m_i`, and samples `z_i`, it computes

```text
q = sum_i(w_i * m_i)
y = sum_i(w_i * m_i * z_i) / q
```

Non-finite samples are excluded in addition to the caller's mask. For
multichannel data, a neighbour is valid only when all of its channels are
finite, so validity remains one shared spatial mask. When `q` is below the
caller-selected threshold, the result is `NaN` and invalid. Otherwise the
result is normalized and valid. `ProjectionResult.valid_weight` exposes `q`,
and `ProjectionResult.validity_mask` exposes the thresholded result. Repeated
pixels at clamped borders keep the sum of their bilinear weights. Torch
gradients flow through valid input values only.

The normalized policy is floating-point bilinear only. Labels, boolean masks,
and integer data continue to require nearest sampling. The library does not
choose a universal threshold because acceptable evidence varies by modality.

## Geometric support, data validity, and fill

`ProjectionResult.support_mask` describes only geometric coverage. It does
not mean that a sampled measurement is finite or scientifically valid:

- a supported floating sample may be `NaN`;
- black RGB, zero labels, and zero-valued measurements may be valid;
- a non-zero fill value outside support remains unsupported;
- a separate data-validity mask is authoritative; strict callers transform it
  with nearest as needed, while normalized bilinear callers pass it explicitly.

ERP-to-gnomonic and ERP-to-cubemap outputs are geometrically supported at
every output pixel. Gnomonic-to-ERP exposes the closed rectangular footprint.
A complete cubemap supports the full sphere.

## Dtype, backend, and numerical precision

Sampling preserves the image/tensor dtype, layout, batch, channels, and Torch
device. Coordinate calculations use `float64` in NumPy. Torch uses `float64`
grids for `float64` tensors except on MPS, and `float32` grids otherwise.
Torch CPU bilinear sampling promotes `float16` and `bfloat16` work to
`float32`, then casts the result back. NumPy `float16` bilinear results are
also cast back to `float16`.

The conformance tolerances below compare the same operation and input values
across supported CPU backends; they are not bounds on resampling error or on a
round trip through two projections.

| data / check | absolute tolerance | relative tolerance | reason |
|---|---:|---:|---|
| nearest | exact | exact | index selection and values are discrete |
| `float64` bilinear | `1e-12` | `1e-12` | double grids and arithmetic |
| `float32` bilinear | `7e-6` | `5e-6` | CPU-platform grid/accumulation rounding; measured Linux maximum `6.244e-6` |
| `float16` bilinear | `2e-3` | `2e-3` | quantization after float32 work |
| Torch `bfloat16` bilinear | `1e-2` | `1e-2` | seven-bit mantissa after float32 work |
| analytic cardinal rays (`float64`) | `1e-12` | `0` | closed-form unit directions |

CUDA and MPS smoke checks are conditional and do not define release success.
Autograd is supported for input values, not for geometric parameters.

## Conformance and policy

The executable fixtures in `tests/fixtures/geometry/v1` are the minimum
portable conformance set. Focused contract tests additionally cover the
operational rules above. The fixtures identify their frame, units, dtype,
tolerance, provenance, and derivation; expected values must not be generated
solely by the implementation under test.

The two-mode policy was authorized for 3.1 with strict propagation as the
default. Conformance covers unchanged strict behavior, normalized closed-form
weights, explicit thresholding, boundary rules, and NumPy/Torch parity.
