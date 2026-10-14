# Estimate dense range with spherical stereo

**Level:** Start/compose · **Input:** two ERPs and known metric pose ·
**Output:** radial range, validity and confidence

```python
from panorai.stereo import estimate_spherical_range

result = estimate_spherical_range(
    erp_a,
    erp_b,
    R_b_from_a,
    t_b_from_a_m,
    options=options,
)
range_m = result.range
valid = result.validity_mask
```

Stereo does not estimate or refine pose. The unit of `result.range` is inherited
from `t_b_from_a_m`; a unit baseline produces baseline-normalized range.
Rejected pixels remain invalid and must not be silently replaced by zero.

The implementation searches inverse range along the spherical epipolar curve,
supports coarse-to-fine refinement and reports confidence and consistency
diagnostics.

## Preconditions

Both arrays must describe central ERPs on the same canonical lattice. Pose uses
`X_b = R_b_from_a @ X_a + t_b_from_a`. The baseline must be nonzero and its
unit becomes the range unit. Exposure changes, moving objects, low texture and
occlusion are photometric limitations, not geometric support.

## What is optimized

For a reference bearing $b_a$ and radial candidate $\rho$, the 3-D point is
$X_a=\rho b_a$. It is transferred into camera B and normalized to a target
bearing. The solver compares local spherical evidence along that epipolar
curve, searches inverse range coarse-to-fine, and can enforce bidirectional
consistency.

`result.validity_mask` marks accepted numerical estimates. It is distinct from
the geometric support of the panoramas. Confidence helps rank accepted pixels
but is not a calibrated probability unless a separate calibration establishes
that claim.

## Diagnose before tuning

- Broad range bands usually indicate weak texture or insufficient baseline.
- Discontinuity bleeding suggests the window or regularization is too broad.
- Left/right inconsistency identifies occlusion or a wrong pose.
- A global scale error points to the baseline, not the stereo optimizer.
- A seam discontinuity is a regression because neighbourhoods are spherical.

Start with the [dense stereo tutorial](../../tutorials/06_spherical_dense_stereo.md),
then use the [geometry and optimization derivation](../../explanation/spherical_dense_stereo.md)
for the objective and limitations. Finish by
[validating the range](validate_range.md).
