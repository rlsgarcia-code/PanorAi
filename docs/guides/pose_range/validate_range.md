# Validate and consume radial range

**Level:** Evaluate · **Applies to:** stereo and monocular outputs

Before creating points or surfaces, record and check:

1. **Quantity:** PanorAi expects radial range `rho`, where `X = rho * bearing`;
   pinhole Z-depth is a different value.
2. **Units and scale:** identify the baseline, model calibration or other source
   of metric scale.
3. **Validity:** retain an explicit mask; zero and black RGB are not invalidity
   signals.
4. **Support:** distinguish unobserved directions from rejected predictions.
5. **Consistency:** inspect reprojection, left/right or cross-view agreement,
   confidence and seam behaviour.
6. **Provenance:** retain model/checkpoint identity, pose, sampling, conversion
   and fusion configuration.

Only accepted pixels should become `X = range * ray`. Compare against scanner
range or another independent reference when making accuracy claims.

## Quantitative checks

When reference range is available, report error over the intersection of
reference validity, prediction validity and geometric support. Include robust
statistics and break them down by latitude, seam distance, range band and
surface discontinuity. Coverage must accompany error: rejecting difficult
pixels can improve RMSE without improving the product.

For stereo, also report bidirectional agreement and reprojection residual. For
monocular methods, report scale alignment separately from shape error unless
the checkpoint is explicitly metric. Never fit scale on the evaluation pixels
and then describe the result as an unadjusted metric prediction.

## 3-D sanity check

Compute rays from the same canonical ERP convention and form
`points = range[..., None] * rays`. Verify a small set of known directions and
distances before exporting a cloud. A transposed ERP, axial/radial mix-up or
wrong camera frame can produce a visually plausible but geometrically wrong
surface.

Continue to [multiview reconstruction](../tutorials/05_multiview_reconstruction.md)
or the [3-D reconstruction guide](https://rlsgarcia-code.github.io/PanorAi/guides/how-to/spherical_reconstruction.html).
