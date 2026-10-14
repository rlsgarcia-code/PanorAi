# Return the right spherical output

**Level:** Understand and extend · **Strategy:** Projection-domain or hybrid

Only raster-shaped outputs should be reconstructed as rasters.

| Face output | Return path | Spherical reduction |
| --- | --- | --- |
| RGB, logits, CAM, confidence, depth | `map` + `reconstruct` | support-aware blend |
| labels or masks | `map` + `reconstruct` | nearest + closest |
| keypoints | `map_points` | bearing lift + angular NMS |
| boxes, polygons or masks | `map_regions` | solid-angle IoU NMS/merge |
| class scores or embeddings | `map_views` | coverage-aware aggregation |

Geometric support, numerical validity and pixel value remain separate. Zero is
valid evidence unless an explicit mask rejects it.

## Dense fields

Register a nonstandard raster with a `RasterFieldSpec`. This prevents a CAM,
label map or range image from silently inheriting RGB interpolation:

```python
from panorai.data import RasterFieldSpec

cam_spec = RasterFieldSpec(kind="continuous", units="score")
with_cam = panorama.with_field("cam", cam_hw, spec=cam_spec, valid=cam_valid)
```

Continuous multichannel fields can represent logits, confidence or dense
embeddings. Categorical fields accept nearest sampling and discrete fusion.
Vector quantities whose components change under projection—such as optical
flow or surface normals—need an explicit basis-transport contract and are
rejected by this generic path.

## Points, regions and whole-view evidence

Use `map_points()` when the face method returns pixel coordinates and aligned
scores. The result keeps face pixels, ERP pixels, unit bearings, validity and
provenance; `angular_nms()` removes overlap duplicates by direction.

Use `map_regions()` for boxes, polygons or masks. Boxes and polygons become
spherical footprints; masks are backprojected with nearest sampling. Region
NMS and merge use solid-angle-weighted overlap rather than ordinary ERP area.

Use `map_views()` when one inference describes the entire face—for example a
classifier score or image embedding. Each observation retains its effective
footprint. `coverage_mean` first normalizes directions seen by several faces,
then integrates them with solid angle; this prevents overlap from counting as
extra evidence.

## Acceptance checks

- Every output carries an explicit `source_id` and face provenance.
- Unsupported observations remain auditable but do not enter fusion.
- Masks and categories use nearest-neighbour sampling.
- Score ties and NMS order are deterministic.
- A zero-valued supported output stays valid.

Continue with the [spherical-observation reference](https://rlsgarcia-code.github.io/PanorAi/reference/spherical_observations.html)
for the object contracts or the
[custom-pipeline tutorial](../../tutorials/01_custom_pipeline.md) for complete
callback examples.
