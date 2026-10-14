# Associate proposals across registered views

**Level:** Extend · **Strategy:** Hybrid learned evidence and spherical
geometry

Two high CAM regions are not automatically the same object. Association uses
registered geometry and appearance to test that identity:

```text
proposal in A
  -> transfer its angular region into B using the registered cameras
  -> compare with proposals in B
  -> test regional overlap, tracks, local features or embeddings
  -> reject, merge, or preserve ambiguity
  -> produce a bearing-only or spatial hypothesis
```

Useful evidence includes transferred-region IoU, centre-bearing stability,
gnomonic coverage, accepted feature tracks, embedding similarity, number of
supporting views and pose uncertainty. Sparse points are used only when they
survived geometric refinement; otherwise the hypothesis must represent pose
or bearing uncertainty rather than invent a precise 3-D point.

## Geometry sources

`SphericalObjectHypothesisMethod.run()` accepts exactly one geometry source:
either a refined `SphericalReconstructionResult`, or registered camera poses
plus bearing tracks. This prevents a mixture of inconsistent camera states.
When pose is known but spatial depth is unsupported, the valid result is a
bearing-only proposal.

For a metric point $X$ and camera centre $C_v$, the predicted view bearing is

$$\hat b_v = R_v(X-C_v)/\|R_v(X-C_v)\|.$$

Its angular residual is compared with the observed bearing or proposal region.
Only tracks that passed geometric refinement should constrain $X$.

## Association evidence

No single cue is sufficient. Evaluate at least:

- symmetric solid-angle IoU after transferring region A into view B;
- overlap of accepted track IDs and their reprojection residuals;
- similarity of a gnomonic feature or embedding extracted with the same FOV;
- agreement of proposal scale and centre under the registered pose;
- the number of independent supporting views and the uncertainty of each pose.

Conflicting cues should preserve separate or ambiguous hypotheses. Greedy
fusion based only on CAM peak distance is especially unsafe in repeated pipe,
panel or shelf structures.

## Output hypothesis

The output retains concept, aggregate score, supporting tracks and views,
per-view centre/FOV regions, optional spatial location, provenance and state
(`proposed`, `ambiguous` or `confirmed`). The result object also preserves
region, view-pair and transfer evaluations plus reduction cues, so users can
see why a proposal was retained or suppressed.

See the [object-hypothesis tutorial](../../tutorials/11_cam_guided_object_hypotheses.md)
for the method, configuration and stage visualizations. Next:
[refine the mask](mask_refinement.md) when a tighter outline is actually
required; otherwise continue to [evaluation](evaluation.md).
