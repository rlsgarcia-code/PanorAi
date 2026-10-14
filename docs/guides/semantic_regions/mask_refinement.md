# Refine an accepted region with local masks

**Level:** Optional extension · **Strategy:** Projection-domain segmentation

Segmentation is not required to create a semantic or spatial hypothesis. When
a tighter outline is useful, open a transient gnomonic view centred on the
accepted angular proposal and prompt a local segmenter.

Retain every candidate mask needed for audit. If a mask reaches a face edge,
continue over overlapping neighbouring views, transfer the prior and fuse the
accepted masks on the sphere. Release each RGB face and model embedding after
evaluation; persist compact spherical masks and provenance rather than a cache
of projected images.

Mask interpolation must be nearest-neighbour. Cross-face fusion must use
explicit support and solid-angle overlap; a visually smooth overlay is not an
independent correctness oracle.

## Local refinement sequence

1. Centre a gnomonic face on the accepted proposal and choose a FOV that
   contains its uncertainty envelope.
2. Generate prompts from the proposal interior and exterior; keep the local
   model's alternative masks and quality scores.
3. Backproject masks with nearest-neighbour sampling and clip them by face and
   source support.
4. If an accepted mask reaches a face border, open an overlapping face in that
   direction, transfer the prior and continue with a cycle/step limit.
5. Merge consistent masks with solid-angle IoU or containment, then compact
   the spherical masks and release transient RGB faces and embeddings.

Preserving alternatives is important: a locally plausible mask can represent
one pipe, an entire pipe bank or background between them. A single attractive
choice hides that uncertainty.

Segmentation refines spatial extent; it does not retroactively validate the
semantic proxy or cross-view identity. Those remain separate evidence fields.

The Experimental facade and its limits are documented in the
[CAM and segmentation tutorial](../tutorials/09_spherical_fcn_cam.md).
Next: [evaluate the result](evaluation.md).
