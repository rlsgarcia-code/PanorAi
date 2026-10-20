# Dense Reconstruction promotion evidence

The exact candidate is
`SphericalDenseReconstruction.reviewed_profile(voxel_size=..., bounds=...)`.
It requires at least three supplied metric registered ERPs and freezes reviewed
MVS, uniform dense TSDF, Marching Cubes and range-consistent median color.

## 2026-10-19 outcome: NO-GO

The analytic implementation evidence is green:

- camera/world transforms and every relative pose use independent equations;
- plane, sphere and box range oracles meet median ≤ 1 voxel and P95 ≤ 2 voxels;
- plane normal median error is ≤ 10 degrees;
- seam, poles, differing ERP resolutions, invalid ranges, unknown voxels,
  black-valid color and deterministic replay are covered;
- Application, public Stages, safe serialization and typed cache agree;
- the surface evaluator now has independent parallel-plane distance,
  opposite-normal, externally masked observability and degeneracy oracles.

The scientific Application gate cannot be opened legitimately. Every locally
available Matterport3D and Stanford2D3D scene/output cohort was previously used
for Stereo development or evaluation. Reusing those scenes would not satisfy
the prospectively frozen, scene-disjoint Dense Reconstruction protocol in
`contracts/dense-reconstruction-promotion-v1.json`. There is therefore no
eligible three-scene-per-corpus manifest, no held-out surface result and no
scientific basis for promotion.

The Application, its stages and dense reconstruction result types remain
Experimental. Only the separately bounded mathematical evaluator is Stable as
`panorai-surface-reconstruction-evaluation/v1`. A future attempt must acquire
new licensed scenes, freeze their SHA-256 identities before inference, then run
the exact thresholds without adjustment.
