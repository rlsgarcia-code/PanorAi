# P74 W121 preliminary result

Status: **negative development control**. This is not publication evidence.

The frozen CNN prior and both refinements were evaluated at the native audited
ERP lattice, 4128×8256, with radial range capped at 15 m. W119 and W124 were
fixed optimization views. Their RGB data and registered `R,t` were constants;
the only optimized variable was a native per-pixel log-depth residual around
the frozen W121 CNN seed. W122 was opened only after predictions and hashes
were frozen. Ground truth was loaded later and used only for evaluation.

| Variant | AbsRel ↓ | scale-aligned relative 3D RMSE ↓ | mean normal error ↓ | W122 feature cost ↓ |
|---|---:|---:|---:|---:|
| CNN seed | **0.31033** | **0.40093** | **48.20°** | **0.17060** |
| multiview, no CRF | 0.32304 | 0.40774 | 70.65° | 0.17095 |
| multiview + CRF | 0.32304 | 0.40771 | 71.08° | 0.17096 |

The no-CRF result reduced feature cost on its two optimization views from
0.17208 to 0.16449 (W119) and from 0.16402 to 0.15842 (W124), but degraded
AbsRel by 4.10%, the scale-invariant 3D measure by 1.70%, mean normal error by
46.57%, and W122 cost by 0.21%. The current CRF weight made no material
difference. The refined maps visibly contain high-frequency residual noise.

An evaluation-only oracle check placed P74 ground-truth depth into the same
fixed spherical warp. Its feature costs were 0.16614 on W119, 0.15343 on W124,
and 0.16417 on W122, all below the CNN seed. Therefore the registered geometry
and fixed-image objective contain useful depth signal; the present independent
per-pixel optimizer nevertheless finds photometric correspondences that fit
the optimization views better than the correct structure and do not generalize.

This result rejects “more epochs” or “a small CRF term” as the next step. A
follow-up must keep the same frozen-variable contract while adding visibility
and confidence handling, patch/feature matching that rejects repetitive
industrial texture, and a spatially coherent residual representation. W122 is
now a development validation view; a later publication experiment needs a new
untouched test split. The backbone-neutral `DepthPrior` boundary is already
suitable for replacing the CNN seed with a ViT seed.

The external evidence bundle is
`/private/tmp/panorai-val028-multiview-depth-consensus`. It contains the native
NPY and uint16 depth maps, independently verified binary PLY clouds, a
pre-W122 freeze manifest, a comparison panel, and `results.json`.
