# Rotation-only stitching promotion evidence

`PanoramaStitcher.reviewed_profile()` freezes the exact central pinhole,
rotation-only candidate described by
`contracts/panorama-stitching-promotion-v1.json`. It does not make the generic
application or its stages Stable.

## 2026-10-19 outcome: NO-GO

The external manifest was frozen before the two `sky4aframe` triplets were
executed. The image bytes remain outside the repository and package.

Command (from the repository root):

```console
python -m benchmarks.panorama_stitching.run_reviewed_external \
  --dataset-root /path/to/HuginSample-at-c26c836 \
  --output /tmp/stitching-external.json
```

Observed with Python 3.14.3 and HuginSample commit
`c26c836220e2d6845e1c9b88cc3c254cf43c088b`:

| Case | Classification | Outcome |
|---|---|---|
| `durlach-370-372` | external development replay | accepted, connected, deterministic |
| `durlach-373-375` | external development replay | accepted, connected, deterministic |
| `durlach-376-378` | external development replay | accepted, connected, deterministic |
| `durlach-379-381` | external development replay | rejected: disconnected image 2 |
| `sky-629-631` | prospective negative | rejected deterministically |
| `sky-693-695` | prospective negative | rejected deterministically |

The accepted real-image cases had median angular residuals from 0.192° to
0.247° and normalized overlap seam errors from 0.023 to 0.050. The negative
cases behaved correctly. The positive gate nevertheless fails because one
fixed positive triplet is disconnected.

The analytic three-view fixture at yaw −32°, 0°, +32° also fails the frozen
default profile's registration gate. Consequently the candidate remains
Experimental. No threshold, case, or profile parameter was changed after the
evidence was opened. A later attempt must use a new profile identity and a new
prospectively frozen cohort.

## Robust rotation v2 development result

``PanoramaStitcher.robust_rotation_profile()`` is a new Experimental profile,
not a revision of reviewed-v1. It retains the primary v1 descriptor and
geometry gates, then selectively retries only weak pairs with the recovery
contract recorded in ``contracts/panorama-stitching-robust-v2.json``.

Run the byte-verified development replay with::

   python -m benchmarks.panorama_stitching.run_reviewed_external \
     --profile robust-v2 \
     --dataset-root /path/to/HuginSample-at-c26c836 \
     --output /tmp/stitching-v2-development.json

On 2026-10-20 all four Durlach triplets were connected deterministically. The
previously disconnected ``durlach-379-381`` case used one recovery edge with
9 calibrated inliers and 0.075° median angular residual. Both frozen
low-texture sky triplets were rejected before recovery. The fixed analytic yaw
triplet also passed with sub-degree rotation error, while the translated-plane
negative remained rejected.

This reuses development imagery and the v1 negative cases. It proves the
regression and failure boundary only; a future promotion requires a new
profile identity, a new scene-disjoint prospective corpus, installed-artifact
evidence and independent review.
