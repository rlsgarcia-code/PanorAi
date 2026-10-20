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
python benchmarks/panorama_stitching/run_reviewed_external.py \
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
