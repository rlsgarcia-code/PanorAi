# Interpret a spherical CAM

**Level:** Understand · **Input:** class-score lattice and explicit ERP support

A CAM is directional evidence: high values indicate directions that supported
one classifier label. It is not an instance mask, a calibrated pixel
probability or proof that the named object occupies every highlighted pixel.

Keep these distinctions explicit:

- output stride controls CAM density; fully convolutional does not mean one
  score per input pixel;
- resizing a CAM is visualization, not added semantic resolution;
- ImageNet labels may act as fixed proxies for visually related industrial
  structures;
- zero CAM is valid evidence; invalid or unobserved directions require a
  separate support mask;
- changing ERP resolution changes angular tap spacing and receptive field.

Inspect the exact label, score, output lattice, solid-angle concentration,
support and model provenance. Compare a spherical run with a controlled planar
or tangent-view oracle before claiming equivalence.

## Resolution and receptive field

For a stride-32 FCN, a $H\times W$ ERP produces approximately
$\lceil H/32\rceil\times\lceil W/32\rceil$ logits, subject to the architecture's
padding rules. Every cell has a large receptive field. It says which broad
directions influenced the class—not that a stride-sized image patch is an
object box. Local-window aggregation can create denser evidence only when it
is a declared model operation; interpolation cannot.

## Interpret proxies conservatively

An ImageNet class can be a reproducible visual proxy for an industrial concept
without being its literal identity. Keep both names: for example, the output
hypothesis may say `grid-like structure`, supported by exact classifier label
`prison`. If several proxies form one concept, freeze their non-negative
weights before evaluation. Insufficient evidence yields unknown; it does not
authorize a more convenient industrial label.

## Useful summaries

Report peak direction, solid-angle-weighted mass inside a candidate region,
concentration relative to the supported sphere, stability under longitude
rotation and enrichment over unrelated control classes. A red overlay alone
does not expose whether the maximum moved, whether support changed, or whether
the class was globally relevant.

See the [CAM tutorial](../../tutorials/09_spherical_fcn_cam.md) for controlled
scenarios. Next: [form angular proposals](angular_proposals.md).
