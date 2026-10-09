# Tutorial image provenance

The photographic tutorial assets come from public, non-industrial Poly Haven
HDRIs published under **CC0**.

## Nature Reserve Forest

`nature-reserve-forest-erp.jpg` and the five derived feature/image-processing
figures in this directory come from **Nature Reserve Forest**, photographed by
Dimitrios Savva and processed by Jarod Guest.

- Asset page: <https://polyhaven.com/a/nature_reserve_forest>
- License: <https://polyhaven.com/license>
- Source file: `nature_reserve_forest_1k.hdr`
- Source URL: <https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/nature_reserve_forest_1k.hdr>
- Source size: 1,900,770 bytes
- Source MD5: `93c0154883ffed7ff5300692fc7d959d`
- Source SHA-256: `6c943ddd683de2f3d9aaa62596961dfccdc9cf206adebfc198e70235ae5707cd`
- Retrieved: 2026-10-04

`scripts/generate_documentation_figures.py` verifies the SHA-256 checksum,
applies a fixed tone map, and generates the tracked JPEGs.
`spherical-image-processing.jpg` contains real spherical Gaussian, Scharr, and
Canny outputs. `spherical-histogram-equalization.jpg` contains the real
solid-angle equalization, pixel histograms, and latitude weights.
`feature-detectors.jpg` shows real PanorAi/OpenCV extractions.
`feature-matches.jpg` compares the panorama with a documented cyclic longitude
shift of the same CC0 image so readers can see matching across an ERP seam
without claiming an independent real capture.
`spherical-dog-sift.jpg` shows real direct spherical DoG detections; its
descriptors are computed by OpenCV SIFT on one tangent patch per keypoint.

The three spherical semantic overlays also derive exclusively from the same
tracked CC0 ERP:

- `spherical-cam-imagenet-lakeside.jpg`: ImageNet-1K ResNet18 class
  `lakeside` (index 975, rank 3);
- `spherical-cam-places365-forest.jpg`: Places365 ResNet18 scene
  `forest/broadleaf` (index 150, rank 3);
- `spherical-cam-openclip-path.jpg`: OpenCLIP RN50 prompt
  `a photo of a path` (rank 2 among six declared prompts).

All runs used the source `512×1024` ERP and produced native `16×32` dense
lattices. The experiment runner min-max-normalized and interpolated each
selected channel for display; ImageMagick then converted the generated PNG to
metadata-free JPEG at quality 88 without changing its dimensions. No model
checkpoint bytes are present in these images. Their SHA-256 checksums are:

- `dd0827f6e6cd5301d5ffe14d086cded0ff7ce02e8955d7a95df9b7cd5c65f43c`;
- `bcfb1df74b4f57cfacab38d8695015f4eb3ed70d16734b2afe8311ff5d9e9861`;
- `baef726a3330669c2d84e711efb8b96e714890d44771f38a14e6d643eff241df`.

These visualizations are qualitative evidence only, not semantic masks or
pixel probabilities. The model checkpoints remain external under their
upstream terms.

## Poly Haven Studio

`poly-haven-studio-erp.jpg` and `spherical-stereo-synthetic.png` derive from
**Poly Haven Studio**, photographed by Greg Zaal and published under CC0.

- Asset page: <https://polyhaven.com/a/poly_haven_studio>
- License: <https://polyhaven.com/license>
- Source file: `poly_haven_studio_1k.hdr`
- Source URL: <https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/poly_haven_studio_1k.hdr>
- Source size: 1,682,195 bytes
- Source MD5: `a1065f613cb6e0388d82a99dcee23d3b`
- Source SHA-256: `dfc8505761018d644997a803f332339328af2b53e82023fb3bec37cefdf43b83`
- Retrieved: 2026-10-04

Three additional spherical semantic overlays derive exclusively from the
tracked `poly-haven-studio-erp.jpg` (SHA-256
`9b6e2b7521e2cf35d998f4087840c4bf98b480467bd753d719533228c984be8a`):

- `spherical-cam-imagenet-studio-desk.jpg`: ImageNet-1K ResNet18 class `desk`
  (index 526, requested channel, rank 95);
- `spherical-cam-places365-studio-lobby.jpg`: Places365 ResNet18 scene `lobby`
  (index 217, rank 1);
- `spherical-cam-openclip-studio-desk.jpg`: OpenCLIP RN50 prompt
  `a photo of a desk` (rank 3 among seven declared prompts).

All three runs used the source `512×1024` ERP and produced native `16×32`
dense lattices. The same display-only normalization, interpolation, and
metadata-free JPEG conversion described above were applied. Their SHA-256
checksums are:

- `ff86186906074ac0235aadfb64b3b5c3654064cc9dc42aacd18fd02b8154163a`;
- `556b3ed4313eb648e2c5e19fe94351b8c4818ae3692778fcd3f17bd2943eecf1`;
- `0eb3d03f247c45973f7ff99999347d69adf89aa073c7ec6dc1f4942492bd4a61`.

All six gallery channels are intentionally cherry-picked for visually
informative qualitative presentation. They are not an unbiased sample and
must not be interpreted as accuracy evidence. In particular, the selected
ImageNet `desk` channel is not presented as a top prediction: it was chosen for
its spatial response despite the low rank.

`scripts/generate_spherical_stereo_docs_assets.py` verifies the source SHA-256
and applies a fixed tone map. The real panorama is mapped onto an analytic
sphere; a deterministic known camera displacement creates the second view and
exact radial-range reference. The panel illustrates the stereo implementation
without claiming that the analytic geometry was captured in the real office.
