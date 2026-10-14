# Tutorial image provenance

`spherical-object-hypothesis-method.svg` and
`application-strategy-routes.svg` are original PanorAi vector diagrams derived
only from the documented geometry, execution-strategy, and object-localization
contracts. They contain no third-party image or generated model output.

The photographic tutorial assets come from the public Poly Haven
HDRIs published under **CC0**.

## Application-first multiscene gallery

`scripts/generate_application_gallery.py` downloads four pinned 2K HDRIs into
an external cache, verifies size and SHA-256 before decoding, applies one fixed
tone map, and produces the six tracked application-gallery JPEGs. Use:

```bash
python scripts/generate_application_gallery.py \
  --cache-dir /path/to/external/poly-haven-cache \
  --output-dir docs/_static/tutorials
```

Pass `--offline` to require an already verified cache. Original HDR/EXR files,
cache contents, model checkpoints, and intermediate faces are not tracked or
distributed. All four assets were retrieved on 2026-10-12 from Poly Haven and
are published under [CC0](https://polyhaven.com/license).

### Boiler Room

- Asset page: <https://polyhaven.com/a/boiler_room>
- Author: Oliksiy Yakovlyev
- Source file: `boiler_room_2k.hdr`
- Source URL: <https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/boiler_room_2k.hdr>
- Source size: 6,707,438 bytes
- Source MD5 (upstream API): `324d15c18b121a23fda28e35ba280072`
- Source SHA-256: `07c353cb275f1e359cdfa1710acb77c5cd075ce7770859ca56641993263d2d19`

The left side of `application-semantics-boiler-room.jpg` is a real 512×1024
spherical ResNet18 run, not a segmentation mock-up. It shows ImageNet class
`prison` (index 743), the fixed `grid-like-structure` proxy, and its
blue–yellow–red CAM. The class ranked first with probability 0.1920 in this
exact run. The dense 16×32 CAM is normalized only within the selected class
for display. The right side is an original **method diagram** showing how two
registered spherical FOV proposals may support one shared hypothesis; it is
not a measured second-view association. The CAM is directional evidence, not
ground truth, an object mask, or a calibrated pixel probability. The verified external
Torchvision checkpoint is `ResNet18_Weights.IMAGENET1K_V1`, SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

### Boma

- Asset page: <https://polyhaven.com/a/boma>
- Authors: Greg Zaal (photography), Jarod Guest (processing)
- Source file: `boma_2k.hdr`
- Source URL: <https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/boma_2k.hdr>
- Source size: 6,954,266 bytes
- Source MD5 (upstream API): `385d91640b007c7cacbc334a03549232`
- Source SHA-256: `9d2e230d5fdd89efd15b6081a3951c90c771687edd81bc1e08cbb66adb98dd8c`

`application-processing-boma.jpg` contains two real 512×1024 sphere-native
outputs: constant-angle Gaussian smoothing and geodesic Canny over the source
ERP. A 0.60 display gamma makes the dark interior legible; the Canny mask is
dilated by three pixels for card-scale display only. The undilated edge count
is retained in metadata.

### Mosaic Tunnel

- Asset page: <https://polyhaven.com/a/mosaic_tunnel>
- Author: Greg Zaal
- Source file: `mosaic_tunnel_2k.hdr`
- Source URL: <https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/mosaic_tunnel_2k.hdr>
- Source size: 6,625,466 bytes
- Source MD5 (upstream API): `a8847cae4e2418f774d5fd6b2cbb4d6b`
- Source SHA-256: `47f9267f0620158ccdb7fa7bdf25fd3aa4cd9de3fb88a9bfea68b056c739d6d4`

`application-features-mosaic-tunnel.jpg` contains real Stable-core SIFT +
FLANN results between the 512×1024 source and a documented 119.8828° cyclic
longitude shift. The fixed run retained 262 valid matches and plots the 90
best descriptor distances; ten plotted matches cross the ERP seam under the
known rotation.

### Sundowner Overlook

- Asset page: <https://polyhaven.com/a/sundowner_overlook>
- Author: Dario Barresi
- Source file: `sundowner_overlook_2k.hdr`
- Source URL: <https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/sundowner_overlook_2k.hdr>
- Source size: 6,872,718 bytes
- Source MD5 (upstream API): `3ad4d3f49c8751272678a508a2ba77d4`
- Source SHA-256: `3cdcb8a8f066e02d5554ce2d6a2591c9a47e02959297f4b2649fd91a49014498`

`application-projection-sundowner.jpg` contains the real 512×1024 source with
six 95° cube-face footprints, the six 256×256 projected faces, Gaussian-blended
ERP reconstruction, round-trip error, and explicit support. The mean supported
RGB round-trip error is 2.9485 on the uint8 scale; support covers the full ERP.

`pinhole-workflow-sundowner.jpg` is a second real run over the same verified
source. From left to right it contains the source ERP, twelve 160×160
icosahedron-sampled gnomonic views at 82° FOV, those views after OpenCV
`GaussianBlur` with a 15×15 kernel and sigma 3.0, and the Gaussian-blended ERP
reconstruction. The final support fraction is 1.0. Intermediate faces are
generated transiently and are not distributed separately.

`application-pose-range.jpg` combines an original PanorAi method diagram of
two registered spherical cameras and their bearing geometry with the estimated
radial-range and relative-error panels from the existing analytic Poly Haven
Studio fixture. The diagram is not presented as measured reconstruction data.

### Tracked derivative checksums

- `application-projection-sundowner.jpg` (1200×600, 130,026 bytes):
  `51f48a1030e6fc9a15b7707ce918d487c2ab9d0fd1f422f3dec965ab3152d6b3`
- `application-processing-boma.jpg` (1200×600, 67,662 bytes):
  `b79152afec7776b58cd04c2f9b616c7113333078eb47429c7389b7dc117675b9`
- `application-features-mosaic-tunnel.jpg` (1200×600, 148,113 bytes):
  `54132e985d473013bfdf03a7767e4ca3aadc6981f83039091886e6a143642f2e`
- `application-pose-range.jpg` (1200×600, 95,842 bytes):
  `7a491dc660b359df2f47611411113149aaafeebab91aa652058692909f8803e9`
- `application-semantics-boiler-room.jpg` (1200×600, 58,605 bytes):
  `25c949dda66f0a9043ab6cbb80a72308eadd4a8cea82718fee17d6b62ab4b5be`
- `pinhole-workflow-sundowner.jpg` (1200×300, 40,542 bytes):
  `dee93ce269ba3f95717cd50a31078f9cc0b7a32bedb38b3b15ac0b75b2834a4d`
- `application-strategy-routes.svg` (1600×900 view box, 7,675 bytes):
  `07e47e13b03a05b5cabbbafdc79e453dec9255181bc5f61edde39fd57d1bd48a`
- `application-gallery-metadata.json` (17,924 bytes):
  `550d3a92d9cf3cbdac841870e8da158c1efc81ec0ce25ce4560db3d01ee14d88`

The JSON metadata records the per-application working resolution, exact model
and checkpoint provenance, processing parameters, counts, output dimensions,
and hashes. JPEGs are saved without EXIF metadata. The 16:9 strategy diagram
is geometry-led rather than a collage: it distinguishes projection-domain,
sphere-native, and hybrid/multiview execution at normal README scale.

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

`scripts/generate_spherical_stereo_docs_assets.py` verifies the source SHA-256
and applies a fixed tone map. The real panorama is mapped onto an analytic
sphere; a deterministic known camera displacement creates the second view and
exact radial-range reference. The panel illustrates the stereo implementation
without claiming that the analytic geometry was captured in the real office.
