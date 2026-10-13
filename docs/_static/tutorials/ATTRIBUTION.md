# Tutorial image provenance

`spherical-object-hypothesis-method.svg` is an original PanorAi vector diagram
derived only from the documented geometry and object-localization contracts; it
contains no third-party image or generated model output.

The photographic tutorial assets come from the public Poly Haven
HDRIs published under **CC0**.

## Application-first multiscene gallery

`scripts/generate_application_gallery.py` downloads four pinned 2K HDRIs into
an external cache, verifies size and SHA-256 before decoding, applies one fixed
tone map, and produces the five tracked application-gallery JPEGs. Use:

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

`application-semantics-boiler-room.jpg` is a real 512×1024 spherical ResNet18
run, not a segmentation mock-up. It shows ImageNet class `prison` (index 743),
the fixed `grid-like-structure` proxy, a seam-aware CAM component, and a 70°
FOV proposal. The class ranked first with probability 0.1632 in this exact run.
The dense 16×32 CAM is normalized only within the selected class for display
with the blue–yellow–red colorbar. It is directional evidence, not ground truth,
an object mask, or a calibrated pixel probability. The verified external
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

`application-processing-boma.jpg` contains real 512×1024 sphere-native
Gaussian, tangent Scharr, and geodesic Canny results beside the source ERP.

### Modern Buildings

- Asset page: <https://polyhaven.com/a/modern_buildings>
- Author: Greg Zaal
- Source file: `modern_buildings_2k.hdr`
- Source URL: <https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/modern_buildings_2k.hdr>
- Source size: 6,356,062 bytes
- Source MD5 (upstream API): `ff4c694bd73a78671fd9bc8e2b411954`
- Source SHA-256: `bdcd68558e138d5c3071dc0f565ad75b33953f589ff74a21d4ea5d8d9c15d87d`

`application-features-modern-buildings.jpg` contains real Stable-core SIFT +
FLANN results between the 512×1024 source and a documented 20° cyclic longitude
shift. The fixed run retained 408 valid matches and plots the 90 best descriptor
distances, including seam-adjacent evidence.

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

### Tracked derivative checksums

- `application-projection-sundowner.jpg` (1200×600, 130,026 bytes):
  `51f48a1030e6fc9a15b7707ce918d487c2ab9d0fd1f422f3dec965ab3152d6b3`
- `application-processing-boma.jpg` (1200×600, 96,801 bytes):
  `dbf797329a7efc07deed6e4dc1b690c509b0f85f08974c0ec64dec5dbd8b4593`
- `application-features-modern-buildings.jpg` (1200×600, 151,472 bytes):
  `e05152f8aa200893f5c48a1aadc83213604105b5fbd4dc4ac4f76c928794889c`
- `application-semantics-boiler-room.jpg` (1200×600, 94,095 bytes):
  `2771f960c2e6db85cee44c4733fc0bb3dbe3e0cec7421ac7e7046b434e2e94cb`
- `applications-overview.jpg` (1600×900, 151,278 bytes):
  `2c71d3bebc98be48f117b2114bbc4fb61ec251c7d4dac4159cc9e76c69d9fbe0`
- `application-gallery-metadata.json` (16,499 bytes):
  `cbb2134c53a4f38b3afc3c281500d13f27e9fa4ff3e4352d18b1d7f2f0bba40c`

The JSON metadata records the per-application working resolution, exact model
and checkpoint provenance, processing parameters, counts, output dimensions,
and hashes. JPEGs are saved without EXIF metadata. The 16:9 overview uses the
four real cards, the existing analytic stereo fixture, and a deliberately
diagrammatic reconstruction thumbnail; the full `reconstruction-graph.svg`
remains the linked method card.

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
