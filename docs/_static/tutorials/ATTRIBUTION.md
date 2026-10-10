# Tutorial image provenance

The photographic tutorial assets come from the public Poly Haven
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
