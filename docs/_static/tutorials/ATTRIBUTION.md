# Tutorial image provenance

The panorama and the four derived feature/image-processing figures in this directory come from
the public, non-industrial **Nature Reserve Forest** HDRI, photographed by
Dimitrios Savva and processed by Jarod Guest, and published by Poly Haven
under **CC0**.

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
