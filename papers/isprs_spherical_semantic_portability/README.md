# ISPRS-style spherical semantic portability paper

This directory contains an anonymized full-paper draft evaluating the direct,
training-free port of three frozen perspective-pretrained semantic models to
equirectangular panoramas:

- ImageNet ResNet18;
- Places365 ResNet18;
- OpenCLIP RN50.

The paper uses no Ollama, VLM judge, or generative oracle. Quantitative claims
come from checksum-pinned model outputs, implementation audits, canonical
planar parity, timing, memory, and output geometry. The six displayed CAMs are
explicitly cherry-picked qualitative examples and are never presented as an
unbiased accuracy sample. Stanford semantic localization remains unreported
because the matching official `pano_semantic` files are absent locally.

## ISPRS format

The source targets the official ISPRS full-paper template, version January
2024: <https://www.isprs.org/documents/orangebook/app5.aspx>. The official
template archive is downloaded only during the build and verified as SHA-256
`e187bf5d55ebb44a14c31e6ae4319969168dc0305e93d04e9d657aa72b7d079d`;
its class and bibliography files are not redistributed by this repository.

The review manuscript is anonymized. Replace the author and affiliation fields
only when preparing a camera-ready submission.

## Build

Requirements: `curl`, `unzip`, and Tectonic 0.17 or newer.

```bash
bash papers/isprs_spherical_semantic_portability/build_paper.sh
```

The stable output path is:

`output/pdf/isprs_spherical_semantic_portability.pdf`

## Evidence boundary

`results/pilot_results.json` is the compact, auditable source for every number
in the pilot tables. Model checkpoints and raw result directories remain in
external user caches or temporary experiment storage. The PDF embeds only the
six tracked CC0-derived gallery images already documented in
`docs/_static/tutorials/ATTRIBUTION.md`.
