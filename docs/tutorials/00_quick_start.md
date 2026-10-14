# Quick start

```console
pip install panorai
```

## Choose what you want to build

`````{grid} 1 1 2 2
:gutter: 2

````{grid-item-card} Project and process panoramas
:link: ../applications/projection_processing/index
:link-type: doc

```{image} ../_static/tutorials/application-projection-sundowner.jpg
:alt: Perspective sampling and reconstruction of a spherical panorama
```

Sample perspective views, run pinhole methods, and reconstruct spherical
outputs.
````

````{grid-item-card} Extract and match features
:link: ../applications/features_matching/index
:link-type: doc

```{image} ../_static/tutorials/application-features-mosaic-tunnel.jpg
:alt: Visual features matched across a spherical panorama
```

Detect, lift, deduplicate, and match visual observations on the sphere.
````

````{grid-item-card} Estimate pose and radial range
:link: ../applications/pose_range/index
:link-type: doc

```{image} ../_static/tutorials/application-pose-range.jpg
:alt: Spherical two-view geometry and radial range estimation
```

Estimate relative pose, stereo range, or monocular radial range.
````

````{grid-item-card} Localize semantic regions
:link: ../applications/semantic_regions/index
:link-type: doc

```{image} ../_static/tutorials/application-semantics-boiler-room.jpg
:alt: Semantic evidence and region association in an industrial panorama
```

Generate spherical CAM evidence, region proposals, and multiview associations.
````

`````

## Run your first projection

```{literalinclude} ../../scripts/run_documentation_examples.py
:language: python
:dedent: 4
:start-after: DOCS_FUNCTIONAL_START = None
:end-before: DOCS_FUNCTIONAL_END = None
```

## Continue

- {doc}`../applications/index` — application guides
- {doc}`02_projection_foundations` — projection, sampling, and fusion
- {doc}`spherical_image_processing` — operations designed for the sphere
- {doc}`12_composing_spherical_models` — NumPy/C++ and differentiable layers
- {ref}`capability-map-quick-start` — complete capability map
