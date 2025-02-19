# Panorai

**Panorai** is a Python package for spherical (360°) image processing, specifically focusing on sampling, projection, forward/backward transformations, and unsharp masking for panoramic or spherical data. It provides a modular pipeline to handle various steps, including:

- Resizing input images  
- Sampling tangent points on a sphere (e.g., cube-based, Fibonacci-based)  
- Projecting spherical images into local tangential (rectilinear) planes  
- Re-projecting these rectilinear images back into the equirectangular space  
- Configuring and applying unsharp masking to sharpen the projected images  
- Logging and comparing configuration results (e.g., MSE) via an automated test suite  

This README will guide you through the repository structure, installation, usage, and testing of **Panorai**.

---

## Table of Contents

1. [Overview](#overview)  
2. [Directory Structure](#directory-structure)  
3. [Installation](#installation)  
4. [Examples](#examples)  
5. [Key Modules and Classes](#key-modules-and-classes)  
6. [Running Tests](#running-tests)  
7. [Extending Panorai](#extending-panorai)  
8. [License](#license)

---

## Overview

**Panorai** is a Python library designed for advanced geometric transformations, projections, and sampling on spherical and equirectangular data. It provides a highly modular framework for implementing gnomonic projections, backward projections, and blending strategies, suitable for 360-degree image processing and panoramic data transformations.

---

## Directory Structure

```markdown
/
├── panorai/
│   ├── __pycache__/
│   ├── common/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   └── shared_config.py
│   ├── pipeline/
│   │   ├── __pycache__/
│   │   ├── utils/
│   │   │   ├── __pycache__/
│   │   │   ├── __init__.py
│   │   │   └── resizer.py
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── pipeline_data.py
│   │   └── pipeline_old.py
│   ├── projection/
│   │   ├── __pycache__/
│   │   ├── utils/
│   │   │   ├── __pycache__/
│   │   │   ├── __init__.py
│   │   │   ├── remapper.py
│   │   │   └── unsharp.py
│   │   ├── __init__.py
│   │   ├── projector.py
│   │   └── projector_deprecated.py
│   ├── readme/
│   │   └── gnomonic.md
│   ├── sampler/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   └── sampler.py
│   ├── __init__.py
├── panorai.egg-info/
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── tests/
│   ├── __pycache__/
│   └── test_config_params.py
├── .gitignore
├── best_configs_chart.png
├── requirements.txt
├── setup.py
└── test_results.db
```

### Notable Contents

- **panorai/**  
  Main package directory containing submodules:
  
  - **pipeline/**: The pipeline logic for forward/backward projection and data handling, plus utility functions like `resizer`.  
  - **projection/**: Projector classes (e.g., `GnomonicProjector`) and remapping/unsharp utility code.  
  - **sampler/**: Sphere sampling strategies (cube, icosahedron, Fibonacci).  
  - **readme/**: Additional notes/documentation (e.g., gnomonic.md).  

- **setup.py**  
  A setuptools-based installation script.

- **requirements.txt**  
  Lists dependencies needed to run the code (e.g., NumPy, OpenCV, etc.).

---

## Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/yourusername/panorai.git
   cd panorai
   ```

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Panorai (Editable Mode or Standard):
   ```bash
   pip install -e .
   # or
   python setup.py install
   ```

---

## Examples

### 1. Data Preparation

Start by loading your input data, typically stored in an `.npz` file containing `rgb`, `depth`, and other channels.

```python
import numpy as np
from panorai.pipeline.pipeline_data import PipelineData

# Load data from an NPZ file
filename = 'path/to/sample.npz'  # Replace with your file path
arr = np.load(filename)

rgb = arr['rgb']  # Shape: (H, W, 3)
depth = np.sqrt(np.sum(arr['z']**2, axis=-1))[:, :, None]  # Shape: (H, W, 1)
xyz = arr['z']  # Shape: (H, W, 3)

# Create a PipelineData instance
data = PipelineData.from_dict({
    "rgb": rgb,
    "depth": depth,
    "xyz_depth": xyz
})
```

### 2. Preprocessing Data

Adjust for shadow angle and optionally rotate the equirectangular image:

```python
from panorai.pipeline.utils.preprocess_eq import PreprocessEquirectangularImage

# Visualize the original data
import matplotlib.pyplot as plt
plt.imshow(data.data['rgb'])
plt.show()

# Preprocess the data (e.g., handle shadow angle)
data.preprocess(shadow_angle=30)
plt.imshow(data.data['rgb'])
plt.show()
```

### 3. Using Projections

#### 3.1. Forward Projection

Project equirectangular data into a gnomonic projection.

```python
from panorai.submodules.projections import ProjectionRegistry

# Access the gnomonic projection
proj = ProjectionRegistry.get_projection('gnomonic', return_processor=True)

# Perform forward projection
face = proj.forward(data.data['rgb'])
plt.imshow(face)
plt.show()
```

#### 3.2. Backward Projection

Reconstruct the equirectangular image from a projection:

```python
# Perform backward projection
eq_img = proj.backward(face)
plt.imshow(eq_img)
plt.show()
```

### 4. Using the ProjectionPipeline

The `ProjectionPipeline` provides a high-level API for projections.

```python
from panorai.pipeline.pipeline import ProjectionPipeline

# Initialize the pipeline
pipe = ProjectionPipeline(projection_name='gnomonic')

# Forward projection
face = pipe.project(data)
plt.imshow(face['rgb'].astype(np.uint8))
plt.show()
```

#### 4.1. Using Samplers

Use samplers to generate multiple projections (e.g., cube or icosahedron samplers):

```python
pipe = ProjectionPipeline(projection_name='gnomonic', sampler_name='CubeSampler')
faces = pipe.project(data)

# Visualize a specific face
plt.imshow(faces['point_1']['rgb'].astype(np.uint8))
plt.show()
```

#### 4.2. Blending Projections

Blend multiple projections into a seamless equirectangular image:

```python
# Reconstruct equirectangular image
reconstructed = pipe.backward(faces)
plt.imshow(reconstructed['stacked'])
plt.show()
```

#### 4.3. Custom Configurations

Modify the pipeline configuration to customize the behavior:

```python
pipe = ProjectionPipeline(projection_name='gnomonic', sampler_name='IcosahedronSampler')
faces = pipe.project(data, subdivisions=2, fov_deg=40)
reconstructed = pipe.backward(faces)
plt.imshow(reconstructed['stacked'])
plt.show()
```

---

## Key Modules and Classes

1. **PipelineData**  
   A container for storing and stacking multiple image channels (e.g., `rgb`, `depth`).


2. **ProjectionPipeline**  
   Manages the forward/backward transformations using a chosen sampler and projector.


3. **Samplers**  
   - CubeSampler: Tangent points for cube-based projections.
   - IcosahedronSampler: Icosahedron-based tangent points.
   - FibonacciSampler: Fibonacci sphere sampling for uniform distribution.

4. **ProjectionRegistry**  
   - Project that implements the projection modules
   https://github.com/RobinsonGarcia/projections/tree/main

---

## Extending Panorai

- **Add new samplers**: Implement a new class inheriting from `Sampler` and register it in `SAMPLER_CLASSES`.
- **Add new projectors**: Implement a new class inheriting from `ProjectionStrategy` and add it to `PROJECTOR_CLASSES`.
- **HF integration to handle depth estimation**: Implement seamless integration with HF to load and run DA inference


---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## TODO

List all configuration options:
method: it can select wither ndimage remap or cv2 remap. With cv2 one has more control and can adjust interpolation c, borderMode and borderValue
interpolation: affects the interpolation method during the projection stages
    INTER_NEAREST      = 0,
    INTER_LINEAR       = 1,
    INTER_CUBIC        = 2,
    INTER_AREA         = 3,
    INTER_LANCZOS4     = 4,
    INTER_MAX          = 7,
    WARP_FILL_OUTLIERS = 8,
    WARP_INVERSE_MAP   = 16
   e.g panorai-cli --input someimage.jpeg --sampler_name=CubeSampler --blender_name=ClosestBlender --delta_lon=45 --delta_lat=0 --fov_deg=90 --kwargs method=cv2 interpolation=0
borderMode:
   Python: cv.BORDER_CONSTANT
   iiiiii|abcdefgh|iiiiiii with some specified i
   BORDER_REPLICATE 
   aaaaaa|abcdefgh|hhhhhhh
   BORDER_REFLECT 
   Python: cv.BORDER_REFLECT
   fedcba|abcdefgh|hgfedcb
   BORDER_WRAP 
   Python: cv.BORDER_WRAP
   cdefgh|abcdefgh|abcdefg
   BORDER_REFLECT_101 
   Python: cv.BORDER_REFLECT_101
   gfedcb|abcdefgh|gfedcba
   BORDER_TRANSPARENT 
   Python: cv.BORDER_TRANSPARENT
   uvwxyz|abcdefgh|ijklmno - Treats outliers as transparent.
   BORDER_REFLECT101 
   Python: cv.BORDER_REFLECT101
   same as BORDER_REFLECT_101
   BORDER_DEFAULT 
   Python: cv.BORDER_DEFAULT
   same as BORDER_REFLECT_101
   BORDER_ISOLATED 
   Python: cv.BORDER_ISOLATED
   Interpolation restricted within the ROI boundaries.
   SOURCE: opencv-python
sampler_name: Selects one of the available sampling strategies (as of today: 'CubeSampler', 'IcosahedronSampler', 'FibonacciSampler')
   panorai-cli --list-samplers

blender_name: ['FeatheringBlender', 'GaussianBlender', 'AverageBlender', 'ClosestBlender']
---

## Contact

For questions or feedback, contact the maintainers:

- **Your Name**: rlsgarcia@icloud.com
- **GitHub**: [https://github.com/RobinsonGarcia](https://github.com/RobinsonGarcia)

---

Enjoy using Panorai for your panoramic image processing!


