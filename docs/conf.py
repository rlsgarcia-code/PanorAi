import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'PanorAi'
author = 'Robinson Luiz Souza Garcia'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.spelling",
]

autodoc_mock_imports = [
    "gradio",
    "depth_anything_v2",
    "tensorboard",
    "html4vision",
    "plyfile", "kapture", "zoedepth", "iopath", "quaternion",
    "dust3r",
    "xformers",
    # Additional heavy or optional dependencies mocked for docs build
    "panorai_models",
    "open3d",
    "cv2",
    "torch",
    "torchvision",
    "PIL",
    "numpy",
    "scipy",
    "joblib",
    "skimage",
    "pydantic",
    "yaml",
]

# If you keep seeing “failed to import …”, just append the module name here.

html_theme = "furo"
autosummary_generate = True
autosectionlabel_prefix_document = True
nb_execution_mode = "off"  # start with 'off'; switch to 'auto' later

html_static_path = ['_static']


# Enable nitpicky mode to warn about all missing references
nitpicky = True

# Silence warnings that are noisy in this project
suppress_warnings = []
suppress_warnings += ["autoapi.*", "myst.domains", "deprecated.object"]

# Show "previous" and "next" links at the bottom of each page
html_theme_options = {
    "navigation_with_keys": True,
}

