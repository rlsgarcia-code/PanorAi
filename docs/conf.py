from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "PanorAi"
author = "Robinson Luiz Souza Garcia"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

# Canonical documentation does not import optional research/visualization
# backends merely to render the stable geometry API.
autodoc_mock_imports = ["open3d", "torch", "torchvision"]
autodoc_typehints = "none"
autosectionlabel_prefix_document = True
nb_execution_mode = "off"
nitpicky = True

# The 3.0 repository contained a broad generated autodoc tree whose pages mixed
# stable, compatibility, experimental, and third-party research namespaces.
# PanorAi 3.1 publishes a curated reference instead. The files remain in the
# source tree for compatibility archaeology but are intentionally not part of
# the public documentation build.
exclude_patterns = [
    "generated/*",
    "modules.rst",
    "api_objects.rst",
    "reference/modules.rst",
    "reference/panorai*.rst",
    "reference/tests*",
    "reference/setup.rst",
    "reference/lmdb_report.rst",
    "tutorials/*.ipynb",
    "how_to/attach_blender.rst",
    "how_to/attach_projector.rst",
    "how_to/attach_samplers.rst",
    "how_to/data_containers.rst",
    "how_to/data_factory.rst",
    "how_to/image_processing.rst",
    "how_to/multichannel.rst",
    "how_to/point_cloud.rst",
    "how_to/preprocess_containers.rst",
]

suppress_warnings = []

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {"navigation_with_keys": True}
