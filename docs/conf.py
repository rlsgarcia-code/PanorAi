from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "PanorAi"
author = "Robinson Luiz Souza Garcia"

extensions = [
    "api_inventory",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

# Newer Sphinx releases split a few nested PEP 585 annotations at commas while
# rendering dataclass fields, producing truncated cross-reference targets.
# Ignore only those malformed generated fragments; real API references remain
# subject to the strict nitpicky build.
nitpick_ignore_regex = [
    ("py:class", r"(?:tuple|Mapping)\[str.*"),
    ("py:class", r"Literal\['angular'.*"),
]

# Canonical documentation does not import optional research/visualization
# backends merely to render the stable geometry API.
autodoc_mock_imports = ["open3d", "torch", "torchvision"]
autodoc_typehints = "none"
autosectionlabel_prefix_document = True
nb_execution_mode = "off"
nitpicky = True

# Documentation is curated. Generated output and executable notebook sources
# are not part of the strict public documentation build.
exclude_patterns = [
    "_build",
    "generated/*",
    "tutorials/*.ipynb",
]

suppress_warnings = []

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {"navigation_with_keys": True}
