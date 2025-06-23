import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'PanorAi'
author = 'Robinson Luiz Souza Garcia'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

html_theme = 'sphinx_rtd_theme'

exclude_patterns = []

html_static_path = ['_static']
