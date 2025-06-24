#!/bin/bash
# Fail on any error
set -e
# Build documentation with Sphinx in nitpick mode
sphinx-build -n -W -b html docs docs/_build/html
