# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.abspath("../../"))
# sys.path.insert(0, os.path.abspath("./"))  # in conf.py


project = "tab-right"
# tab-right is a Python package for easy analysis of tabular data for inference models (ML and non-ML),
# focusing on model-agnostic diagnostics using predictions.

try:
    version = version("tab-right")
except:
    version = "0.1.0"  # Default version if package is not installed
release = version

copyright = "2025, DanielAvdar"  # noqa
author = "DanielAvdar"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core extension for pulling docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.githubpages",  # If deploying to GitHub Pages
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinxmermaid",  # Support for Mermaid diagrams
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
master_doc = "index"

plotly_output_format = "html"
