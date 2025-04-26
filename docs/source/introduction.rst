.. _introduction

Introduction
============

Tab-right is a Python package for easy, model-agnostic analysis of tabular data for inference models (ML and non-ML). It focuses on diagnostics and interpretability using only model predictions and features—no need for model internals.

Key Features
------------
- Segment tabular data by categorical or continuous features
- Compute custom metrics for each segment (classification, regression, etc.)
- Vectorized, efficient implementation for large datasets
- Probability mode for multi-class or probabilistic outputs
- Simple API for integration into any workflow

Installation
------------

Install the latest version from PyPI:

.. code-block:: bash

   pip install tab-right

For development or the latest features, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/DanielAvdar/tab-right.git
   cd tab-right
   pip install -e .
