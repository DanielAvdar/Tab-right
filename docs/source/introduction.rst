.. _introduction

Introduction
============

.. image:: https://img.shields.io/pypi/v/tab-right.svg
   :target: https://pypi.org/project/tab-right/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/tab-right.svg
   :target: https://pypi.org/project/tab-right/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=ubuntu
   :alt: Ubuntu

.. image:: https://img.shields.io/badge/win-blue?logo=windows
   :alt: Windows

.. image:: https://img.shields.io/badge/mac-blue?logo=apple
   :alt: MacOS

.. image:: https://codecov.io/gh/DanielAvdar/tab-right/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/DanielAvdar/tab-right
   :alt: Coverage

.. image:: https://img.shields.io/github/last-commit/DanielAvdar/tab-right/main
   :target: https://github.com/DanielAvdar/tab-right/commits/main
   :alt: Last Commit

.. image:: https://github.com/DanielAvdar/tab-right/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/DanielAvdar/tab-right/actions/workflows/ci.yml
   :alt: Tests

.. image:: https://github.com/DanielAvdar/tab-right/actions/workflows/code-checks.yml/badge.svg
   :target: https://github.com/DanielAvdar/tab-right/actions/workflows/code-checks.yml
   :alt: Code Checks

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

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
