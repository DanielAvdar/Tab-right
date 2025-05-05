Welcome

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

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=windows
   :alt: Windows

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=apple
   :alt: MacOS

.. image:: https://codecov.io/gh/DanielAvdar/tab-right/branch/main/graph/badge.svg
   :alt: Coverage

.. image:: https://img.shields.io/github/last-commit/DanielAvdar/tab-right/main
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


Tab-right
=========

Tab-right is a Python package for easy analysis of tabular data for inference models (ML and non-ML), focusing on model-agnostic diagnostics using predictions.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction

.. toctree::
   :maxdepth: 1
   :caption: Guides & Examples

   drift_example
   seg_calc_example
   seg_double_example

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Quickstart
----------

.. code-block:: bash

   pip install tab-right

Advanced Usage
--------------

- For continuous features, set ``is_categorical=False`` and optionally adjust the ``bins`` parameter.
- For multi-class or probabilistic outputs, pass a list of label columns and use probability mode.
- Use any metric function compatible with scikit-learn (e.g., ``accuracy_score``, ``r2_score``).

Contributing
------------
See the CONTRIBUTING.md file for guidelines.

License
-------
MIT License
