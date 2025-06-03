Tab-right
=========

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




Tab-right is a Python package for easy analysis of tabular data for inference models (ML and non-ML), focusing on model-agnostic diagnostics using predictions.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   introduction

.. toctree::
   :maxdepth: 1
   :caption: Guides & Examples

   architecture
   analysis_guides/guides

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Quickstart
----------

Install tab-right using pip:

.. code-block:: bash

   pip install tab-right

Basic Example:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from tab_right.drift.drift_calculator import DriftCalculator
   from tab_right.plotting.drift_plotter import DriftPlotter

   # Create sample datasets
   df_reference = pd.DataFrame({
       'numeric_feature': np.random.normal(0, 1, 100),
       'categorical_feature': np.random.choice(['A', 'B', 'C'], 100)
   })

   df_current = pd.DataFrame({
       'numeric_feature': np.random.normal(0.5, 1.2, 100),
       'categorical_feature': np.random.choice(['A', 'B', 'C'], 100, p=[0.6, 0.3, 0.1])
   })

   # Calculate and visualize drift
   drift_calc = DriftCalculator(df_reference, df_current)
   drift_plotter = DriftPlotter(drift_calc)

   # Get drift metrics and visualize results
   drift_results = drift_calc()
   drift_plot = drift_plotter.plot_multiple()

Advanced Usage
--------------

- **Feature types**: For continuous features, set ``is_categorical=False`` and optionally adjust the ``bins`` parameter.
- **Multi-class outputs**: For multi-class or probabilistic outputs, pass a list of label columns and use probability mode.
- **Custom metrics**: Use any metric function compatible with scikit-learn (e.g., ``accuracy_score``, ``r2_score``).
- **Drift thresholds**: Set custom thresholds for different severity levels with the ``thresholds`` parameter.
- **Visualization options**: Customize plots with matplotlib parameters and custom color schemes.

Contributing
------------
See the CONTRIBUTING.md file for guidelines.

License
-------
MIT License
