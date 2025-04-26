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

.. include:: introduction.rst

Tab-right
=========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   examples
   drift_example
   segmentation_example
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
