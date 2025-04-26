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
