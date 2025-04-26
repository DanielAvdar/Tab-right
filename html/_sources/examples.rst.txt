.. _example

Examples
========

Basic Usage
-----------

.. code-block:: python

   import pandas as pd
   from tab_right.segmentations.base import SegmentationStats
   from sklearn.metrics import accuracy_score

   data = pd.DataFrame({
       'feature': ['A', 'A', 'B', 'B'],
       'label': [0, 1, 1, 0],
       'prediction': [0, 1, 1, 0]
   })

   stats = SegmentationStats(
       data,
       label_col='label',
       pred_col='prediction',
       feature='feature',
       metric=accuracy_score,
       is_categorical=True
   )
   result = stats.run()
   print(result)

Advanced Usage
--------------

- For continuous features, set ``is_categorical=False`` and optionally adjust the ``bins`` parameter.
- For multi-class or probabilistic outputs, pass a list of label columns and use probability mode.
- Use any metric function compatible with scikit-learn (e.g., ``accuracy_score``, ``r2_score``).
