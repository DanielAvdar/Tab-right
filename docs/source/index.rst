.. python-template documentation master file, created by
   sphinx-quickstart on Wed Apr  9 13:29:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tab-right
========

A Python package for easy analysis of tabular data for inference models (ML and non-ML), focusing on model-agnostic diagnostics using predictions.

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

SegmentationStats
==================

The `SegmentationStats` class provides segmentation statistics for tabular data based on model predictions.

Usage Example
-------------

```python
import pandas as pd
from tab_right.seg import SegmentationStats

data = pd.DataFrame({
    'feature1': [1, 2, 1, 2],
    'label': [0, 1, 1, 0],
    'prediction': [0, 1, 1, 0]
})

stats = SegmentationStats(data, label_col='label', pred_col='prediction')
result = stats.run()
print(result)
```

This will output a DataFrame with segmentation statistics such as count, mean label, mean prediction, std of prediction, and accuracy (if the label is binary).
