.. _segmentation_example:

Segmentation Example
====================

This standalone example demonstrates segmentation analysis using random data.

.. code-block:: python

    """Standalone example for segmentation analysis using random data."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from tab_right.segmentations.base import SegmentationStats

    # Generate random classification data
    df = pd.DataFrame({
        'feature': np.random.choice(['X', 'Y', 'Z'], 1000),
        'label': np.random.choice([0, 1], 1000),
        'prediction': np.random.choice([0, 1], 1000)
    })

    # Segmentation analysis by categorical feature
    seg = SegmentationStats(
        df,
        label_col='label',
        pred_col='prediction',
        feature='feature',
        metric=accuracy_score,
        is_categorical=True
    )
    result = seg.run()
    print("Segmentation analysis by categorical feature:")
    print(result)

    # Plot segmentation result (categorical)
    import plotly.express as px
    fig = px.bar(result, x='segment', y='score', title='Segmentation Metric by Category')
    fig.show()

    # Segmentation analysis by continuous feature
    df['cont_feature'] = np.random.normal(0, 1, 1000)
    seg2 = SegmentationStats(
        df,
        label_col='label',
        pred_col='prediction',
        feature='cont_feature',
        metric=accuracy_score,
        is_categorical=False
    )
    result2 = seg2.run(bins=5)
    print("\nSegmentation analysis by continuous feature:")
    print(result2)

    # Plot segmentation result (continuous)
    # Convert Interval objects to strings for plotting
    result2['segment'] = result2['segment'].astype(str)
    fig2 = px.bar(result2, x='segment', y='score', title='Segmentation Metric by Bin')
    fig2.show()

Run this file directly:

.. code-block:: bash

   python segmentation_example.py
