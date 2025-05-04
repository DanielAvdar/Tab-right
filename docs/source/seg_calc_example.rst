Segmentation Calculation
=======================

This page explains how to use the segmentation calculation features in tab-right, which
provide tools for analyzing model performance across different data segments.

Introduction to Segmentation Calculation
---------------------------------------

Segmentation calculation allows you to measure how your model performs in different
segments of your data. This is useful for:

- Identifying underperforming segments where your model needs improvement
- Understanding how different segments contribute to overall model error
- Validating the model's performance across various data distributions

Basic Usage
----------

.. code-block:: python
    :hidden:

    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from tab_right.segmentations.calc_seg import SegmentationCalc

    # Create a sample DataFrame for demonstration
    data = {
        'segment': ['A', 'A', 'B', 'B', 'C', 'C'],
        'target': [10, 12, 20, 22, 30, 32],
        'prediction': [11, 11, 21, 23, 29, 31]
    }
    df = pd.DataFrame(data)

    # Group your data by a segment column
    grouped_df = df.groupby('segment')

    # Create a SegmentationCalc instance
    seg_calc = SegmentationCalc(
        gdf=grouped_df,
        label_col='target',
        prediction_col='prediction'
    )

.. code-block:: python

    # Group your data by a segment column
    grouped_df = df.groupby('segment')

    # Create a SegmentationCalc instance
    seg_calc = SegmentationCalc(
        gdf=grouped_df,
        label_col='target',
        prediction_col='prediction'
    )

    # Calculate metrics for each segment
    results = seg_calc(mean_absolute_error)

Working with Custom Metrics
--------------------------

You can use any compatible metric function with SegmentationCalc:

.. code-block:: python

    from sklearn.metrics import mean_squared_error, r2_score

    # Calculate different metrics for comparison
    mae_results = seg_calc(mean_absolute_error)
    mse_results = seg_calc(mean_squared_error)
    r2_results = seg_calc(r2_score)

Key Applications
--------------

- **Model Debugging**: Identify segments where your model underperforms
- **Fairness Assessment**: Evaluate model performance across different demographic groups
- **Data Quality Analysis**: Discover data issues in specific segments
