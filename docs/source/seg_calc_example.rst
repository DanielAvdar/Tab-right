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
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from tab_right.segmentations.calc_seg import SegmentationCalc

    # Create a sample DataFrame for demonstration
    data = {
        'segment': ['A', 'A', 'B', 'B', 'C', 'C'],
        'target': [10, 12, 20, 22, 30, 32],
        'prediction': [11, 11, 21, 23, 29, 31]
    }
    df = pd.DataFrame(data)

    # Create a numerical encoding of the segment column
    segment_mapping = {segment: i for i, segment in enumerate(df['segment'].unique())}
    df['segment_id'] = df['segment'].map(segment_mapping)

    # Group your data by the numerical segment ID
    grouped_df = df.groupby('segment_id')

    # Create a reverse mapping for segment names (from ID to original name)
    segment_names = {i: segment for segment, i in segment_mapping.items()}

    # Create a SegmentationCalc instance
    seg_calc = SegmentationCalc(
        gdf=grouped_df,
        label_col='target',
        prediction_col='prediction',
        segment_names=segment_names
    )

.. code-block:: python

    # Create a numerical encoding of the segment column
    segment_mapping = {segment: i for i, segment in enumerate(df['segment'].unique())}
    df['segment_id'] = df['segment'].map(segment_mapping)

    # Group your data by the numerical segment ID
    grouped_df = df.groupby('segment_id')

    # Create a reverse mapping for segment names (from ID to original name)
    segment_names = {i: segment for segment, i in segment_mapping.items()}

    # Create a SegmentationCalc instance
    seg_calc = SegmentationCalc(
        gdf=grouped_df,
        label_col='target',
        prediction_col='prediction',
        segment_names=segment_names
    )

    # Calculate metrics for each segment
    results = seg_calc(mean_absolute_error)
    print(results)

Working with Custom Metrics
--------------------------

You can use any compatible metric function with SegmentationCalc:

.. code-block:: python

    from sklearn.metrics import mean_squared_error, r2_score

    # Using the seg_calc instance created earlier
    mae_results = seg_calc(mean_absolute_error)
    mse_results = seg_calc(mean_squared_error)
    r2_results = seg_calc(r2_score)

Understanding segment_names
--------------------------

The `segment_names` parameter is a dictionary that maps segment IDs (integers) to their original names:

.. code-block:: python

    # For categorical segments
    df['category'] = ['X', 'X', 'Y', 'Y', 'Z', 'Z']

    # Create a numerical encoding
    cat_mapping = {cat: i for i, cat in enumerate(df['category'].unique())}
    df['category_id'] = df['category'].map(cat_mapping)

    # Group by the numerical ID column
    cat_segments = df.groupby('category_id')

    # Create the segment_names mapping (ID -> original name)
    cat_names = {i: name for name, i in cat_mapping.items()}

    # For numerical segments (using pd.cut)
    df['age'] = [25, 35, 45, 55, 65, 75]
    bins = [0, 30, 60, 90]
    df['age_group'] = pd.cut(df['age'], bins)

    # Convert intervals to numerical IDs
    age_intervals = df['age_group'].unique()
    age_mapping = {interval: i for i, interval in enumerate(age_intervals)}
    df['age_group_id'] = df['age_group'].map(age_mapping)

    # Group by numerical ID
    num_segments = df.groupby('age_group_id')

    # Create segment_names mapping (ID -> interval)
    num_names = {i: interval for interval, i in age_mapping.items()}

Key Applications
--------------

- **Model Debugging**: Identify segments where your model underperforms
- **Fairness Assessment**: Evaluate model performance across different demographic groups
- **Data Quality Analysis**: Discover data issues in specific segments
