Segmentation Calculation
=======================

.. _seg_calc_example:

This page explains how to use the segmentation calculation features in tab-right, which
provide tools for analyzing model performance across different data segments.

Introduction to Segmentation Calculation
---------------------------------------

Segmentation calculation allows you to measure how your model performs in different
segments of your data. This is useful for:

- Identifying underperforming segments where your model needs improvement
- Understanding how different segments contribute to overall model error
- Validating the model's performance across various data distributions

Visualization Options
--------------------

Tab-right provides two plotting backends for segmentation visualizations:

1. **Matplotlib**: Static plots using the ``_mp`` suffix functions (e.g., ``plot_single_segmentation_mp``)
2. **Plotly**: Interactive plots using the standard function names (e.g., ``plot_single_segmentation``)

Basic Usage with Matplotlib
--------------------------

.. plot::
    :context: close-figs

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from tab_right.segmentations.calc_seg import SegmentationCalc
    from tab_right.plotting import plot_single_segmentation_mp

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

    # Calculate metrics for each segment
    results = seg_calc(mean_absolute_error)

    # Convert results to DataFrame for plotting
    results_df = pd.DataFrame({
        'segment_id': range(len(segment_names)),
        'segment_name': [segment_names[i] for i in range(len(segment_names))],
        'score': [results[i] for i in range(len(segment_names))]
    })

    # Plot the results using tab_right's built-in function
    fig = plot_single_segmentation_mp(results_df)
    plt.show()

Basic Usage with Plotly (Interactive)
-----------------------------------

You can also create interactive visualizations using Plotly:

.. code-block:: python

    from tab_right.plotting import plot_single_segmentation

    # Using the same results_df from above
    fig = plot_single_segmentation(results_df)
    fig.show()

Working with Custom Metrics
--------------------------

You can use any compatible metric function with SegmentationCalc:

.. plot::
    :context: close-figs

    from sklearn.metrics import mean_squared_error, r2_score

    # Using the seg_calc instance created earlier
    metrics_funcs = [mean_absolute_error, mean_squared_error, r2_score]
    metric_names = ['MAE', 'MSE', 'R²']

    results_dict = {}
    for metric_func in metrics_funcs:
        results_dict[metric_func.__name__] = seg_calc(metric_func)

    # Create a plot comparing metrics across segments
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, (metric_name, func_name) in enumerate(zip(metric_names, [func.__name__ for func in metrics_funcs])):
        metric_values = [results_dict[func_name][j] for j in range(len(segments))]
        axs[i].bar(segments, metric_values, color='lightgreen')
        axs[i].set_title(f'{metric_name} by Segment')
        axs[i].set_xlabel('Segment')
        axs[i].set_ylabel(metric_name)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

Understanding segment_names
--------------------------

The `segment_names` parameter is a dictionary that maps segment IDs (integers) to their original names:

.. plot::
    :context: close-figs

    # For categorical segments
    df['category'] = ['X', 'X', 'Y', 'Y', 'Z', 'Z']

    # Create a numerical encoding
    cat_mapping = {cat: i for i, cat in enumerate(df['category'].unique())}
    df['category_id'] = df['category'].map(cat_mapping)

    # Group by the numerical ID column
    cat_segments = df.groupby('category_id')

    # Create the segment_names mapping (ID -> original name)
    cat_names = {i: name for name, i in cat_mapping.items()}

    # Visualize the mapping
    plt.figure(figsize=(8, 5))
    categories = list(cat_names.values())
    cat_ids = list(cat_names.keys())

    plt.scatter(cat_ids, [1]*len(cat_ids), s=100, c=cat_ids, cmap='viridis')
    for i, cat in enumerate(categories):
        plt.annotate(cat, (cat_ids[i], 1), ha='center', va='bottom', fontsize=12)

    plt.yticks([])
    plt.xlabel('Category ID')
    plt.title('Mapping of Category IDs to Original Names')
    plt.tight_layout()
    plt.show()

    # For numerical segments (using pd.cut)
    df['age'] = [25, 35, 45, 55, 65, 75]
    bins = [0, 30, 60, 90]
    df['age_group'] = pd.cut(df['age'], bins)

    # Visualize the binning
    plt.figure(figsize=(10, 6))
    plt.scatter(df['age'], [1]*len(df['age']), s=100, c='blue', label='Age values')

    for b in bins:
        plt.axvline(x=b, color='red', linestyle='--')

    plt.yticks([])
    plt.xlabel('Age')
    plt.title('Age Binning with pd.cut')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

Key Applications
--------------

- **Model Debugging**: Identify segments where your model underperforms
- **Fairness Assessment**: Evaluate model performance across different demographic groups
- **Data Quality Analysis**: Discover data issues in specific segments
