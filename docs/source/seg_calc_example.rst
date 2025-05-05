Segmentation Calculation
========================

.. _seg_calc_example:

This page demonstrates how to use tab-right's segmentation calculation features to analyze model performance across different data segments.

What is Segmentation Analysis?
------------------------------

Segmentation analysis is a powerful technique for understanding how your model performs across different subsets of your data. Tab-right provides comprehensive tools to:

1. Calculate metrics for each segment of your data
2. Visualize segment performance with built-in plotting functions
3. Compare multiple segments to identify potential model weaknesses

Using tab-right for Segmentation Analysis
-----------------------------------------

Tab-right provides the following key components for segmentation analysis:

1. ``SegmentationCalc`` - Core class for calculating metrics across segments
2. ``plot_single_segmentation`` / ``plot_single_segmentation_mp`` - Visualize segment metrics
3. ``DoubleSegmPlotting`` / ``DoubleSegmPlottingMp`` - Visualize interactions between two segment features

Basic Segmentation with tab-right
---------------------------------

Here's a complete example of how to use tab-right's segmentation analysis tools:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from tab_right.segmentations.calc_seg import SegmentationCalc
    from tab_right.plotting import plot_single_segmentation_mp

    # Create a sample DataFrame with segments
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

    # Create a SegmentationCalc instance - tab-right's core segmentation class
    seg_calc = SegmentationCalc(
        gdf=grouped_df,
        label_col='target',
        prediction_col='prediction',
        segment_names=segment_names
    )

    # Calculate metrics for each segment using tab-right
    results = seg_calc(mean_absolute_error)

    # Convert to dataframe for plotting with tab-right
    results_df = pd.DataFrame({
        'segment_id': list(range(len(segment_names))),
        'segment_name': [segment_names[i] for i in range(len(segment_names))],
        'score': [results[i] for i in range(len(segment_names))]
    })

    # Use tab-right's built-in visualization function
    fig = plot_single_segmentation_mp(results_df)
    plt.title("Segment Analysis with tab-right")
    plt.show()

Working with Multiple Metrics
-----------------------------

Tab-right makes it easy to apply different metrics to your segmented data:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tab_right.segmentations.calc_seg import SegmentationCalc
    from tab_right.plotting import plot_single_segmentation_mp

    # Using the same setup from the previous example
    # We can apply multiple metrics using tab-right's SegmentationCalc

    # Create a simple function to calculate and plot a metric using tab-right
    def analyze_with_tab_right(seg_calc, metric_func, title):
        # Calculate metrics for each segment using tab-right
        results = seg_calc(metric_func)

        # Convert to the format needed for tab-right's plotting
        results_df = pd.DataFrame({
            'segment_id': list(range(len(segment_names))),
            'segment_name': [segment_names[i] for i in range(len(segment_names))],
            'score': [results[i] for i in range(len(segment_names))]
        })

        # Create a figure
        plt.figure(figsize=(8, 5))

        # Use tab-right's visualization function
        fig = plot_single_segmentation_mp(results_df)
        plt.title(title)
        plt.show()

    # Apply different metrics with tab-right
    analyze_with_tab_right(seg_calc, mean_absolute_error, "MAE by Segment (tab-right)")
    analyze_with_tab_right(seg_calc, mean_squared_error, "MSE by Segment (tab-right)")
    analyze_with_tab_right(seg_calc, r2_score, "R² by Segment (tab-right)")

Segmentation with Numerical Features
-------------------------------------

Tab-right also works with numerical features by automatically binning them:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    from tab_right.segmentations.calc_seg import SegmentationCalc
    from tab_right.plotting import plot_single_segmentation_mp

    # Create sample data with a numerical feature
    np.random.seed(42)
    n = 100
    numerical_data = {
        'age': np.random.randint(20, 80, n),
        'target': np.random.normal(50, 10, n)
    }
    df_num = pd.DataFrame(numerical_data)

    # Add predictions with some error that varies by age group
    df_num['prediction'] = df_num['target'] + np.random.normal(0, 5 + 0.1 * (df_num['age'] - 20), n)

    # Create age groups using pandas cut
    bins = [20, 35, 50, 65, 80]
    df_num['age_group'] = pd.cut(df_num['age'], bins)

    # Convert to numerical IDs for tab-right
    age_groups = df_num['age_group'].unique()
    age_mapping = {group: i for i, group in enumerate(age_groups)}
    df_num['age_group_id'] = df_num['age_group'].map(age_mapping)

    # Group by age group ID
    age_grouped = df_num.groupby('age_group_id')

    # Create mapping from ID to interval name
    age_names = {i: str(group) for i, group in enumerate(age_groups)}

    # Use tab-right's SegmentationCalc
    age_seg_calc = SegmentationCalc(
        gdf=age_grouped,
        label_col='target',
        prediction_col='prediction',
        segment_names=age_names
    )

    # Calculate metrics with tab-right
    age_results = age_seg_calc(mean_absolute_error)

    # Prepare data for tab-right's visualization
    age_plot_df = pd.DataFrame({
        'segment_id': list(range(len(age_names))),
        'segment_name': [age_names[i] for i in range(len(age_names))],
        'score': [age_results[i] for i in range(len(age_names))]
    })

    # Use tab-right's built-in visualization
    plt.figure(figsize=(8, 5))
    age_fig = plot_single_segmentation_mp(age_plot_df)
    plt.title('Mean Absolute Error by Age Group')
    plt.show()

Interactive Visualization with Plotly
--------------------------------------

Tab-right also provides Plotly-based interactive visualizations:

.. code-block:: python

    from tab_right.plotting import plot_single_segmentation

    # Using the data prepared in the previous examples
    interactive_fig = plot_single_segmentation(age_plot_df)
    interactive_fig.show()

    # For the original segmentation example
    interactive_seg_fig = plot_single_segmentation(results_df)
    interactive_seg_fig.show()

Key Benefits of Using tab-right for Segmentation
------------------------------------------------

- **Standardized API**: Consistent interface for all segmentation analyses
- **Automatic handling of missing values**: Robust processing of incomplete data
- **Support for multiple metrics**: Easy comparison across various evaluation metrics
- **Flexible visualization options**: Both static and interactive plotting
- **Compatible with scikit-learn**: Works with any scikit-learn compatible metric function

Tab-right's segmentation functionality helps you understand where your model performs well and where it needs improvement, enabling targeted model enhancements and better decision-making.
