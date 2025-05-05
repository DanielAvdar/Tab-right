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

Here's a simple example showing how to visualize segmentation results with tab-right:

.. plot::
    :include-source:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_single_segmentation_mp

    # Create a simple results DataFrame directly
    results_df = pd.DataFrame({
        'segment_id': [0, 1, 2],
        'segment_name': ['A', 'B', 'C'],
        'score': [0.5, 1.2, 0.8]
    })

    # Plot using tab-right's visualization function
    fig = plot_single_segmentation_mp(results_df)
    plt.title("Segment Analysis with tab-right")

Working with Multiple Metrics
-----------------------------

Tab-right makes it easy to apply different metrics to your segmented data:

.. plot::
    :include-source:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_single_segmentation_mp

    # Create a DataFrame simulating MAE results across segments
    mae_results = pd.DataFrame({
        'segment_id': [0, 1, 2],
        'segment_name': ['A', 'B', 'C'],
        'score': [0.5, 1.2, 0.8]
    })

    # Plot using tab-right's visualization
    plt.figure(figsize=(8, 5))
    plot_single_segmentation_mp(mae_results)
    plt.title("MAE by Segment (tab-right)")

    # Example showing how you could plot multiple metrics
    # (commented out to focus on one plot for the example)
    '''
    # MSE example would be similar
    mse_results = pd.DataFrame({
        'segment_id': [0, 1, 2],
        'segment_name': ['A', 'B', 'C'],
        'score': [0.25, 1.44, 0.64]  # Squared values of MAE
    })

    plt.figure(figsize=(8, 5))
    plot_single_segmentation_mp(mse_results)
    plt.title("MSE by Segment (tab-right)")

    # R² example
    r2_results = pd.DataFrame({
        'segment_id': [0, 1, 2],
        'segment_name': ['A', 'B', 'C'],
        'score': [0.92, 0.86, 0.90]
    })

    plt.figure(figsize=(8, 5))
    plot_single_segmentation_mp(r2_results)
    plt.title("R² by Segment (tab-right)")
    '''

Segmentation with Numerical Features
-------------------------------------

Tab-right also works with numerical features by automatically binning them:

.. plot::
    :include-source:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_single_segmentation_mp

    # Create a results DataFrame simulating age group segmentation results
    age_plot_df = pd.DataFrame({
        'segment_id': [0, 1, 2, 3],
        'segment_name': ['(20, 35]', '(35, 50]', '(50, 65]', '(65, 80]'],
        'score': [6.2, 7.5, 8.9, 10.1]  # MAE values increasing with age
    })

    # Use tab-right's built-in visualization
    plt.figure(figsize=(8, 5))
    age_fig = plot_single_segmentation_mp(age_plot_df)
    plt.title('Mean Absolute Error by Age Group')

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
