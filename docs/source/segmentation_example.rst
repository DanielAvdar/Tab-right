Segmentation Examples
===================

This page provides examples of using tab-right's segmentation features to analyze
and visualize data segmentation.

Single Segmentation
------------------

Single segmentation allows you to analyze how a specific feature impacts model performance.

Basic Single Segmentation Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import plotly.graph_objects as go
    from tab_right.plotting.plot_segmentations import plot_single_segmentation

    # Create a sample DataFrame with segmentation results
    # In a real scenario, this would come from FindSegmentationImp
    segmentation_df = pd.DataFrame({
        "segment_id": [1, 2, 3, 4],
        "segment_name": ["Age <= 30", "30 < Age <= 45", "45 < Age <= 60", "Age > 60"],
        "score": [0.18, 0.25, 0.15, 0.22],
    })

    # Plot the segmentation
    fig = plot_single_segmentation(segmentation_df)
    # fig.show()  # Uncomment to display the plot

Double Segmentation
------------------

Double segmentation lets you analyze how two features together affect model performance
through a heatmap visualization.

Basic Double Segmentation Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import plotly.graph_objects as go
    from tab_right.plotting.plot_segmentations import DoubleSegmPlotting

    # Create a sample DataFrame with double segmentation results
    # In a real scenario, this would come from DoubleSegmentationImp
    double_segmentation_df = pd.DataFrame({
        "segment_id": [1, 2, 3, 4, 5, 6],
        "feature_1": ["Age <= 30", "Age <= 30", "30 < Age <= 60", "30 < Age <= 60", "Age > 60", "Age > 60"],
        "feature_2": ["Income <= 50k", "Income > 50k", "Income <= 50k", "Income > 50k", "Income <= 50k", "Income > 50k"],
        "score": [0.15, 0.25, 0.20, 0.18, 0.22, 0.12],
    })

    # Create the double segmentation plotter
    plotter = DoubleSegmPlotting(df=double_segmentation_df)

    # Get the pivoted DataFrame for heatmap
    heatmap_df = plotter.get_heatmap_df()
    print(heatmap_df)

    # Create the heatmap visualization
    fig = plotter.plotly_heatmap()
    # fig.show()  # Uncomment to display the plot

Practical Application Scenarios
------------------------------

Model Performance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Use segmentation to identify where your model performs poorly across different
feature distributions:

.. code-block:: python

    import pandas as pd

    # Sample segmentation results
    segmentation_df = pd.DataFrame({
        "segment_id": [1, 2, 3, 4],
        "segment_name": ["Feature <= 25", "25 < Feature <= 50", "50 < Feature <= 75", "Feature > 75"],
        "score": [0.12, 0.25, 0.18, 0.31],
    })

    # Analyze model performance across segments
    # High error segments might indicate areas where your model needs improvement
    high_error_segments = segmentation_df[segmentation_df["score"] > segmentation_df["score"].mean()]
    print("Segments with above-average error:")
    print(high_error_segments)

Feature Interaction Investigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use double segmentation to discover how two features interact to affect model performance:

.. code-block:: python

    import pandas as pd
    from tab_right.plotting.plot_segmentations import DoubleSegmPlotting

    # Sample double segmentation results
    double_segmentation_df = pd.DataFrame({
        "segment_id": [1, 2, 3, 4, 5, 6],
        "feature_1": ["Feature1 <= 25", "Feature1 <= 25", "25 < Feature1 <= 75", "25 < Feature1 <= 75", "Feature1 > 75", "Feature1 > 75"],
        "feature_2": ["Feature2 <= 50", "Feature2 > 50", "Feature2 <= 50", "Feature2 > 50", "Feature2 <= 50", "Feature2 > 50"],
        "score": [0.15, 0.25, 0.20, 0.35, 0.10, 0.05],
    })

    # Create plotter
    plotter = DoubleSegmPlotting(df=double_segmentation_df)

    # Get the heatmap to analyze feature interactions
    heatmap_df = plotter.get_heatmap_df()

    # Find the segment combination with highest error
    max_error_segment = heatmap_df.stack().idxmax()
    print(f"Segment combination with highest error: {max_error_segment}")

    # Find the segment combination with lowest error
    min_error_segment = heatmap_df.stack().idxmin()
    print(f"Segment combination with lowest error: {min_error_segment}")
