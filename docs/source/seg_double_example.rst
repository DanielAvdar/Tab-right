Double Segmentation
==================

This page demonstrates how to use tab-right's double segmentation capabilities to analyze
how pairs of features interact to affect model performance.

Introduction to Double Segmentation
---------------------------------

Double segmentation allows you to discover and visualize how two features jointly affect
model performance. This is especially useful for identifying feature interactions that single
feature analysis might miss.

Basic Usage
----------

.. code-block:: python
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    from tab_right.segmentations.find_seg import FindSegmentationImp
    from tab_right.segmentations.double_seg import DoubleSegmentationImp

    # Ensure data does not contain NaN values
    data = {
        'feature1': [1, 1, 2, 2, 1, 2],
        'feature2': [1, 2, 1, 2, 1, 2],
        'target': [110, 12, 20, 22, 11, 21],
        'prediction': [11, 11, 21, 23, 10, 20]
    }
    df = pd.DataFrame(data).dropna()

    # Define error and score functions
    def absolute_error(y_true, y_pred):
        return abs(y_true - y_pred)

    # Initialize the segmentation finder
    find_seg = FindSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Initialize the double segmentation
    double_seg = DoubleSegmentationImp(
        segmentation_finder=find_seg
    )

    # Find double segmentation with two features
    model = DecisionTreeRegressor(max_depth=2)
    double_seg_results = double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        error_func=absolute_error,
        model=model,
        score_metric=mean_absolute_error
    )

    # Ensure double_seg_results is defined before usage
    if double_seg_results is not None:
        best_segment = double_seg_results.loc[double_seg_results['score'].idxmin()]
        worst_segment = double_seg_results.loc[double_seg_results['score'].idxmax()]

Visualizing Feature Interactions
-----------------------------

The best way to understand feature interactions is through visualization:

.. code-block:: python

    from tab_right.plotting.plot_segmentations import DoubleSegmPlotting

    # Create plotter with lower_is_better set to True for error metrics
    plotter = DoubleSegmPlotting(
        df=double_seg_results,
        metric_name='score',
        lower_is_better=True  # Green for lower errors (better performance)
    )

    # Get the heatmap representation
    heatmap_df = plotter.get_heatmap_df()

    # Create the heatmap visualization
    heatmap_fig = plotter.plotly_heatmap()

    # For metrics like accuracy or R², use lower_is_better=False
    # This shows higher values in green (better) and lower values in red (worse)

Analyzing Interaction Patterns
----------------------------

Double segmentation helps identify specific feature combinations that affect performance:

.. code-block:: python

    # Ensure double_seg_results is defined before usage
    if double_seg_results is not None:
        # Find best and worst performing segments
        best_segment = double_seg_results.loc[double_seg_results['score'].idxmin()]
        worst_segment = double_seg_results.loc[double_seg_results['score'].idxmax()]

        # Analyze performance disparity
        disparity = double_seg_results['score'].max() - double_seg_results['score'].min()

Practical Applications
--------------------

- **Feature Interaction Discovery**: Identify how features jointly influence predictions
- **Model Improvement**: Create interaction features for problematic segments
- **Targeted Optimizations**: Focus improvement efforts on specific feature combinations
- **Data Quality Analysis**: Detect data issues in specific feature intersections
