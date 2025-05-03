Segmentation Examples
===================

This documentation demonstrates how to use the tree-based segmentation functionality in the tab-right package for error analysis and visualization.

Segmentation Basics
------------------

Tab-right provides powerful tools for segmenting your data based on model errors, allowing you to identify regions in the feature space where your model underperforms. This is particularly useful for:

- Error analysis in machine learning models
- Identifying feature interactions that lead to poor model performance
- Visualizing error patterns across different feature values
- Making targeted improvements to your models



Single and Double Feature Segmentation
-------------------------------------

Tab-right supports both single-feature and two-feature segmentation analysis:

1. **Single Feature Segmentation**: Analyzes how errors vary across different ranges of a single feature
2. **Double Feature Segmentation**: Examines the interaction between two features and their impact on model errors

Visualization Functions
---------------------

The package provides several visualization functions for exploring segmentation results:


Advanced Analysis
---------------

You can extract detailed information about the segments:


Using Lower-Level APIs
--------------------

For more control, you can use the lower-level segmentation APIs:

.. code-block:: python

    from tab_right.segmentations.find_seg import FindSegmentationImp
    from tab_right.segmentations.double_seg import DoubleSegmentationImp
    from sklearn.tree import DecisionTreeRegressor

    # Create custom segmentation
    df = pd.DataFrame({"feature1": X_test["feature1"],
                       "feature2": X_test["feature2"],
                       "label": y_test,
                       "prediction": y_pred})

    # Initialize the segmentation finder
    segmentation_finder = FindSegmentationImp(df, "label", "prediction")

    # Create a single-feature segmentation
    model = DecisionTreeRegressor(max_depth=3)
    single_seg_results = segmentation_finder("feature1",
                                             lambda y, p: np.abs(y - p.iloc[:, 0]),
                                             model)

    # Plot the single feature segmentation
    from tab_right.plotting.plot_segmentations import plot_single_segmentation
    fig_single = plot_single_segmentation(single_seg_results)
    fig_single.show()

    # Create a double-feature segmentation
    double_segmentation = DoubleSegmentationImp(segmentation_finder)
    model = DecisionTreeRegressor(max_depth=3)
    double_seg_results = double_segmentation("feature1",
                                             "feature2",
                                             lambda y, p: np.abs(y - p.iloc[:, 0]),
                                             model)

    # Plot the double feature segmentation
    from tab_right.plotting.plot_segmentations import plot_double_segmentation
    fig_double = plot_double_segmentation(double_seg_results)
    fig_double.show()
