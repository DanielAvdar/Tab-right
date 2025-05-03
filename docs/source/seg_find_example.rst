Finding Segmentations
===================

This page demonstrates how to use tab-right's feature segmentation capabilities to automatically
discover meaningful data segments that affect model performance.

Introduction to Finding Segmentations
-----------------------------------

The FindSegmentation functionality in tab-right helps you automatically identify
segments in your data based on how they affect model performance. It uses decision trees
to find natural breakpoints in your features that correspond to performance differences.

Basic Usage
----------

.. code-block:: python

    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from tab_right.segmentations.find_seg import FindSegmentationImp

    # Define an error function that calculates error for each datapoint
    def absolute_error(y_true, y_pred):
        return abs(y_true - y_pred)

    # Initialize the segmentation finder
    find_seg = FindSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Find segmentations using a specific feature
    model = DecisionTreeRegressor(max_depth=3)
    segmentation_results = find_seg('feature_name', absolute_error, model)

Understanding the Results
-----------------------

The segmentation results contain segments with different error profiles:

.. code-block:: python

    # View segments ranked by performance
    sorted_segments = segmentation_results.sort_values('score')

    # Each segment has:
    # - segment_id: Unique identifier for the segment
    # - segment_name: Description of the segment condition
    # - score: The average error metric for that segment

Multiple Feature Analysis
-----------------------

You can analyze different features separately to identify which ones create the most impactful segments:

.. code-block:: python

    # Compare segmentation results across multiple features
    features = ['feature_A', 'feature_B', 'feature_C']

    for feature in features:
        segments = find_seg(feature, absolute_error, DecisionTreeRegressor(max_depth=2))
        # Analyze segment disparity to identify influential features

Practical Applications
--------------------

- **Feature Importance**: Discover which features have the most impact on model errors
- **Model Optimization**: Train specialized models for problematic segments
- **Feature Engineering**: Create segment indicator features to improve model performance
