Finding Segmentations
===================

This page demonstrates how to use tab-right's feature segmentation capabilities to automatically
discover meaningful data segments that affect model performance.

Introduction to Finding Segmentations
-----------------------------------

The FindSegmentation functionality in tab-right helps you automatically identify
segments in your data based on how they affect model performance. It uses decision trees
to find natural breakpoints in your features that correspond to performance differences.

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from tab_right.segmentations.find_seg import FindSegmentationImp

    # Create sample data
    np.random.seed(42)
    data = {
        'feature_A': [1, 1, 2, 2, 3, 3],
        'feature_B': [20, 21, 22, 23, 24, 25],
        'feature_C': [30, 31, 32, 33, 34, 35],
        'target': [100, 105, 110, 115, 120, 125],
        'prediction': [101, 104, 111, 114, 121, 124]
    }
    df = pd.DataFrame(data)

    # Define an error function
    def absolute_error(y_true, y_pred):
        return abs(y_true - y_pred)

    # Initialize the segmentation finder
    finder = FindSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Find segmentations using a specific feature
    model = DecisionTreeRegressor(max_depth=3)
    segments = finder('feature_A', absolute_error, model)
    print("Found segments:", len(segments))

Basic Usage
----------

The basic workflow involves:

1. Preparing your data
2. Defining an error metric
3. Creating a segmentation finder
4. Analyzing specific features

Here's a complete example:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from tab_right.segmentations.find_seg import FindSegmentationImp

    # Create sample data with some variation in errors
    np.random.seed(42)
    n_samples = 100
    data = {
        'feature_A': np.random.normal(10, 2, n_samples),
        'target': np.random.normal(100, 10, n_samples)
    }
    # Add predictions with varying error rates
    data['prediction'] = data['target'] + np.random.normal(0, abs(data['feature_A'] - 10), n_samples)

    df = pd.DataFrame(data)

    # Define error function
    def absolute_error(y_true, y_pred):
        return abs(y_true - y_pred)

    # Initialize segmentation finder
    finder = FindSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Find segmentations
    model = DecisionTreeRegressor(max_depth=2)
    segments = finder('feature_A', absolute_error, model)

    # Print results
    if len(segments) > 0:
        print(f"Found {len(segments)} segments")
        print("\nSegment details:")
        print(segments)

Understanding the Results
-----------------------

The segmentation results contain segments with different error profiles:

.. code-block:: python

    # Continuing from previous example
    if len(segments) > 0:
        # Sort segments by score (average error)
        sorted_segments = segments.sort_values('score')

        # Print summary
        print("\nSegments summary:")
        for _, segment in sorted_segments.iterrows():
            print(f"Segment {segment['segment_id']}: {segment['segment_name']}")
            print(f"Average error: {segment['score']:.2f}")

Advanced Usage: Multiple Feature Analysis
---------------------------------------

You can analyze different features separately to identify which ones create the most impactful segments:

.. code-block:: python

    # Initialize with multiple features
    data = {
        'feature_A': np.random.normal(10, 2, n_samples),
        'feature_B': np.random.normal(20, 3, n_samples),
        'feature_C': np.random.normal(30, 4, n_samples),
        'target': np.random.normal(100, 10, n_samples)
    }
    data['prediction'] = data['target'] + np.random.normal(0, 5, n_samples)
    df = pd.DataFrame(data)

    # Initialize finder
    finder = FindSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Compare segmentation results across multiple features
    features = ['feature_A', 'feature_B', 'feature_C']
    results = {}

    for feature in features:
        segments = finder(feature, absolute_error, DecisionTreeRegressor(max_depth=2))
        if len(segments) > 0:
            results[feature] = segments

    # Compare results
    if results:
        print("\nFeature comparison:")
        for feature, segments in results.items():
            print(f"\n{feature}:")
            print(f"Number of segments: {len(segments)}")
            if len(segments) > 0:
                print("Score range:", segments['score'].min(), "-", segments['score'].max())

Practical Applications
--------------------

- **Feature Importance**: Discover which features have the most impact on model errors
- **Model Optimization**: Train specialized models for problematic segments
- **Feature Engineering**: Create segment indicator features to improve model performance
- **Error Analysis**: Identify segments with consistently high or low prediction errors
