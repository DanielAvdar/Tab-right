.. _seg_double_example:

Double Segmentation
==================

This example demonstrates how to use double segmentation in tab-right for analyzing model performance across combinations of two features in tabular data.

Visualization Options
--------------------

Tab-right provides two plotting backends for double segmentation visualizations:

1. **Matplotlib**: Static plots using ``DoubleSegmPlotting_mp`` class
2. **Plotly**: Interactive plots using ``DoubleSegmPlotting`` class

What is Double Segmentation?
---------------------------

Double segmentation allows you to analyze how your model performs across different combinations of two features. This is especially useful for:

- Identifying feature interactions that affect model performance
- Finding specific combinations of feature values where your model underperforms
- Understanding complex patterns in your data that single-feature analysis might miss

Basic Usage with Matplotlib
-------------------------

.. plot::
    :context: close-figs

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from tab_right.segmentations.double_seg import DoubleSegmentationImp
    from tab_right.plotting import DoubleSegmPlotting_mp

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Generate features with correlations
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = 0.5 * feature1 + 0.5 * np.random.normal(0, 1, n_samples)

    # Generate target with complex interaction
    target = 2 + 3 * feature1 + 2 * feature2 + 4 * (feature1 * feature2) + np.random.normal(0, 1, n_samples)

    # Generate prediction (biased in some regions)
    prediction = 2 + 3 * feature1 + 2 * feature2 + np.random.normal(0, 2, n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'target': target,
        'prediction': prediction
    })

    # Create DoubleSegmentationImp instance
    double_seg = DoubleSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply double segmentation
    result_df = double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        score_metric=mean_squared_error,
        bins_1=4,
        bins_2=4
    )

    # Visualize results with matplotlib using tab_right's built-in class
    double_plot = DoubleSegmPlotting_mp(df=result_df)
    fig = double_plot.plot_heatmap()
    plt.show()

Using Plotly for Interactive Visualizations
-----------------------------------------

For interactive heatmaps that allow zooming, hovering for details, and more, you can use Plotly:

.. code-block:: python

    from tab_right.plotting import DoubleSegmPlotting

    # Using the result_df from the previous example
    double_plot = DoubleSegmPlotting(df=result_df)
    fig = double_plot.plot_heatmap()
    fig.show()

Implementation Details
--------------------

The ``DoubleSegmentationImp`` class works by:

1. Binning each feature into a specified number of segments (using equal-width or equal-frequency binning for numeric features)
2. Creating all possible combinations of segments from both features
3. Evaluating model performance (using the provided metric) for each segment combination
4. Returning a DataFrame with segment information and performance metrics

Class Parameters
--------------

.. code-block:: python

    DoubleSegmentationImp(
        df: pd.DataFrame,
        label_col: str,
        prediction_col: Union[str, List[str]]
    )

- **df**: DataFrame containing features, true labels, and predictions
- **label_col**: Name of the column with true labels
- **prediction_col**: Name(s) of column(s) with predictions

Method Parameters
--------------

.. code-block:: python

    double_seg(
        feature1_col: str,
        feature2_col: str,
        score_metric: Callable,
        bins_1: int = 4,
        bins_2: int = 4
    )

- **feature1_col**: Name of the first feature column
- **feature2_col**: Name of the second feature column
- **score_metric**: Function to evaluate performance (e.g., mean_squared_error, accuracy_score)
- **bins_1**: Number of bins for the first feature (default: 4)
- **bins_2**: Number of bins for the second feature (default: 4)

Advanced Example: Finding Performance Issues
------------------------------------------

Double segmentation is particularly useful for finding specific combinations of feature values where your model underperforms:

.. code-block:: python

    # Calculate average performance overall
    overall_mse = mean_squared_error(df['target'], df['prediction'])

    # Find segments with MSE > 2x overall MSE
    problem_segments = result_df[result_df['metric'] > 2 * overall_mse]

    print(f"Overall MSE: {overall_mse:.2f}")
    print(f"Problematic segments:")
    for _, row in problem_segments.iterrows():
        print(f"  Feature1: {row['feature1_segment_name']}, "
              f"Feature2: {row['feature2_segment_name']}, "
              f"MSE: {row['metric']:.2f}")
