.. _seg_double_example:

Double Segmentation
===================

This page demonstrates how to use tab-right's double segmentation functionality to analyze model performance across combinations of two features.

What is Double Segmentation?
----------------------------

Double segmentation allows you to analyze how your model performs across different combinations of two features. This is useful for:

- Identifying feature interactions affecting model performance
- Finding specific feature value combinations where your model underperforms
- Understanding complex patterns single-feature analysis might miss

Tab-right's Double Segmentation Tools
-------------------------------------

Tab-right provides these tools for double segmentation analysis:

1. ``DoubleSegmentationImp`` - Main class for performing double segmentation
2. ``DoubleSegmPlotting`` - Interactive Plotly-based visualization
3. ``DoubleSegmPlottingMp`` - Matplotlib-based visualization

Basic Usage
-----------

Here's a simple example of double segmentation with tab-right:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from tab_right.segmentations import DoubleSegmentationImp
    from tab_right.plotting import DoubleSegmPlottingMp

    # Create sample data
    np.random.seed(42)
    n_samples = 200

    # Generate features and target
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)

    # Target with interaction effect
    target = 2 + feature1 + feature2 + 2 * (feature1 * feature2) + np.random.normal(0, 1, n_samples)

    # Prediction missing the interaction term
    prediction = 2 + feature1 + feature2 + np.random.normal(0, 1, n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'target': target,
        'prediction': prediction
    })

    # Perform double segmentation
    double_seg = DoubleSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply segmentation with 3 bins for each feature
    result_df = double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        score_metric=mean_squared_error,
        bins_1=3,
        bins_2=3
    )

    # Visualize results with a heatmap
    plotter = DoubleSegmPlottingMp(df=result_df)
    fig = plotter.plot_heatmap()
    plt.title("MSE by Feature1 and Feature2 Segments")

Interactive Visualization with Plotly
-------------------------------------

Tab-right also offers interactive Plotly visualization:

.. code-block:: python

    from tab_right.plotting import DoubleSegmPlotting

    # Create interactive visualization from the results
    interactive_plot = DoubleSegmPlotting(df=result_df)
    fig = interactive_plot.plot_heatmap()
    fig.update_layout(title="Interactive Double Segmentation Heatmap")
    fig.show()

Using Different Metrics
-----------------------

You can use any metric compatible with scikit-learn:

.. code-block:: python

    from sklearn.metrics import mean_absolute_error, r2_score

    # Using MAE instead of MSE
    mae_results = double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        score_metric=mean_absolute_error,
        bins_1=3,
        bins_2=3
    )

    # For metrics where higher is better (like R²)
    r2_results = double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        score_metric=r2_score,
        bins_1=3,
        bins_2=3
    )

    # Visualize with appropriate settings
    r2_plotter = DoubleSegmPlottingMp(df=r2_results, lower_is_better=False)
    r2_plotter.plot_heatmap()
    plt.title("R² Score by Feature Segments")

Working with Categorical Features
---------------------------------

Double segmentation works with categorical features without needing to specify bins:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from tab_right.segmentations import DoubleSegmentationImp
    from tab_right.plotting import DoubleSegmPlottingMp

    # Create sample categorical data
    np.random.seed(42)
    n = 300

    # Generate categorical features
    category1 = np.random.choice(['A', 'B', 'C'], n)
    category2 = np.random.choice(['X', 'Y', 'Z'], n)

    # Generate target with different patterns for combinations
    target = np.zeros(n, dtype=int)

    # Add different effects for different combinations
    target[(category1 == 'A') & (category2 == 'X')] = 1
    target[(category1 == 'B') & (category2 == 'Y')] = 1
    target[(category1 == 'C') & (category2 == 'Z')] = 1

    # Simple prediction without capturing all patterns
    prediction = np.zeros(n, dtype=int)
    prediction[category1 == 'A'] = 1
    prediction[category2 == 'Z'] = 1

    # Create DataFrame
    cat_df = pd.DataFrame({
        'category1': category1,
        'category2': category2,
        'target': target,
        'prediction': prediction
    })

    # Perform double segmentation
    cat_seg = DoubleSegmentationImp(
        df=cat_df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply segmentation (no bins needed for categorical features)
    cat_results = cat_seg(
        feature1_col='category1',
        feature2_col='category2',
        score_metric=accuracy_score
    )

    # Plot with higher is better for accuracy
    cat_plot = DoubleSegmPlottingMp(
        df=cat_results,
        lower_is_better=False
    )
    fig = cat_plot.plot_heatmap()
    plt.title("Accuracy by Category Segments")

Key Features of Double Segmentation
-----------------------------------

- **Discover interactions**: Find how combinations of features affect performance
- **Automatic handling**: Works with both numerical and categorical features
- **Flexible metrics**: Compatible with any scikit-learn metric
- **Visual insights**: Interactive and static visualization options
- **Performance diagnosis**: Quickly identify problem areas in your model

Double segmentation provides deeper insights than single-feature analysis, helping you better understand your model's behavior across different data segments.
