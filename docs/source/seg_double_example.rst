.. _seg_double_example:

Double Segmentation
===================

This example demonstrates how to use tab-right's double segmentation functionality to analyze model performance across combinations of two features in tabular data.

What is Double Segmentation?
----------------------------

Double segmentation is a powerful tab-right feature that allows you to analyze how your model performs across different combinations of two features. This is especially useful for:

- Identifying feature interactions that affect model performance
- Finding specific combinations of feature values where your model underperforms
- Understanding complex patterns in your data that single-feature analysis might miss

Tab-right's Double Segmentation Tools
-------------------------------------

Tab-right provides comprehensive tools for double segmentation analysis:

1. ``DoubleSegmentationImp`` - Main implementation class for performing double segmentation
2. ``DoubleSegmPlotting`` - Interactive Plotly-based visualization of double segmentation results
3. ``DoubleSegmPlottingMp`` - Matplotlib-based visualization of double segmentation results

Basic Usage with tab-right
--------------------------

Here's a complete example showing how to use tab-right's double segmentation capabilities:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from tab_right.segmentations.double_seg import DoubleSegmentationImp
    from tab_right.plotting.plot_segmentations import DoubleSegmPlottingMp

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

    # Use tab-right's DoubleSegmentationImp to perform double segmentation
    double_seg = DoubleSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply double segmentation using tab-right
    result_df = double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        score_metric=mean_squared_error,
        bins_1=4,
        bins_2=4
    )

    # Visualize results using tab-right's DoubleSegmPlottingMp class
    double_plot = DoubleSegmPlottingMp(df=result_df)
    fig = double_plot.plot_heatmap()
    plt.title("Model MSE by Feature Segments (tab-right Double Segmentation)")
    plt.show()

Interactive Visualization with tab-right's Plotly Backend
---------------------------------------------------------

Tab-right provides interactive visualizations using Plotly for better exploration of segment interactions:

.. code-block:: python

    from tab_right.plotting.plot_segmentations import DoubleSegmPlotting

    # Using the result_df from the double segmentation
    # Create an interactive visualization with tab-right's Plotly backend
    interactive_plot = DoubleSegmPlotting(df=result_df)
    fig = interactive_plot.plot_heatmap()
    fig.update_layout(title="Interactive Double Segmentation Heatmap")
    fig.show()

Customizing Double Segmentation with tab-right
-----------------------------------------------

Tab-right offers flexibility in how you configure double segmentation:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from tab_right.segmentations.double_seg import DoubleSegmentationImp
    from tab_right.plotting.plot_segmentations import DoubleSegmPlottingMp

    # Using the same data from before

    # Create a double segmentation instance with tab-right
    custom_double_seg = DoubleSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply custom double segmentation with different binning and metric
    custom_result = custom_double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        score_metric=mean_absolute_error,  # Using MAE instead of MSE
        bins_1=5,  # More bins for feature1
        bins_2=3   # Fewer bins for feature2
    )

    # Visualize with tab-right's built-in plotting class
    # Note: The column name in the result dataframe is 'score' by default
    custom_plot = DoubleSegmPlottingMp(
        df=custom_result,
        metric_name="score",  # Use the default column name
        lower_is_better=True  # Indicate that lower values are better
    )

    # Generate the plot using tab-right's visualization
    fig = custom_plot.plot_heatmap()
    plt.title("Model MAE by Feature Segments (Custom Configuration)")
    plt.show()

Finding Performance Issues with Double Segmentation
---------------------------------------------------

Tab-right's double segmentation is particularly useful for identifying problem areas in your model:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from tab_right.segmentations.double_seg import DoubleSegmentationImp

    # Using the original data
    # Calculate overall model performance
    overall_mse = mean_squared_error(df['target'], df['prediction'])

    # Run double segmentation with tab-right
    problem_double_seg = DoubleSegmentationImp(
        df=df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply double segmentation
    problem_result_df = problem_double_seg(
        feature1_col='feature1',
        feature2_col='feature2',
        score_metric=mean_squared_error,
        bins_1=3,
        bins_2=3
    )

    # Find segments with MSE > 1.5x overall MSE (potential problem areas)
    threshold = 1.5 * overall_mse
    problem_segments = problem_result_df[problem_result_df['score'] > threshold]

    # Print information about problem segments
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Found {len(problem_segments)} problematic segments out of {len(problem_result_df)}")

    # Print details about problem segments
    for _, row in problem_segments.iterrows():
        print(f"Problem area: feature1={row['feature_1']}, feature2={row['feature_2']}, MSE={row['score']:.4f}")

    # You can then investigate these specific segments further or target them for model improvements

Working with Categorical Features
---------------------------------

Tab-right's double segmentation also works with categorical features:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from tab_right.segmentations.double_seg import DoubleSegmentationImp
    from tab_right.plotting.plot_segmentations import DoubleSegmPlottingMp

    # Create sample data with categorical features
    np.random.seed(42)
    n = 1000

    # Create categorical features
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n)
    industry = np.random.choice(['Tech', 'Healthcare', 'Finance', 'Retail'], n)

    # Create binary target with interaction effect
    # People with higher education in Tech and Finance have higher success rates
    base_prob = np.ones(n) * 0.5

    # Education effect
    edu_effect = np.zeros(n)
    edu_effect[education == 'High School'] = -0.1
    edu_effect[education == 'Bachelor'] = 0
    edu_effect[education == 'Master'] = 0.1
    edu_effect[education == 'PhD'] = 0.2

    # Industry effect
    ind_effect = np.zeros(n)
    ind_effect[industry == 'Tech'] = 0.15
    ind_effect[industry == 'Finance'] = 0.10
    ind_effect[industry == 'Healthcare'] = 0.05
    ind_effect[industry == 'Retail'] = -0.05

    # Interaction effect (extra boost for PhD in Tech)
    interaction = np.zeros(n)
    interaction[(education == 'PhD') & (industry == 'Tech')] = 0.1

    # Calculate probability and generate target
    probability = base_prob + edu_effect + ind_effect + interaction
    probability = np.clip(probability, 0.1, 0.9)  # Ensure probabilities are between 0.1 and 0.9
    target = np.random.binomial(1, probability)

    # Create simple prediction (without capturing the interaction effect)
    simple_prob = base_prob + edu_effect + ind_effect
    simple_prob = np.clip(simple_prob, 0.1, 0.9)
    prediction = np.random.binomial(1, simple_prob)

    # Create DataFrame
    cat_df = pd.DataFrame({
        'education': education,
        'industry': industry,
        'target': target,
        'prediction': prediction
    })

    # Use tab-right's DoubleSegmentationImp
    cat_double_seg = DoubleSegmentationImp(
        df=cat_df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply double segmentation (no need to specify bins for categorical features)
    cat_result_df = cat_double_seg(
        feature1_col='education',
        feature2_col='industry',
        score_metric=accuracy_score,  # Use accuracy for binary classification
    )

    # Visualize with tab-right's plotting
    cat_plot = DoubleSegmPlottingMp(
        df=cat_result_df,
        metric_name="score",
        lower_is_better=False  # For accuracy, higher is better
    )

    # Create the visualization
    fig = cat_plot.plot_heatmap()
    plt.title("Model Accuracy by Education and Industry Segments")
    plt.show()

Key Features of tab-right's Double Segmentation
------------------------------------------------

- **Feature interaction analysis**: Discover how combinations of features affect model performance
- **Automatic binning**: Handles both categorical and numerical features appropriately
- **Flexible metric support**: Works with any scikit-learn compatible metric function
- **Interactive visualizations**: Explore results with both Matplotlib and Plotly backends
- **Comprehensive API**: Consistent interface with the rest of the tab-right toolkit

Double segmentation is one of tab-right's most powerful features for identifying specific areas where your model needs improvement.
