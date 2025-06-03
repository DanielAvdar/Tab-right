.. _model_analysis:
.. _drift:
.. _seg_calc:
.. _seg_double:

Model Analysis Guide
====================

This guide demonstrates how to use tab-right's comprehensive analysis tools for understanding model performance and detecting data changes. Tab-right provides three main analysis capabilities:

1. **Drift Detection** - Identify changes in data distributions between reference and current datasets
2. **Segmentation Calculation** - Analyze model performance across different data segments
3. **Double Segmentation** - Examine performance across combinations of two features

.. _drift_detection:

Drift Detection
===============

This section demonstrates how to use tab-right for detecting and visualizing data drift between reference and current datasets. Tab-right provides comprehensive tools for drift detection that help you identify changes in data distributions.

Drift Detection with tab-right
------------------------------

Tab-right offers specialized components for drift detection:

1. ``DriftCalculator`` - Core class for calculating drift between datasets
2. ``DriftPlotter`` - Visualization class for creating plots with both matplotlib and plotly backends
3. ``univariate`` module - Lower-level functions for specific drift calculations

Available Drift Metrics
-----------------------

Tab-right provides multiple metrics for different types of features:

**Numerical Features:**
- **Wasserstein Distance** (default): Measures the earth mover's distance between distributions
- **Kolmogorov-Smirnov Test**: Statistical test for equality of continuous distributions

**Categorical Features:**
- **Cramer's V** (default): Normalized measure of association between categorical variables
- **Chi-Square Test**: Statistical test for independence of categorical variables

Example: Using DriftCalculator and DriftPlotter
-----------------------------------------------

The most concise way to analyze and visualize drift with tab-right is to use the ``DriftCalculator`` and ``DriftPlotter`` classes:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tab_right.drift.drift_calculator import DriftCalculator
    from tab_right.plotting.drift_plotter import DriftPlotter

    # Generate simple dataset for demo
    np.random.seed(42)
    df1 = pd.DataFrame({
        'numeric': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])
    })

    df2 = pd.DataFrame({
        'numeric': np.random.normal(1, 1.2, 120),  # Shift in distribution
        'category': np.random.choice(['A', 'B', 'C'], 120, p=[0.2, 0.3, 0.5])  # Different proportions
    })

    # Create the drift calculator
    drift_calc = DriftCalculator(df1, df2)

    # Create the plotter
    plotter = DriftPlotter(drift_calc)

    # Plot summary of drift across features
    fig = plotter.plot_multiple()
    plt.tight_layout()
    plt.show()

Feature-Level Distribution Comparison
-------------------------------------

You can also examine the distribution shifts for individual features:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tab_right.drift.drift_calculator import DriftCalculator
    from tab_right.plotting.drift_plotter import DriftPlotter

    # Generate datasets with drift
    np.random.seed(42)
    df1 = pd.DataFrame({
        'numeric': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])
    })

    df2 = pd.DataFrame({
        'numeric': np.random.normal(1, 1.2, 120),
        'category': np.random.choice(['A', 'B', 'C'], 120, p=[0.2, 0.3, 0.5])
    })

    # Create calculator and plotter
    drift_calc = DriftCalculator(df1, df2)
    plotter = DriftPlotter(drift_calc)

    # Plot numerical feature distribution comparison
    fig_numeric = plotter.plot_single('numeric')
    plt.tight_layout()
    plt.show()

Categorical Feature Visualization
---------------------------------

Tab-right also makes it easy to visualize categorical feature drift:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tab_right.drift.drift_calculator import DriftCalculator
    from tab_right.plotting.drift_plotter import DriftPlotter

    # Generate datasets with categorical drift
    np.random.seed(42)
    df1 = pd.DataFrame({
        'numeric': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2])
    })

    df2 = pd.DataFrame({
        'numeric': np.random.normal(1, 1.2, 120),
        'category': np.random.choice(['A', 'B', 'C'], 120, p=[0.2, 0.3, 0.5])
    })

    # Create calculator and plotter
    drift_calc = DriftCalculator(df1, df2)
    plotter = DriftPlotter(drift_calc)

    # Plot categorical feature distribution comparison
    fig_cat = plotter.plot_single('category')
    plt.tight_layout()
    plt.show()

Direct Functions API
--------------------

For simpler use cases, tab-right also provides direct functions for drift analysis:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tab_right.drift import univariate
    from tab_right.plotting import DriftPlotter

    # Generate datasets
    np.random.seed(42)
    df_ref = pd.DataFrame({
        'num_feature': np.random.normal(0, 1, 500),
        'cat_feature': np.random.choice(['A', 'B', 'C'], 500)
    })

    df_cur = pd.DataFrame({
        'num_feature': np.random.normal(0.3, 1.2, 500),
        'cat_feature': np.random.choice(['A', 'B', 'C'], 500, p=[0.2, 0.5, 0.3])
    })

    # Calculate drift across all features
    result = univariate.detect_univariate_drift_df(df_ref, df_cur)

    # Plot the results using DriftPlotter
    fig = DriftPlotter.plot_drift_mp(None, result)
    plt.tight_layout()
    plt.show()

Working with Multiple Drift Metrics
-----------------------------------

Tab-right supports various drift metrics that can be customized:

.. plot::
    :include-source:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.drift import univariate
    from tab_right.drift.drift_calculator import DriftCalculator
    from tab_right.plotting.drift_plotter import DriftPlotter

    # Generate data
    np.random.seed(42)
    df_ref = pd.DataFrame({
        'feat1': np.random.normal(0, 1, 500),
        'feat2': np.random.choice(['A', 'B', 'C'], 500),
    })

    df_cur = pd.DataFrame({
        'feat1': np.random.normal(0.5, 1.5, 500),
        'feat2': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2]),
    })

    # Using DriftCalculator with default metrics
    calc = DriftCalculator(df_ref, df_cur)

    # Create a plotter
    plotter = DriftPlotter(calc)

    # Plot the results
    fig = plotter.plot_multiple()
    plt.title('Drift Analysis with Default Metrics')
    plt.tight_layout()
    plt.show()

Visualizing Different Types of Drift
------------------------------------

Let's look at how different degrees of drift appear in tab-right visualizations:

.. plot::
    :include-source:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.drift.drift_calculator import DriftCalculator
    from tab_right.plotting.drift_plotter import DriftPlotter

    # Create datasets with increasing levels of drift
    np.random.seed(42)
    ref_data = np.random.normal(0, 1, 500)

    # Create three datasets with different levels of drift
    slight_drift = np.random.normal(0.2, 1.1, 500)  # slight drift
    moderate_drift = np.random.normal(0.5, 1.3, 500)  # moderate drift
    severe_drift = np.random.normal(2.0, 1.8, 500)  # severe drift

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Set up titles
    titles = ['Slight Drift', 'Moderate Drift', 'Severe Drift']
    drift_data = [slight_drift, moderate_drift, severe_drift]

    # Create and plot each dataset using tab_right
    for i, current_data in enumerate(drift_data):
        # Create DataFrames
        df_ref = pd.DataFrame({'value': ref_data})
        df_cur = pd.DataFrame({'value': current_data})

        # Calculate drift
        drift_calc = DriftCalculator(df_ref, df_cur)
        drift_result = drift_calc()
        drift_score = round(drift_result.iloc[0]['score'], 3)

        # Create plotter
        plotter = DriftPlotter(drift_calc)

        # Plot distribution on the corresponding subplot
        dist_fig = plotter.plot_single('value')

        # Remove the original figure and copy its content to our subplot
        for line in dist_fig.axes[0].lines:
            axes[i].plot(line.get_xdata(), line.get_ydata(),
                         color=line.get_color(), label=line.get_label())

        # Set title with drift score
        axes[i].set_title(f"{titles[i]}\nDrift Score: {drift_score}")
        axes[i].legend()

        # Close the original figure to prevent display
        plt.close(dist_fig)

    plt.tight_layout()
    plt.show()

Key Features of tab-right's Drift Detection
-------------------------------------------

Tab-right offers comprehensive drift detection capabilities:

- **Flexible API**: Choose between object-oriented (DriftCalculator/DriftPlotter) or functional approaches
- **Automatic feature type detection**: Appropriate metrics are selected based on the data type
- **Multiple drift metrics**: Including Wasserstein distance, KS test, and Cramer's V
- **Matplotlib integration**: Create publication-ready plots with built-in matplotlib figures
- **Multi-feature analysis**: Analyze drift across all features at once
- **Probability density comparison**: Examine detailed distribution changes

These tools make it easy to track and analyze distribution shifts in your data, helping you maintain model performance over time.

.. _segmentation_calculation:

Segmentation Calculation
========================

This section demonstrates how to use tab-right's segmentation calculation (SegmentationCalc) and its plotting functionality.

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
3. ``DoubleSegmPlotting`` - Visualize interactions between two segment features (supports both plotly and matplotlib backends)

Basic Usage
-----------

Here's a simple example showing how to create segment data and visualize it:

.. plot::
    :include-source:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_single_segmentation_mp

    # Create a simple results DataFrame with segments
    segments = pd.DataFrame({
        'segment_id': [0, 1, 2],
        'segment_name': ['Age < 30', '30 ≤ Age < 50', 'Age ≥ 50'],
        'score': [0.85, 0.92, 0.77]
    })

    # Plot the segmentation results using matplotlib
    plot_single_segmentation_mp(segments)
    plt.show()

Working with Actual Data
------------------------

For real-world analysis with your own data:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error

    # Create DataFrameGroupBy object with segment information
    df = pd.DataFrame({
        'age': [25, 28, 35, 42, 55, 60],
        'segment_id': [0, 0, 1, 1, 2, 2],
        'true_value': [10, 12, 15, 14, 20, 18],
        'prediction': [11, 13, 14, 16, 17, 16]
    })

    # Group by segment_id to create the DataFrameGroupBy object
    grouped_df = df.groupby('segment_id')

    # Create mapping from segment_id to readable names
    segment_names = {
        0: 'Age < 30',
        1: '30 ≤ Age < 50',
        2: 'Age ≥ 50'
    }

    # Define metric function (MAE)
    def calc_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    # Create segmentation calculator
    from tab_right.segmentations import SegmentationCalc
    seg_calc = SegmentationCalc(
        gdf=grouped_df,
        label_col='true_value',
        prediction_col='prediction',
        segment_names=segment_names
    )

    # Apply metric to calculate segment scores
    segments = seg_calc(calc_mae)

    # Plot the results
    from tab_right.plotting import plot_single_segmentation_mp
    plot_single_segmentation_mp(segments)
    plt.show()

Visualization with Higher-is-Better Metrics
-------------------------------------------

For metrics where higher values are better (like R²), use the `lower_is_better=False` parameter:

.. plot::
    :include-source:

    import pandas as pd
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_single_segmentation_mp

    # Create a DataFrame with example R² values by segment
    r2_segments = pd.DataFrame({
        'segment_id': [0, 1, 2, 3],
        'segment_name': ['Age < 30', '30 ≤ Age < 50', '50 ≤ Age < 65', 'Age ≥ 65'],
        'score': [0.82, 0.91, 0.76, 0.68]  # R² values (higher is better)
    })

    # Plot with lower_is_better=False for R²
    plot_single_segmentation_mp(r2_segments, lower_is_better=False)
    plt.title("R² by Age Segment")
    plt.show()

Interactive Visualization with Plotly
-------------------------------------

Tab-right also supports Plotly for interactive visualizations:

.. code-block:: python

    from tab_right.plotting import plot_single_segmentation

    # Create interactive visualization
    fig = plot_single_segmentation(segments)
    fig.show()  # Opens in browser or notebook

Key Steps for Segmentation Analysis
-----------------------------------

1. **Group your data** by segments using pandas' groupby function
2. **Create segment names** for better interpretation of results
3. **Initialize SegmentationCalc** with the grouped data
4. **Apply a metric function** to calculate segment scores
5. **Visualize the results** using plot_single_segmentation_mp (matplotlib) or plot_single_segmentation (Plotly)

This workflow makes it easy to identify segments where your model performs differently, helping you understand where improvements are needed.

Key Benefits of Using tab-right for Segmentation
------------------------------------------------

- **Standardized API**: Consistent interface for all segmentation analyses
- **Automatic handling of missing values**: Robust processing of incomplete data
- **Support for multiple metrics**: Easy comparison across various evaluation metrics
- **Flexible visualization options**: Both static and interactive plotting
- **Compatible with scikit-learn**: Works with any scikit-learn compatible metric function

Tab-right's segmentation functionality helps you understand where your model performs well and where it needs improvement, enabling targeted model enhancements and better decision-making.

.. _double_segmentation:

Double Segmentation
===================

This section demonstrates how to use tab-right's double segmentation functionality to analyze model performance across combinations of two features.

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
2. ``DoubleSegmPlotting`` - Visualization with support for both interactive Plotly and static Matplotlib backends

Basic Usage with Continuous Features
------------------------------------

Here's a simple example of double segmentation with tab-right using continuous features:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from tab_right.segmentations import DoubleSegmentationImp
    from tab_right.plotting import DoubleSegmPlotting

    # Create sample data
    np.random.seed(42)
    n_samples = 500

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
    plotter = DoubleSegmPlotting(df=result_df, backend="matplotlib")
    fig = plotter.plot_heatmap()
    plt.title("MSE by Feature1 and Feature2 Segments")

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
    from tab_right.plotting import DoubleSegmPlotting

    # Create sample categorical data
    np.random.seed(42)
    n = 800

    # Generate categorical features with non-uniform distributions
    category1 = np.random.choice(
        ['A', 'B', 'C', 'D'],
        n,
        p=[0.4, 0.3, 0.2, 0.1]  # Different probabilities for each category
    )
    category2 = np.random.choice(
        ['X', 'Y', 'Z'],
        n,
        p=[0.5, 0.3, 0.2]
    )

    # Generate target with different patterns for combinations
    target = np.zeros(n, dtype=int)

    # Add different effects for different combinations
    target[(category1 == 'A') & (category2 == 'X')] = 1
    target[(category1 == 'B') & (category2 == 'Y')] = 1
    target[(category1 == 'C') & (category2 == 'Z')] = 1
    # Special case with stronger effect
    target[(category1 == 'D') & (category2 == 'Z')] = np.random.binomial(1, 0.8, np.sum((category1 == 'D') & (category2 == 'Z')))

    # Add some noise
    noise_mask = np.random.choice([True, False], n, p=[0.1, 0.9])
    target[noise_mask] = 1 - target[noise_mask]

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
    cat_plot = DoubleSegmPlotting(
        df=cat_results,
        lower_is_better=False,
        backend="matplotlib"
    )
    fig = cat_plot.plot_heatmap()
    plt.title("Accuracy by Category Segments")

Mixed Categorical and Continuous Features
-----------------------------------------

Double segmentation can analyze combinations of categorical and continuous features:

.. plot::
    :include-source:

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    from tab_right.segmentations import DoubleSegmentationImp
    from tab_right.plotting import DoubleSegmPlotting

    # Create sample data with mixed feature types
    np.random.seed(42)
    n_samples = 500

    # Generate categorical feature - product type
    product_types = ['Basic', 'Standard', 'Premium', 'Enterprise']
    product = np.random.choice(product_types, n_samples, p=[0.4, 0.3, 0.2, 0.1])

    # Generate continuous feature - customer spending
    spending = np.random.gamma(shape=5, scale=20, size=n_samples)

    # Add variation by product type
    spending[product == 'Premium'] *= 1.5
    spending[product == 'Enterprise'] *= 2.0

    # Simple model: customers return if they have premium products OR spend a lot
    premium_mask = np.logical_or(product == 'Premium', product == 'Enterprise')
    return_prob = 0.2 + 0.3 * premium_mask + 0.4 * (spending > np.percentile(spending, 70))
    return_prob = np.clip(return_prob, 0.1, 0.9)

    # Generate actual returns (target)
    customer_return = np.random.binomial(1, return_prob)

    # Simple prediction (missing some patterns)
    pred_prob = 0.2 + 0.4 * (product == 'Enterprise') + 0.3 * (spending > np.percentile(spending, 80))
    pred_prob = np.clip(pred_prob, 0.1, 0.9)
    prediction = np.random.binomial(1, pred_prob)

    # Create DataFrame
    mixed_df = pd.DataFrame({
        'product': product,
        'spending': spending,
        'target': customer_return,
        'prediction': prediction
    })

    # Perform double segmentation
    mixed_seg = DoubleSegmentationImp(
        df=mixed_df,
        label_col='target',
        prediction_col='prediction'
    )

    # Apply segmentation
    mixed_results = mixed_seg(
        feature1_col='product',
        feature2_col='spending',
        score_metric=f1_score,
        bins_2=4  # 4 bins for spending
    )

    # Plot with higher is better for F1 score
    mixed_plot = DoubleSegmPlotting(
        df=mixed_results,
        lower_is_better=False,
        backend="matplotlib"
    )
    fig = mixed_plot.plot_heatmap()
    plt.title("F1 Score by Product Type and Spending")

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
    r2_plotter = DoubleSegmPlotting(df=r2_results, lower_is_better=False, backend="matplotlib")
    r2_plotter.plot_heatmap()
    plt.title("R² Score by Feature Segments")

Key Features of Double Segmentation
-----------------------------------

- **Discover interactions**: Find how combinations of features affect performance
- **Automatic handling**: Works with both numerical and categorical features
- **Flexible metrics**: Compatible with any scikit-learn metric
- **Visual insights**: Interactive and static visualization options
- **Performance diagnosis**: Quickly identify problem areas in your model

Double segmentation provides deeper insights than single-feature analysis, helping you better understand your model's behavior across different data segments.