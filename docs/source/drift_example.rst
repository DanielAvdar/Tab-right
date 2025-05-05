.. _drift_example:

Drift Example
=============

This example demonstrates how to use tab-right for detecting and visualizing data drift between reference and current datasets. Tab-right provides comprehensive tools for drift detection that help you identify changes in data distributions.

Drift Detection with tab-right
------------------------------

Tab-right offers specialized functions for drift detection:

1. ``univariate.detect_univariate_drift_df`` - Detect drift across all features in a DataFrame
2. ``univariate.detect_univariate_drift`` - Detect drift for a single feature
3. ``plot_drift`` / ``plot_drift_mp`` - Visualize drift metrics across features
4. ``plot_feature_drift`` / ``plot_feature_drift_mp`` - Compare distributions of a specific feature

Visualization Options
---------------------

Tab-right provides two plotting backends:

1. **Matplotlib** (default): Static plots using the ``_mp`` suffix functions (e.g., ``plot_drift_mp``)
2. **Plotly**: Interactive plots using the standard function names (e.g., ``plot_drift``)

Example: Detecting Drift with tab-right
---------------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from tab_right.drift import univariate
    from tab_right.plotting import plot_drift_mp
    import matplotlib.pyplot as plt

    # Generate random data for reference and current datasets
    np.random.seed(42)
    df_ref = pd.DataFrame({
        'num_feature': np.random.normal(0, 1, 1000),
        'cat_feature': np.random.choice(['A', 'B', 'C'], 1000)
    })

    # Create current dataset with slight distributional shift
    df_cur = pd.DataFrame({
        'num_feature': np.random.normal(0.2, 1.2, 1000),  # Shifted mean and variance
        'cat_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.2, 0.5, 0.3])  # Different probabilities
    })

    # Run drift detection for all columns using tab-right's univariate module
    result = univariate.detect_univariate_drift_df(df_ref, df_cur)

    # Plot the drift results using tab-right's built-in function
    fig = plot_drift_mp(result)
    plt.show()

Visualizing Feature-level Drift
-------------------------------

Tab-right provides specialized functions to visualize drift at the individual feature level:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_feature_drift_mp

    # Create sample numeric feature data
    np.random.seed(42)
    reference_data = pd.Series(np.random.normal(0, 1, 1000), name="numeric_feature")
    current_data = pd.Series(np.random.normal(0.5, 1.5, 1000), name="numeric_feature")

    # Use tab-right's feature drift visualization
    fig = plot_feature_drift_mp(
        reference=reference_data,
        current=current_data,
        feature_name="numeric_feature"
    )
    plt.show()

Categorical Feature Drift
-------------------------

Tab-right also handles categorical features with specialized drift detection and visualization:

.. code-block:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_feature_drift_mp

    # Create sample categorical feature data
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D']
    reference_data = pd.Series(np.random.choice(categories, 1000, p=[0.4, 0.3, 0.2, 0.1]), name="category")
    current_data = pd.Series(np.random.choice(categories, 1000, p=[0.2, 0.2, 0.3, 0.3]), name="category")

    # Use tab-right's feature drift visualization for categorical data
    fig = plot_feature_drift_mp(
        reference=reference_data,
        current=current_data,
        feature_name="category"
    )
    plt.show()

Working with Multiple Drift Metrics
-----------------------------------

Tab-right makes it easy to analyze drift using different metrics:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from tab_right.drift import univariate

    # Generate data with multiple features
    np.random.seed(42)
    df_ref = pd.DataFrame({
        'feat1': np.random.normal(0, 1, 1000),
        'feat2': np.random.normal(5, 2, 1000),
        'feat3': np.random.choice(['A', 'B', 'C'], 1000),
        'feat4': np.random.choice(['X', 'Y', 'Z'], 1000),
    })

    # Create current dataset with various kinds of drift
    df_cur = pd.DataFrame({
        'feat1': np.random.normal(0.5, 1.5, 1000),  # Mean and variance shift
        'feat2': np.random.normal(5, 2, 1000),      # No significant drift
        'feat3': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2]),  # Distribution shift
        'feat4': np.random.choice(['X', 'Y', 'Z'], 1000)  # No significant drift
    })

    # Run drift detection (tab-right automatically selects appropriate metrics)
    result = univariate.detect_univariate_drift_df(df_ref, df_cur)

    # You can also check individual features with specific metrics:
    wasserstein_metric, wasserstein_value = univariate.detect_univariate_drift(
        df_ref['feat1'], df_cur['feat1'],
        kind='continuous',
        metric='wasserstein'  # Explicitly request Wasserstein distance
    )

    print(f"Wasserstein distance for feat1: {wasserstein_value:.4f}")

Key Drift Detection Features in tab-right
-----------------------------------------

Tab-right offers comprehensive drift detection capabilities:

- **Automatic feature type detection**: tab-right selects appropriate metrics based on feature type
- **Multiple drift metrics**: Wasserstein distance, KS test, PSI, Cramer's V
- **Visualization tools**: Compare distributions visually with histogram overlays and statistical metrics
- **Seamless integration**: Works with pandas DataFrames and Series for easy integration with data workflows
- **Multi-feature analysis**: Analyze drift across all features in a dataset at once
- **Interactive and static plots**: Choose between Plotly (interactive) or Matplotlib (static) visualizations

These tools make it easy to track and analyze distribution shifts in your data, helping you maintain model performance over time.
