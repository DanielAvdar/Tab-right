.. _drift_example:

Drift Example
=============

This standalone example demonstrates univariate drift detection using the tab-right library. Drift detection identifies changes in data distributions between reference and current datasets.

Visualization Options
--------------------

Tab-right provides two plotting backends:

1. **Matplotlib** (default): Static plots using the ``_mp`` suffix functions (e.g., ``plot_drift_mp``)
2. **Plotly**: Interactive plots using the standard function names (e.g., ``plot_drift``)

Example: Basic Drift Visualization with Matplotlib
-------------------------------------------------

.. plot::
    :context: close-figs

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tab_right.plotting import plot_drift_mp

    # Create a simple drift metrics dataframe
    df = pd.DataFrame({
        "feature": ["A", "B", "C"],
        "value": [0.1, 0.2, 0.3],
        "metric": ["wasserstein", "cramer_v", "wasserstein"]
    })

    # Plot the drift metrics using the built-in function
    fig = plot_drift_mp(df)
    plt.show()

Example: Basic Drift Visualization with Plotly
----------------------------------------------

Using Plotly for interactive visualizations:

.. code-block:: python

    import pandas as pd
    from tab_right.plotting import plot_drift

    # Create a simple drift metrics dataframe
    df = pd.DataFrame({
        "feature": ["A", "B", "C"],
        "value": [0.1, 0.2, 0.3],
        "metric": ["wasserstein", "cramer_v", "wasserstein"]
    })

    # Plot the drift metrics with Plotly
    fig = plot_drift(df)
    fig.show()

Example: Complete Drift Detection Workflow
-----------------------------------------

.. plot::
    :context: close-figs

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tab_right.drift import univariate
    from tab_right.plotting import plot_drift_mp

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

    # Run drift detection for all columns
    result = univariate.detect_univariate_drift_df(df_ref, df_cur)

    # Plot the drift results using the built-in function
    fig = plot_drift_mp(result)
    plt.show()

Visualizing Feature-level Drift
-------------------------------

.. plot::
    :context: close-figs

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tab_right.drift import univariate
    from tab_right.plotting import plot_feature_drift_mp

    # Using the data from the previous example
    # Analyze drift for a single continuous feature
    metric, value = univariate.detect_univariate_drift(
        df_ref['num_feature'], df_cur['num_feature'], kind='continuous'
    )

    # Plot the feature drift using the built-in function
    fig = plot_feature_drift_mp(
        reference=df_ref['num_feature'],
        current=df_cur['num_feature'],
        feature_name='num_feature'
    )
    plt.show()

Feature-level Drift with Plotly (Interactive)
--------------------------------------------

For interactive visualizations, you can use the Plotly version:

.. code-block:: python

    from tab_right.plotting import plot_feature_drift

    # Interactive feature drift visualization
    fig = plot_feature_drift(
        reference=df_ref['num_feature'],
        current=df_cur['num_feature'],
        feature_name='num_feature'
    )
    fig.show()

Categorical Feature Drift
-------------------------

.. plot::
    :context: close-figs

    # Analyze drift for a categorical feature
    cat_metric, cat_value = univariate.detect_univariate_drift(
        df_ref['cat_feature'], df_cur['cat_feature'], kind='categorical'
    )

    # Calculate category frequencies
    ref_counts = df_ref['cat_feature'].value_counts(normalize=True).sort_index()
    cur_counts = df_cur['cat_feature'].value_counts(normalize=True).sort_index()

    # Create a comparison bar chart
    categories = sorted(list(set(ref_counts.index) | set(cur_counts.index)))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Set positions for the bars
    x = np.arange(len(categories))
    width = 0.35

    # Create the bars
    ax.bar(x - width/2, [ref_counts.get(cat, 0) for cat in categories],
           width, label='Reference', color='blue', alpha=0.7)
    ax.bar(x + width/2, [cur_counts.get(cat, 0) for cat in categories],
           width, label='Current', color='orange', alpha=0.7)

    # Add text and styling
    ax.set_xlabel('Category')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Categorical Feature Comparison: {cat_metric}={cat_value:.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
