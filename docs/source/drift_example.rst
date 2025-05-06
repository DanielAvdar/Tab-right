.. _drift_example:

Drift Example
=============

This example demonstrates how to use tab-right for detecting and visualizing data drift between reference and current datasets. Tab-right provides comprehensive tools for drift detection that help you identify changes in data distributions.

Drift Detection with tab-right
------------------------------

Tab-right offers specialized components for drift detection:

1. ``DriftCalculator`` - Core class for calculating drift between datasets
2. ``DriftPlotter`` - Visualization class for creating matplotlib-based drift plots
3. ``univariate`` module - Lower-level functions for specific drift calculations
4. ``plot_drift`` / ``plot_feature_drift`` modules - Simplified plotting functions

Available Drift Metrics
-----------------------

Tab-right provides multiple metrics for different types of features:

**Numerical Features:**
- **Wasserstein Distance** (default): Measures the earth mover's distance between distributions
- **Kolmogorov-Smirnov Test**: Statistical test for equality of continuous distributions
- **Population Stability Index (PSI)**: Measure of population stability over time

**Categorical Features:**
- **Cramer's V** (default): Normalized measure of association between categorical variables
- **Chi-Square Test**: Statistical test for independence of categorical variables
- **PSI**: Can also be applied to categorical features by comparing proportions

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
    from tab_right.plotting import plot_drift_mp

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

    # Plot the results using matplotlib
    fig = plot_drift_mp(result)
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

    # Create figure with 3 subplots for different drift levels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Generate base reference data
    np.random.seed(42)
    df_ref = pd.DataFrame({
        'feature': np.random.normal(0, 1, 500),
    })

    # Generate current data with increasing levels of drift
    df_slight = pd.DataFrame({'feature': np.random.normal(0.2, 1.1, 500)})
    df_moderate = pd.DataFrame({'feature': np.random.normal(0.5, 1.3, 500)})
    df_severe = pd.DataFrame({'feature': np.random.normal(2.0, 1.8, 500)})

    # Calculate and plot for each drift level
    for i, (title, df_cur) in enumerate([
        ('Slight Drift', df_slight),
        ('Moderate Drift', df_moderate),
        ('Severe Drift', df_severe)
    ]):
        calc = DriftCalculator(df_ref, df_cur)
        result = calc()
        score = result['feature'].values[0]

        # Get distributions for plotting
        ref_hist, cur_hist = calc.get_prob_density('feature')

        # Plot distributions
        ax = axes[i]
        ax.plot(ref_hist[1][:-1], ref_hist[0], 'b-', label='Reference')
        ax.plot(cur_hist[1][:-1], cur_hist[0], 'r-', label='Current')
        ax.set_title(f"{title}\nDrift Score: {score:.3f}")
        ax.legend()

    plt.tight_layout()
    plt.show()

Key Features of tab-right's Drift Detection
-------------------------------------------

Tab-right offers comprehensive drift detection capabilities:

- **Flexible API**: Choose between object-oriented (DriftCalculator/DriftPlotter) or functional approaches
- **Automatic feature type detection**: Appropriate metrics are selected based on the data type
- **Multiple drift metrics**: Including Wasserstein distance, KS test, PSI, and Cramer's V
- **Matplotlib integration**: Create publication-ready plots with built-in matplotlib figures
- **Multi-feature analysis**: Analyze drift across all features at once
- **Probability density comparison**: Examine detailed distribution changes

These tools make it easy to track and analyze distribution shifts in your data, helping you maintain model performance over time.
