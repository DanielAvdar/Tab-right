.. _drift_example:

Drift Example
=============

This standalone example demonstrates univariate drift detection using random data.

.. code-block:: python

    """Standalone example for univariate drift detection using random data."""
    import numpy as np
    import pandas as pd
    from tab_right.drift import univariate

    # Generate random data for reference and current
    df_ref = pd.DataFrame({
        'num_feature': np.random.normal(0, 1, 1000),
        'cat_feature': np.random.choice(['A', 'B', 'C'], 1000)
    })
    df_cur = pd.DataFrame({
        'num_feature': np.random.normal(0.2, 1.2, 1000),
        'cat_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.2, 0.5, 0.3])
    })

.. code-block:: python

    # Run drift detection for all columns
    result = univariate.detect_univariate_drift_df(df_ref, df_cur)
    print("Univariate drift detection results:")
    print(result)

.. code-block:: python

    # Plot drift results for all features
    from tab_right.plotting.plot_drift import plot_drift
    fig = plot_drift(result)
    fig.show()

.. code-block:: python

    # Example: run for a single column
    metric, value = univariate.detect_univariate_drift(df_ref['num_feature'], df_cur['num_feature'], kind='continuous')
    print(f"\nDrift metric for 'num_feature': {metric} = {value:.4f}")

.. code-block:: python

    # Plot drift for a single feature
    from tab_right.plotting.plot_feature_drift import plot_feature_drift
    fig_feat = plot_feature_drift(df_ref['num_feature'], df_cur['num_feature'], feature_name='num_feature')
    fig_feat.show()
