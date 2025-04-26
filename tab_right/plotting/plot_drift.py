"""Module for plotting drift values for each feature as a line chart."""

import pandas as pd
import plotly.graph_objects as go


def plot_drift(drift_df: pd.DataFrame, value_col: str = "value", feature_col: str = "feature") -> go.Figure:
    """Plot drift values for each feature as a line chart.

    Parameters
    ----------
    drift_df : pd.DataFrame
        DataFrame with drift results. Should contain columns for feature names and drift values.
    value_col : str, default "value"
        Name of the column containing drift values.
    feature_col : str, default "feature"
        Name of the column containing feature names.

    Returns
    -------
    go.Figure
        Plotly line chart of drift values by feature.

    """
    drift_df_sorted = drift_df.sort_values(value_col, ascending=False)
    fig = go.Figure(
        go.Scatter(
            x=drift_df_sorted[feature_col],
            y=drift_df_sorted[value_col],
            mode="lines+markers",
            line=dict(color="indianred"),
            marker=dict(size=8),
            name="Drift Value",
        )
    )
    fig.update_layout(
        title="Univariate Drift by Feature", xaxis_title="Feature", yaxis_title="Drift Value", xaxis_tickangle=-45
    )
    return fig
