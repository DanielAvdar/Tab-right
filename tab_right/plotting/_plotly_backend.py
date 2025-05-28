"""Plotly backend for drift plotting."""

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go


def plot_drift_values(
    drift_df: pd.DataFrame,
    value_col: str = "value",
    feature_col: str = "feature",
) -> go.Figure:
    """Plot drift values for each feature as a bar chart using Plotly.

    Args:
        drift_df: DataFrame with drift results. Should contain columns for feature names and drift values.
        value_col: Name of the column containing drift values.
        feature_col: Name of the column containing feature names.

    Returns:
        go.Figure: Plotly bar chart of drift values by feature.
    """
    drift_df_sorted = drift_df.sort_values(value_col, ascending=False)
    fig = go.Figure(
        go.Bar(
            x=drift_df_sorted[feature_col],
            y=drift_df_sorted[value_col],
            marker_color="indianred",
            name="Drift Value",
        )
    )
    fig.update_layout(
        title="Univariate Drift by Feature",
        xaxis_title="Feature",
        yaxis_title="Drift Value",
        xaxis_tickangle=-45,
    )
    return fig