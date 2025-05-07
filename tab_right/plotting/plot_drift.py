"""Module for plotting drift values for each feature as a bar chart."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure


def plot_drift(drift_df: pd.DataFrame, value_col: str = "value", feature_col: str = "feature") -> go.Figure:
    """Plot drift values for each feature as a bar chart.

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
        Plotly bar chart of drift values by feature.

    """
    drift_df_sorted = drift_df.sort_values(value_col, ascending=False)
    fig = go.Figure(
        go.Bar(
            x=drift_df_sorted[feature_col], y=drift_df_sorted[value_col], marker_color="indianred", name="Drift Value"
        )
    )
    fig.update_layout(
        title="Univariate Drift by Feature", xaxis_title="Feature", yaxis_title="Drift Value", xaxis_tickangle=-45
    )
    return fig


def plot_drift_mp(drift_df: pd.DataFrame, value_col: str = "value", feature_col: str = "feature") -> Figure:
    """Plot drift values for each feature as a bar chart using matplotlib.

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
    Figure
        Matplotlib figure with bar chart of drift values by feature.

    """
    drift_df_sorted = drift_df.sort_values(value_col, ascending=False)

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(drift_df_sorted[feature_col], drift_df_sorted[value_col], color="indianred")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.3f}", ha="center", va="bottom", fontsize=9
        )

    # Customize plot
    ax.set_title("Univariate Drift by Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Drift Value")
    plt.xticks(rotation=-45, ha="left")
    plt.tight_layout()

    return fig
