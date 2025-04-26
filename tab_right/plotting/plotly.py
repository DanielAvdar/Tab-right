"""Plotting utilities for tab-right plotting subpackage."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import wasserstein_distance


def plot_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str = None,
    show_score: bool = True,
    ref_label: str = "Train Dataset",
    cur_label: str = "Test Dataset",
) -> go.Figure:
    """Plot distribution drift for a single feature, with means, medians, and drift score (Earth Mover's Distance).

    Parameters
    ----------
    reference : pd.Series
        Reference (train) data for the feature.
    current : pd.Series
        Current (test) data for the feature.
    feature_name : str, optional
        Name of the feature (for labeling plots).
    show_score : bool, default True
        Whether to display the drift score annotation.
    ref_label : str, default "Train Dataset"
        Label for the reference data.
    cur_label : str, default "Test Dataset"
        Label for the current data.

    Returns
    -------
    go.Figure
        Plotly figure with overlaid histograms, means, medians, and drift score annotation.

    """
    feature_name = feature_name or reference.name or "feature"
    # Compute drift score
    drift_score = None
    if len(reference) > 0 and len(current) > 0:
        drift_score = wasserstein_distance(reference, current)
    # KDEs
    ref_kde = reference
    cur_kde = current
    # Plot
    fig = go.Figure()
    # Reference KDE
    fig.add_trace(
        go.Histogram(
            x=ref_kde,
            histnorm="probability density",
            name=ref_label,
            opacity=0.5,
            marker_color="navy",
            nbinsx=40,
            showlegend=True,
        )
    )
    # Current KDE
    fig.add_trace(
        go.Histogram(
            x=cur_kde,
            histnorm="probability density",
            name=cur_label,
            opacity=0.5,
            marker_color="mediumaquamarine",
            nbinsx=40,
            showlegend=True,
        )
    )
    # Means and medians
    for arr, color, label, dash in [
        (reference, "navy", f"{ref_label} Mean", "dash"),
        (current, "mediumaquamarine", f"{cur_label} Mean", "dash"),
        (reference, "navy", f"{ref_label} Median", "dot"),
        (current, "mediumaquamarine", f"{cur_label} Median", "dot"),
    ]:
        stat = np.mean(arr) if "Mean" in label else np.median(arr)
        fig.add_vline(
            x=stat,
            line=dict(color=color, dash=dash, width=2),
            annotation_text=label,
            annotation_position="top",
            annotation_font_color=color,
            annotation_font_size=12,
        )
    # Layout
    fig.update_layout(
        title=feature_name,
        xaxis_title=feature_name,
        yaxis_title="Probability Density",
        barmode="overlay",
        legend_title="Legend",
        template="plotly_white",
    )
    if show_score:
        score_text = (
            f"Drift Score ({feature_name}): <b>{drift_score:.3f}</b>"
            if drift_score is not None
            else "Drift Score: N/A (empty input)"
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.13,
            showarrow=False,
            text=score_text,
            font=dict(size=16, color="black"),
            align="center",
            bgcolor="rgba(255,255,255,0.7)",
        )
    return fig


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
    fig = go.Figure(go.Bar(x=drift_df_sorted[feature_col], y=drift_df_sorted[value_col], marker_color="indianred"))
    fig.update_layout(
        title="Univariate Drift by Feature", xaxis_title="Feature", yaxis_title="Drift Value", xaxis_tickangle=-45
    )
    return fig


def plot_segmentations(
    df: pd.DataFrame,
    good_color: str = "green",
    bad_color: str = "red",
    score_col: str = "score",
    segment_col: str = "segment",
    ascending: bool = False,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Plot the segmentations of a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing segmentation data.
        good_color (str, optional): The color for 'good' segments. Defaults to "green".
        bad_color (str, optional): The color for 'bad' segments. Defaults to "red".
        score_col (str, optional): The column name for the scores. Defaults to "score".
        segment_col (str, optional): The column name for the segments. Defaults to "segment".
        ascending (bool, optional): Whether to sort the segments in ascending order. Defaults to False.
        fig (Any, optional): An existing figure to which the segmentations will be added. Defaults to None.

    Returns:
        Any: The figure object with the segmentations plotted.

    """
    if fig is None:
        fig = go.Figure()

    # Sort the DataFrame based on the segment column
    df_sorted = df.sort_values(by=segment_col, ascending=ascending)

    # Add traces for each segment
    for segment in df_sorted[segment_col].unique():
        segment_data = df_sorted[df_sorted[segment_col] == segment]
        fig.add_trace(
            go.Scatter(
                x=segment_data.index,
                y=segment_data[score_col],
                mode="lines+markers",
                name=str(segment),
                line=dict(color=good_color if segment_data[score_col].mean() >= 0 else bad_color),
            )
        )

    return fig
