"""Plotting utilities for tab-right plotting subpackage."""

import pandas as pd
import plotly.graph_objects as go  # type: ignore


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
