"""Module for plotting segmentations of a DataFrame."""

import pandas as pd
import plotly.graph_objects as go


def plot_segmentations(
    df: pd.DataFrame,
    good_color: str = "green",
    bad_color: str = "red",
    score_col: str = "score",
    segment_col: str = "segment",
    ascending: bool = False,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Plot the segmentations of a given DataFrame as a bar chart.

    Args:
        df (pd.DataFrame): The input data.
        good_color (str, optional): Color for good segments. Defaults to "green".
        bad_color (str, optional): Color for bad segments. Defaults to "red".
        score_col (str, optional): Column name for scores. Defaults to "score".
        segment_col (str, optional): Column name for segments. Defaults to "segment".
        ascending (bool, optional): Sort order. Defaults to False.
        fig (go.Figure, optional): Existing figure to update. Defaults to None.

    Returns:
        go.Figure: A plotly figure object.

    """
    if fig is None:
        fig = go.Figure()
    df = df.sort_values(by=score_col, ascending=ascending)
    for segment, group in df.groupby(segment_col):
        color = good_color if group[score_col].iloc[0] >= 0 else bad_color
        fig.add_trace(
            go.Bar(
                x=group.index,
                y=group[score_col],
                name=str(segment),
                marker_color=color,
            )
        )
    fig.update_layout(barmode="group")
    return fig
