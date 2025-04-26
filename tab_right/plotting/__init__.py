"""Plotting subpackage for tab-right: provides utilities for visualizing segmentations and drift."""

from typing import Any

import pandas as pd
import plotly.graph_objects as go  # type: ignore


def plot_segmentations(
    df: pd.DataFrame,
    good_color: str = "green",
    bad_color: str = "red",
    score_col: str = "score",
    segment_col: str = "segment",
    ascending: bool = False,
    fig: Any = None,
) -> Any:
    """Plot segmentation DataFrame (segment, score) with green for best performance and red for worst using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'segment' and 'score' columns.
    good_color : str, default 'green'
        Color for the best performing segment.
    bad_color : str, default 'red'
        Color for the worst performing segment.
    score_col : str, default 'score'
        Name of the score column.
    segment_col : str, default 'segment'
        Name of the segment column.
    ascending : bool, default False
        If True, lower scores are better (e.g., for loss). If False, higher scores are better (e.g., for accuracy).
    fig : plotly.graph_objects.Figure, optional
        Plotly Figure to plot on. If None, a new figure is created.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure with the plot.

    """
    df_sorted = df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    n = len(df_sorted)
    colors = [bad_color] + ["gray"] * (n - 2) + [good_color] if n > 1 else [good_color]
    if ascending:
        colors = colors[::-1]
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_sorted[segment_col].astype(str),
            y=df_sorted[score_col],
            marker_color=colors,
        )
    )
    fig.update_layout(
        xaxis_title="Segment",
        yaxis_title="Score",
        title="Segmentation Performance by Segment",
        bargap=0.2,
    )
    return fig
