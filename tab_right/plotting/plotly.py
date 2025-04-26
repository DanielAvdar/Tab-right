"""Plotting utilities for tab-right plotting subpackage."""

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go


def plot_segmentations(
    df: pd.DataFrame,
    good_color: str = "green",
    bad_color: str = "red",
    score_col: str = "score",
    segment_col: str = "segment",
    ascending: bool = False,
    fig: Any = None,
) -> Any:
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


def plot_segmentation_heatmap(
    z: pd.DataFrame,
    x_labels: Optional[list] = None,
    y_labels: Optional[list] = None,
    title: str = "Segmentation Heatmap",
    zmin: float = None,
    zmax: float = None,
    annotation_fmt: str = ".2f",
    colorbar_title: str = "Score",
    colorscale: str = "RdYlGn_r",
) -> go.Figure:
    """Generate a heatmap for the given matrix data.

    Args:
        z (pd.DataFrame): The matrix data for the heatmap.
        x_labels (Optional[list], optional): Labels for the x-axis. Defaults to None.
        y_labels (Optional[list], optional): Labels for the y-axis. Defaults to None.
        title (str, optional): The title of the heatmap. Defaults to "Segmentation Heatmap".
        zmin (float, optional): The minimum value for the color scale. Defaults to None.
        zmax (float, optional): The maximum value for the color scale. Defaults to None.
        annotation_fmt (str, optional): The format for annotations. Defaults to ".2f".
        colorbar_title (str, optional): The title for the color bar. Defaults to "Score".
        colorscale (str, optional): The colorscale to use. Defaults to "RdYlGn_r".

    Returns:
        go.Figure: A Plotly Figure object containing the heatmap.

    """
    if x_labels is None:
        x_labels = list(z.columns)
    if y_labels is None:
        y_labels = list(z.index)

    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=x_labels,
            y=y_labels,
            colorbar=dict(title=colorbar_title),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            hovertemplate=z.round(2).astype(str).ravel(),
            text=z.round(2).astype(str),
            texttemplate=annotation_fmt,
        )
    )

    fig.update_layout(title=title)
    return fig
