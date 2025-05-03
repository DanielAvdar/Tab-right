"""Module for defining plotting protocols."""

import pandas as pd
from plotly.graph_objects import Figure


def plot_double_segmentation(
    df: pd.DataFrame,
) -> Figure:
    """Plot the double segmentation of a given DataFrame as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the groups defined by the decision tree model.
        columns:
        - `segment_id`: The ID of the segment, for grouping.
        - `feature_1`: (str) the range or category of the first feature.
        - `feature_2`: (str) the range or category of the second feature.
        - `score`: (float) The calculated error metric for the segment.

    Returns
    -------
    Figure
        A heatmap showing each segment with its corresponding avg score, col of heatmap is feature_1, row is feature_2.

    """


def plot_single_segmentation(
    df: pd.DataFrame,
) -> Figure:
    """Plot the single segmentation of a given DataFrame as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the groups defined by the decision tree model.
        columns:
        - `segment_id`: The ID of the segment, for grouping.
        - `feature_1`: (str) the range or category of the first feature.
        - `score`: (float) The calculated error metric for the segment.

    Returns
    -------
    Figure
        A bar chart showing each segment with its corresponding avg score, col of bar chart is
        feature_1, y axis is score.

    """
