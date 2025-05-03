"""Module for defining plotting protocols."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd
from plotly.graph_objects import Figure


@runtime_checkable
@dataclass
class DoubleSegmPlotting(Protocol):
    """Class schema for double segmentation plotting.

    This class is used to define the interface for plotting double segmentations.
    It includes the DataFrames to be plotted and the kind of plot to be created.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the groups defined by the decision tree model.
        columns:
        - `segment_id`: The ID of the segment, for grouping.
        - `feature_1`: (str) the range or category of the first feature.
        - `feature_2`: (str) the range or category of the second feature.
        - `score`: (float) The calculated error metric for the segment.
    metric_name : str, default="score"
        The name of the metric column in the DataFrame.
    lower_is_better : bool, default=True
        Whether lower values of the metric indicate better performance.
        Affects the color scale in visualizations (green for better, red for worse).

    """

    df: pd.DataFrame
    metric_name: str = "score"
    lower_is_better: bool = True

    def get_heatmap_df(self) -> pd.DataFrame:
        """Get the DataFrame for the heatmap. from the double segmentation df.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the groups defined by the decision tree model.
            columns: feature_1 ranges or categories
            index: feature_2 ranges or categories
            content: The calculated error metric for the segment.

        """

    def plotly_heatmap(self) -> Figure:
        """Plot the double segmentation of a given DataFrame as a heatmap.

        Returns
        -------
        Figure
            A heatmap showing each segment with its corresponding avg score,
            from get_heatmap_df() method. Colors are determined by the lower_is_better parameter:
            - If lower_is_better=True: Lower values are green (better), higher values are red (worse)
            - If lower_is_better=False: Higher values are green (better), lower values are red (worse)

        """


def plot_single_segmentation(
    df: pd.DataFrame,
    lower_is_better: bool = True,
) -> Figure:
    """Plot the single segmentation of a given DataFrame as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the groups defined by the decision tree model.
        columns:
        - `segment_id`: The ID of the segment, for grouping.
        - `segment_name`: (str) the range or category of the feature.
        - `score`: (float) The calculated error metric for the segment.
    lower_is_better : bool, default=True
        Whether lower values of the metric indicate better performance.
        Affects the color scale in visualizations (green for better, red for worse).

    Returns
    -------
    Figure
        A bar chart showing each segment with its corresponding avg score, x-axis represents
        the feature segments (segment_name), and y-axis shows the score. Colors are determined 
        by the lower_is_better parameter:
        - If lower_is_better=True: Lower values are green (better), higher values are red (worse)
        - If lower_is_better=False: Higher values are green (better), lower values are red (worse)

    """
