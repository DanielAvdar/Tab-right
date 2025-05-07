"""Module for defining plotting protocols."""

from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

import pandas as pd
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure

Figure = Union[PlotlyFigure, MatplotlibFigure]


@runtime_checkable
@dataclass
class DoubleSegmPlottingP(Protocol):
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

    def plot_heatmap(self) -> Figure:
        """Plot the double segmentation of a given DataFrame as a heatmap.

        Returns
        -------
        Figure
            A heatmap showing each segment with its corresponding avg score,
            from get_heatmap_df() method. Colors are determined by the lower_is_better parameter:
            - If lower_is_better=True: Lower values are green (better), higher values are red (worse)
            - If lower_is_better=False: Higher values are green (better), lower values are red (worse)

        """
