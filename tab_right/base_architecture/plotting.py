from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import pandas as pd
from pandas.api.typing import DataFrameGroupBy


@runtime_checkable
@dataclass
class PlotSegmentation2F(Protocol):
    """Base protocol for segmentation performance calculations.

    Parameters
    ----------
    gdf : DataFrameGroupBy
        Grouped DataFrame, each group represents a segment.
    label_col : str
        Column name for the true target values.

    """

    gdf: DataFrameGroupBy
    label_col: str

    def __call__(self, metric: Callable[[pd.Series, pd.Series], float]) -> pd.DataFrame:
        """Call method to apply the metric to each group in the DataFrameGroupBy object.

        Parameters
        ----------
        metric : Callable[[pd.Series, pd.Series], float]
            A function that takes two pandas Series (true and predicted values)
            and returns a float representing the error metric.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated error metrics for each segment.
            with 2 main columns:
            - `segment_id`: The ID of the segment.
            - `score`: The calculated error metric for the segment.

        """
