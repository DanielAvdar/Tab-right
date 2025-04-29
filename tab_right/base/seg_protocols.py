"""Protocol definitions for data segmentation analysis in tab-right.

This module defines protocol classes and type aliases for segmentation analysis,
including interfaces for segmentation calculations and feature-based segmentation.
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import BaseDecisionTree


@dataclass
class SegmentationCalc(Protocol):
    """Class schema for segmentation performance calculations.

    Parameters
    ----------
    gdf : DataFrameGroupBy
        Grouped DataFrame, each group represents a segment.
    label_col : str
        Column name for the true target values.
    prediction_col : str
        Column name for the predicted values.


    """

    gdf: DataFrameGroupBy
    label_col: str
    prediction_col: str

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


@dataclass
class FindSegmentation2F(Protocol):
    """Class schema for segmentation performance calculations.

    Parameters.
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be segmented.
    feature1_col : str
        Column name for the first feature to be used in segmentation.
    feature2_col : str
        Column name for the second feature to be used in segmentation.
    error_col : str
        Column name for the error metric to be calculated.

    """

    df: pd.DataFrame
    feature1_col: str
    feature2_col: str
    error_col: str

    def train_tree_model(self, model: BaseDecisionTree) -> BaseDecisionTree:
        """Train the decision tree model.

        Parameters.
        ----------
        model: BaseDecisionTree
            The decision tree model to be fitted, on the features to predict the error.

        Returns
        -------
        BaseDecisionTree
            The trained decision tree model.

        """

    def __call__(self, model: BaseDecisionTree) -> DataFrameGroupBy:
        """Call method to apply the model to the DataFrame.

        This method fits the tree model and produces a DataFrameGroupBy based on the
        decision tree of the fitted model.

        Parameters
        ----------
        model : BaseDecisionTree
            The decision tree model to fit



        """


# def find_segmentation_combinations(df: pd.DataFrame, features: list[str], error_col: str) -> list[FindSegmentation2F]:
#     """Find all combinations of two features in the DataFrame and return a list of FindSegmentation2F objects."""
find_segmentation_combinations = Callable[[pd.DataFrame, list[FindSegmentation2F], str], list[FindSegmentation2F]]
