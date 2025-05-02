"""Protocol definitions for data segmentation analysis in tab-right.

This module defines protocol classes and type aliases for segmentation analysis,
including interfaces for segmentation calculations and feature-based segmentation.
"""

from dataclasses import dataclass
from typing import Callable, List, Protocol, runtime_checkable

import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import BaseDecisionTree


@runtime_checkable
@dataclass
class BaseSegmentationCalc(Protocol):
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


@runtime_checkable
@dataclass
class PredictionSegmentationCalc(Protocol):
    """Protocol for segmentation performance calculations using a single prediction column.

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


@runtime_checkable
@dataclass
class ProbabilitySegmentationCalc(Protocol):
    """Protocol for segmentation performance calculations using probability columns.

    Parameters
    ----------
    gdf : DataFrameGroupBy
        Grouped DataFrame, each group represents a segment.
    label_col : str
        Column name for the true target values.
    probability_cols : List[str]
        Column names for the predicted probability values.

    """

    gdf: DataFrameGroupBy
    label_col: str
    probability_cols: List[str]

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


# For backward compatibility, SegmentationCalc remains as the main protocol
# but now it's just a type alias for PredictionSegmentationCalc
SegmentationCalc = PredictionSegmentationCalc


@runtime_checkable
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
    decision_tree: BaseDecisionTree

    def fit(
        self,
    ) -> BaseDecisionTree:
        """Train the decision tree model.

        update the self.decision_tree with the fitted model.

        Returns
        -------
        BaseDecisionTree
            The trained decision tree model.

        """

    def groups(self, model: BaseDecisionTree) -> pd.DataFrame:
        """Get the groups of the DataFrame based on the decision tree model.

        Parameters
        ----------
        model : BaseDecisionTree
            Fitted decision tree model.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the groups defined by the decision tree model.
            columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `feature1`: (str) the range or category of the first feature.
            - `feature2`: (str) the range or category of the second feature.

        """

    def __call__(self, metric: Callable[[pd.Series, pd.Series], float]) -> pd.DataFrame:
        """Call method to apply the model to the DataFrame.

        This method fits (if needed) the tree model, groups by the tree model leaves, and use one of the
        SegmentationCalc

        Parameters
        ----------
        model : BaseDecisionTree
            The decision tree model to fit

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the groups defined by the decision tree model.
            columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `feature1`: (str) the range or category of the first feature.
            - `feature2`: (str) the range or category of the second feature.
            - `score`: The calculated error metric for the segment.

        """


# def find_segmentation_combinations(df: pd.DataFrame, features: list[str], error_col: str) -> list[FindSegmentation2F]:
#     """Find all combinations of two features in the DataFrame and return a list of FindSegmentation2F objects."""
find_segmentation_combinations = Callable[[pd.DataFrame, list[FindSegmentation2F], str], list[FindSegmentation2F]]
