"""Protocol definitions for data segmentation analysis in tab-right.

This module defines protocol classes and type aliases for segmentation analysis,
including interfaces for segmentation calculations and feature-based segmentation.
"""

from dataclasses import dataclass
from typing import Callable, List, Protocol, Union, runtime_checkable

import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from sklearn.tree import BaseDecisionTree
import abc

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
class FindSegmentation(Protocol):
    """Class schema for find feature segmentation, by using a decision tree.

    Parameters.
    ----------
    df : pd.DataFrame
        DataFrame containing the data to be segmented.
    label_col : str
        Column name for the true target values.
    prediction_col : Union[str, List[str]]

    """

    df: pd.DataFrame
    label_col: str
    prediction_col: Union[str, List[str]]

    def __call__(
        self,
        feature_col: str,
        error_metric: Callable[[pd.Series, pd.DataFrame], pd.Series],
        model: BaseDecisionTree,
    ) -> pd.DataFrame:
        """Call method to apply the model to the DataFrame.

        This method fits (if needed) the tree model, groups by the tree model leaves, and use one of the
        SegmentationCalc

        Parameters
        ----------
        feature_col : str
            The name of the feature, which we want to find the segmentation for.
        error_metric : Callable[[pd.Series, pd.DataFrame], pd.Series]
            A function that takes a pandas Series (true values) and a DataFrame (predicted values)
            and returns a Series representing the error metric for each row in the DataFrame.
        model : BaseDecisionTree
            The decision tree model to fit

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the groups defined by the decision tree model.
            columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `segment_name`: (str) the range or category of the first feature.
            - `score`: (float) The calculated error metric for the segment.

        """

    def _calc_error(
        metric: Callable[[pd.Series, pd.DataFrame], pd.Series],
        y_true: pd.Series,
        y_pred: pd.DataFrame,
    ) -> pd.Series:
        """Calculate the error metric for each group in the DataFrame.

        Parameters
        ----------
        metric : Callable[[pd.Series, pd.DataFrame], pd.Series]
            A function that takes a pandas Series (true values) and a DataFrame (predicted values)
            and returns a Series representing the error metric for each row in the DataFrame.
        y_true : pd.Series
            The true target values.
        y_pred : pd.DataFrame
            The predicted values for each group, can be probabilities (multiple columns)
             or classes or continuous values.

        """
    def _fit_model(
        model: BaseDecisionTree,
        feature: pd.Series,
        error: pd.Series,
    ) -> BaseDecisionTree:
        """Fit the decision tree model to the feature and error data.

        Parameters
        ----------
        model : BaseDecisionTree
            The decision tree model to fit.
        feature : pd.Series
            The feature data to use for fitting the model.
        error : pd.Series
            The error calculated for each row in the DataFrame, which is used as the target variable.

        Returns
        -------
        BaseDecisionTree
            The fitted decision tree model.

        """

    def _extract_leaves(
        model: BaseDecisionTree,
    ) -> pd.DataFrame:
        """Extract the leaves of the fitted decision tree model.

        Parameters
        ----------
        model : BaseDecisionTree
            The fitted decision tree model to extract leaves from.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the leaves of the decision tree.
            columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `segment_name`: (str) the range or category of the first feature.

        """
