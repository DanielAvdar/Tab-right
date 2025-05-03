"""Protocol definitions for data segmentation analysis in tab-right.

This module defines protocol classes and type aliases for segmentation analysis,
including interfaces for segmentation calculations and feature-based segmentation.
"""

from dataclasses import dataclass
from typing import Callable, List, Protocol, Union, runtime_checkable

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
    prediction_col : Union[str, List[str]]
        Column names for the predicted values. Can be a single column or a list of columns.
        Can be probabilities (multiple columns) or classes or continuous values.

    """

    gdf: DataFrameGroupBy
    label_col: str
    prediction_col: Union[str, List[str]]

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
            - `score`: The avg error metric for the segment.

        """


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

    @classmethod
    def _calc_error(
        cls,
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

    @classmethod
    def _fit_model(
        cls,
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

    @classmethod
    def _extract_leaves(
        cls,
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


@runtime_checkable
@dataclass
class DoubleSegmentation(Protocol):
    """Class schema for find feature segmentation, by using a decision tree.

    Parameters.
    ----------
    segmentation_finder : FindSegmentation
        The segmentation finder object to use for finding segmentations.

    """

    segmentation_finder: FindSegmentation

    @classmethod
    def _combine_2_features(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Combine two DataFrames by concatenating them along the columns.

        Parameters
        ----------
        df1 : pd.DataFrame
            The first DataFrame to combine.
        df2 : pd.DataFrame
            The second DataFrame to combine.

        Description
        -----------
        The DataFrames containing segmentation information, which are
        columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `segment_name`: (str) the range or category of the first feature.
            - `score`: (Optional[float]) The calculated error metric for the segment.


        -------
        pd.DataFrame
        columns:
            - `segment_id`: The ID of the segment, for grouping.
            - `feature_1`: (str) the range or category of the first feature.
            - `feature_2`: (str) the range or category of the second feature.
            - `score`: (Optional[float]) score if available in one of the DataFrames.

        """

    @classmethod
    def _group_by_segment(
        cls,
        df: pd.DataFrame,
        seg: pd.Series,
    ) -> DataFrameGroupBy:
        """Group the DataFrame by segment ID and calculate the mean score for each group.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to group.
        seg : pd.Series
            The segment ID to group by.


        Returns
        -------
        DataFrameGroupBy
            A DataFrameGroupBy object containing the grouped DataFrame.

        """

    def __call__(
        self,
        feature1_col: str,
        feature2_col: str,
        error_metric: Callable[[pd.Series, pd.DataFrame], pd.Series],
        model: BaseDecisionTree,
    ) -> pd.DataFrame:
        """Call method to apply the model to the DataFrame.

        This method fits (if needed) the tree model using two features, groups by the tree model leaves,
        and use one of the SegmentationCalc.

        Parameters
        ----------
        feature1_col : str
            The name of the first feature, which we want to find the segmentation for.
        feature2_col : str
            The name of the second feature, which we want to find the segmentation for.
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
            - `feature_1`: (str) the range or category of the first feature.
            - `feature_2`: (str) the range or category of the second feature.
            - `score`: (float) The calculated error metric for the segment.

        """
