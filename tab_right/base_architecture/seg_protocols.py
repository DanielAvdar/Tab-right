"""Protocol definitions for data segmentation analysis in tab-right.

This module defines protocol classes and type aliases for segmentation analysis,
including interfaces for segmentation calculations and feature-based segmentation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Union, runtime_checkable

import pandas as pd
from pandas.api.typing import DataFrameGroupBy

MetricType = Callable[[pd.Series, pd.Series], pd.Series]
ScoreMetricType = Callable[[pd.Series, pd.Series], float]


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
    segment_names : Optional[Dict[int, Any]], default=None
        Optional mapping from an integer segment ID to the original group name
        (category, interval, or tuple). If provided, these IDs should match the
        grouping keys if gdf is grouped by integer IDs.

    """

    gdf: DataFrameGroupBy
    label_col: str
    prediction_col: Union[str, List[str]]
    segment_names: Optional[Dict[int, Any]] = None

    def _reduce_metric_results(
        self,
        results: Union[float, pd.Series],
    ) -> float:
        """Reduce the metric results to a single value, the metric produce series of values.

        if produce a single value, return it. it used for getting single value for each segment.

        Parameters
        ----------
        results : Union[float, pd.Series]
            The metric results to reduce.

        Returns
        -------
        float
            The reduced metric result.

        """

    def __call__(self, metric: Callable) -> pd.DataFrame:
        """Call method to apply the metric to each group in the DataFrameGroupBy object.

        Parameters
        ----------
        metric : Callable[[pd.Series, pd.Series], pd.Series]
            A function that takes two pandas Series (true and predicted values)
            and returns a float representing the error metric.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated error metrics for each segment.
            Expected columns:
            - `segment_id`: The ID of the segment (either the original group key or an assigned int).
            - `name`: The name of the segment (category or bin range string).
            - `score`: The avg error metric for each segment.

        """


@runtime_checkable
@dataclass
class DoubleSegmentation(Protocol):
    """Class schema for calculating double segmentation, segmentation based on two features.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing to analyze.
    label_col : str
        The name of the column containing the true target values.
    prediction_col : str
        The name of the column containing the predicted values.
        Can be probabilities (multiple columns) or classes or continuous values.

    """

    df: pd.DataFrame
    label_col: str
    prediction_col: str

    def _group_2_features(
        self,
        feature1: str,
        feature2: str,
        bins_1: int,
        bins_2: int,
    ) -> BaseSegmentationCalc:
        """Group the DataFrame by two features and returns a DataFrameGroupBy object.

        Parameters
        ----------
        feature1 : str
            The name of the first feature, which we want to find the segmentation for.
        feature2 : str
            The name of the second feature, which we want to find the segmentation for.
        bins_1 : int
            The number of bins to use for the first feature, if the feature is continuous.
        bins_2 : int
            The number of bins to use for the second feature, if the feature is continuous.

        Returns
        -------
        BaseSegmentationCalc
            A SegmentationCalc instance with grouped data.
            The DataFrame is grouped by the two features, and the segments are defined by the bins.

        """

    def __call__(
        self,
        feature1_col: str,
        feature2_col: str,
        score_metric: ScoreMetricType,
        bins_1: int,
        bins_2: int,
    ) -> pd.DataFrame:
        """Call method to apply grouping and scoring to the segment.

        Parameters
        ----------
        feature1_col : str
            The name of the first feature, which we want to find the segmentation for.
        feature2_col : str
            The name of the second feature, which we want to find the segmentation for.
        score_metric : ScoreMetricType
            A function that takes two pandas Series (true and predicted values)
            and returns a float representing the error metric.
        bins_1 : int, default=4
            The number of bins to use for the first feature, if the feature is continuous.
            ignore if the feature is categorical.
        bins_2 : int, default=4
            The number of bins to use for the second feature, if the feature is continuous.
            ignore if the feature is categorical.

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
