"""Module for double segmentation implementation."""

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from tab_right.base_architecture.seg_protocols import FindSegmentation, ScoreMetricType


@dataclass
class DoubleSegmentationImp:
    """Implementation of double segmentation logic."""

    segmentation_finder: FindSegmentation

    @classmethod
    def _combine_2_features(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Combine two feature DataFrames.

        Args:
            df1: First feature DataFrame
            df2: Second feature DataFrame

        Returns:
            pd.DataFrame: Combined features with scores

        """
        combined = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        combined = combined.iloc[:, :5]  # Ensure the combined DataFrame has exactly 5 columns
        combined.columns = ["segment_id", "feature_1", "score_1", "feature_2", "score_2"]
        combined["score"] = combined[["score_1", "score_2"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        return combined

    @classmethod
    def _group_by_segment(
        cls,
        df: pd.DataFrame,
        seg: pd.Series,
    ) -> pd.core.groupby.DataFrameGroupBy:
        return df.groupby(seg)

    def __call__(
        self,
        feature1_col: str,
        feature2_col: str,
        error_func: Callable[[pd.Series, pd.Series], pd.Series],
        model: DecisionTreeRegressor,
        score_metric: ScoreMetricType = None,
    ) -> pd.DataFrame:
        """Perform double segmentation on two features.

        Args:
            feature1_col: Name of the first feature column
            feature2_col: Name of the second feature column
            error_func: Function to calculate error between true and predicted values for each datapoint
            model: Decision tree regressor model to use for segmentation
            score_metric: Optional function to calculate overall score for datapoints, returns a float

        Returns:
            pd.DataFrame: Combined segmentation results with scores

        """
        seg1 = self.segmentation_finder(feature1_col, error_func, model)
        seg2 = self.segmentation_finder(feature2_col, error_func, model)
        combined = self._combine_2_features(seg1, seg2)
        return combined
