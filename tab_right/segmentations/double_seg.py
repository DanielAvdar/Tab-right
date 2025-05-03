"""Module for double segmentation implementation."""

from typing import Callable

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from tab_right.base_architecture.seg_protocols import DoubleSegmentation, FindSegmentation


class DoubleSegmentationImp(DoubleSegmentation):
    """Implementation of double segmentation logic."""

    def __init__(self, segmentation_finder: FindSegmentation):
        """Initialize the double segmentation implementation."""
        self.segmentation_finder = segmentation_finder

    @classmethod
    def _combine_2_features(cls, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
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
        error_metric: Callable[[pd.Series, pd.DataFrame], pd.Series],
        model: DecisionTreeRegressor,
    ) -> pd.DataFrame:
        seg1 = self.segmentation_finder(feature1_col, error_metric, model)
        seg2 = self.segmentation_finder(feature2_col, error_metric, model)
        combined = self._combine_2_features(seg1, seg2)
        return combined
