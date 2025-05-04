"""Module for double segmentation implementation."""

from dataclasses import dataclass
from typing import List, Union

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from tab_right.base_architecture.seg_protocols import (
    FindSegmentation,
    MetricType,
    ScoreMetricType,
)
from tab_right.segmentations.calc_seg import SegmentationCalc


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
        # Create column names and explicitly convert to pandas Index
        column_names: List[str] = ["segment_id", "feature_1", "score_1", "feature_2", "score_2"]
        combined.columns = pd.Index(column_names)
        combined["score"] = combined[["score_1", "score_2"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        return combined

    def _group_by_segment(
        self,
        df: pd.DataFrame,
        seg: pd.Series,
    ) -> SegmentationCalc:
        """Group DataFrame by segment ID and create a SegmentationCalc instance.

        Args:
            df: DataFrame to group
            seg: Segment IDs to group by

        Returns:
            SegmentationCalc: A SegmentationCalc instance with grouped data

        """
        grouped_df = df.groupby(seg)
        # Ensure the prediction_col is compatible with the protocol
        prediction_col: Union[str, List[str]] = self.segmentation_finder.prediction_col

        return SegmentationCalc(
            gdf=grouped_df,
            label_col=self.segmentation_finder.label_col,
            prediction_col=prediction_col,
        )

    def __call__(
        self,
        feature1_col: str,
        feature2_col: str,
        error_func: MetricType,
        model: DecisionTreeRegressor,
        score_metric: ScoreMetricType,
    ) -> pd.DataFrame:
        """Perform double segmentation on two features.

        Args:
            feature1_col: Name of the first feature column
            feature2_col: Name of the second feature column
            error_func: Function to calculate error between true and predicted values for each datapoint
            model: Decision tree regressor model to use for segmentation
            score_metric: Function to calculate overall score for segments

        Returns:
            pd.DataFrame: Combined segmentation results with scores

        """
        seg1 = self.segmentation_finder(feature1_col, error_func, model)
        seg2 = self.segmentation_finder(feature2_col, error_func, model)
        combined = self._combine_2_features(seg1, seg2)
        df = self.segmentation_finder.df
        seg_calc = self._group_by_segment(df, combined["segment_id"])

        result_df = seg_calc(score_metric)

        # Merge the scores back into the combined DataFrame
        combined = pd.merge(
            combined, result_df[["segment_id", "score"]], on="segment_id", how="left", suffixes=("_old", "")
        )

        # Drop the old score column if it exists
        if "score_old" in combined.columns:
            combined = combined.drop(columns=["score_old"])

        return combined
