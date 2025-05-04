"""Module for double segmentation implementation."""

from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype

from tab_right.base_architecture.seg_protocols import BaseSegmentationCalc, ScoreMetricType
from tab_right.segmentations.calc_seg import SegmentationCalc


@dataclass
class DoubleSegmentationImp:
    """Implementation of double segmentation logic based on two features.

    Conforms to the DoubleSegmentation protocol.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to segment.
    label_col : str
        The name of the column containing the true target values.
    prediction_col : Union[str, List[str]]
        The name(s) of the column(s) containing the predicted values.

    """

    df: pd.DataFrame
    label_col: str
    prediction_col: Union[str, List[str]]

    def _group_2_features(
        self,
        feature1_col: str,
        feature2_col: str,
        bins_1: int,
        bins_2: int,
    ) -> BaseSegmentationCalc:
        """Group the DataFrame by two features.

        Handles discretization for numeric features based on specified bins.

        Parameters
        ----------
        feature1_col : str
            Name of the first feature column.
        feature2_col : str
            Name of the second feature column.
        bins_1 : int
            Number of bins for the first feature (if numeric).
        bins_2 : int
            Number of bins for the second feature (if numeric).

        Returns
        -------
        BaseSegmentationCalc
            An instance ready to calculate metrics on the grouped data.

        """
        temp_df = self.df.copy()
        group_cols = []

        # Handle feature 1 binning/grouping
        if is_numeric_dtype(temp_df[feature1_col]):
            temp_df[f"{feature1_col}_binned"] = pd.cut(temp_df[feature1_col], bins=bins_1, include_lowest=True)
            group_cols.append(f"{feature1_col}_binned")
        else:
            group_cols.append(feature1_col)

        # Handle feature 2 binning/grouping
        if is_numeric_dtype(temp_df[feature2_col]):
            temp_df[f"{feature2_col}_binned"] = pd.cut(temp_df[feature2_col], bins=bins_2, include_lowest=True)
            group_cols.append(f"{feature2_col}_binned")
        else:
            group_cols.append(feature2_col)

        # Group by the (potentially binned) features
        gdf = temp_df.groupby(group_cols, observed=False)

        # Store group names for later use in __call__
        self._group_names = {i: name for i, name in enumerate(gdf.groups.keys())}
        self._group_cols = group_cols  # Store which columns were used for grouping

        # Assign segment IDs based on group enumeration
        segment_map = {name: i for i, name in enumerate(gdf.groups.keys())}

        def get_segment_id(row: pd.Series) -> int:
            key: Tuple[Any, ...] = tuple(row[col] for col in group_cols)
            return segment_map.get(key, -1)  # Assign -1 if key not found (shouldn't happen with observed=False)

        temp_df["segment_id"] = temp_df.apply(get_segment_id, axis=1)

        # Regroup with segment_id to pass to SegmentationCalc
        final_gdf = temp_df.groupby("segment_id", observed=False)

        return SegmentationCalc(gdf=final_gdf, label_col=self.label_col, prediction_col=self.prediction_col)

    def __call__(
        self,
        feature1_col: str,
        feature2_col: str,
        score_metric: ScoreMetricType,
        bins_1: int = 4,
        bins_2: int = 4,
    ) -> pd.DataFrame:
        """Perform double segmentation and calculate scores.

        Parameters
        ----------
        feature1_col : str
            Name of the first feature column.
        feature2_col : str
            Name of the second feature column.
        score_metric : ScoreMetricType
            Metric function to calculate segment scores.
        bins_1 : int, default=4
            Number of bins for the first feature (if numeric).
        bins_2 : int, default=4
            Number of bins for the second feature (if numeric).

        Returns
        -------
        pd.DataFrame
            DataFrame with segment details and scores.
            Columns: segment_id, feature_1, feature_2, score.

        """
        calc_instance = self._group_2_features(feature1_col, feature2_col, bins_1, bins_2)
        scores_df = calc_instance(score_metric)  # Calculate scores per segment_id

        # Map segment_id back to feature names/bins
        feature_info = []
        for seg_id, group_key in self._group_names.items():
            # Ensure group_key is always a tuple, even if only one grouping column was used (shouldn't happen here)
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            f1_val = group_key[0]
            f2_val = group_key[1] if len(group_key) > 1 else None  # Handle potential edge case

            # Convert intervals to strings for readability
            if isinstance(f1_val, pd.Interval):
                f1_val = str(f1_val)
            if isinstance(f2_val, pd.Interval):
                f2_val = str(f2_val)

            feature_info.append({"segment_id": seg_id, "feature_1": f1_val, "feature_2": f2_val})

        feature_df = pd.DataFrame(feature_info)

        # Merge scores with feature information
        result_df = pd.merge(feature_df, scores_df, on="segment_id")

        # Ensure correct column order
        return result_df[["segment_id", "feature_1", "feature_2", "score"]]
