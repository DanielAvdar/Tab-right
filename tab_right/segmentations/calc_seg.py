"""Module for calculating segmentation metrics."""

from typing import Callable

import pandas as pd

from tab_right.base_architecture.seg_protocols import BaseSegmentationCalc


class BaseSegmentationCalcImp(BaseSegmentationCalc):
    """Implementation of base segmentation calculations."""

    def __init__(self, gdf: pd.core.groupby.DataFrameGroupBy, label_col: str, prediction_col: str):
        """Initialize the base segmentation calculation implementation."""
        self.gdf = gdf
        self.label_col = label_col
        self.prediction_col = prediction_col

    def __call__(self, metric: Callable[[pd.Series, pd.Series], float]) -> pd.DataFrame:
        """Perform the segmentation calculation using the provided metric."""
        results = []
        for segment_id, group in self.gdf:
            score = metric(group[self.label_col], group[self.prediction_col])
            results.append({"segment_id": segment_id, "score": score})
        return pd.DataFrame(results)
