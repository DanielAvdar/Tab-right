"""Module for calculating segmentation metrics."""

from dataclasses import dataclass
from typing import Callable, Union

import pandas as pd


@dataclass
class SegmentationCalc:
    """Implementation of base segmentation calculations."""

    gdf: pd.core.groupby.DataFrameGroupBy
    label_col: str
    prediction_col: Union[str, list]

    def _reduce_metric_results(
        self,
        results: Union[float, pd.Series],
    ) -> float:
        """Reduce metric results to a single value.
        
        Args:
            results: Metric results to reduce
            
        Returns:
            float: Reduced metric result
        """
        if isinstance(results, pd.Series):
            return results.mean()
        return float(results)

    def __call__(self, metric: Callable[[pd.Series, pd.Series], float]) -> pd.DataFrame:
        """Perform the segmentation calculation using the provided metric.

        Args:
            metric: A callable that takes two pd.Series (labels and predictions) and returns a float metric.

        Returns:
            pd.DataFrame: A dataframe containing segment IDs and their corresponding metric scores.

        """
        results = []
        for segment_id, group in self.gdf:
            score = metric(group[self.label_col], group[self.prediction_col])
            results.append({"segment_id": segment_id, "score": self._reduce_metric_results(score)})
        return pd.DataFrame(results)
