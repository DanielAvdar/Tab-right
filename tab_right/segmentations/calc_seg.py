"""Module for calculating segmentation metrics."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TypeVar, Union

import pandas as pd
from pandas.api.typing import DataFrameGroupBy

T = TypeVar("T")


@dataclass
class SegmentationCalc:
    """Implementation of base segmentation calculations."""

    gdf: DataFrameGroupBy
    label_col: str
    prediction_col: Union[str, List[str]]

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

    # Define a helper method to handle both string and list cases
    def _get_prediction_data(self, group: pd.DataFrame) -> Any:
        """Get prediction data from a group based on prediction_col.

        This abstracts away the type complexity for mypy.

        Args:
            group: DataFrame group to extract prediction data from

        Returns:
            Any: The prediction data (either Series or DataFrame)

        """
        return group[self.prediction_col]

    def __call__(self, metric: Callable[[pd.Series, Any], pd.Series]) -> pd.DataFrame:
        """Perform the segmentation calculation using the provided metric.

        Args:
            metric: A callable that takes label Series and prediction data and returns a pd.Series metric.

        Returns:
            pd.DataFrame: A dataframe containing segment IDs and their corresponding metric scores.

        """
        results: List[Dict[str, Union[str, int, float]]] = []
        for segment_id, group in self.gdf:
            y_true = group[self.label_col]
            # Use helper method to abstract away type complexity
            y_pred = self._get_prediction_data(group)

            # Apply the metric and convert result to a pd.Series if it's not already
            score = metric(y_true, y_pred)
            score_value = self._reduce_metric_results(score)

            results.append({"segment_id": segment_id, "score": score_value})

        return pd.DataFrame(results)
