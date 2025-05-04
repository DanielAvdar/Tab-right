"""Module for calculating segmentation metrics."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union  # Standard library

import pandas as pd  # Third-party
from pandas.api.typing import DataFrameGroupBy  # Third-party


@dataclass
class SegmentationCalc:
    """Implementation of BaseSegmentationCalc protocol.

    Calculates scores for pre-defined segments.

    Attributes
    ----------
    gdf : DataFrameGroupBy
        Grouped DataFrame, each group represents a segment (grouped by segment_id).
    label_col : str
        Column name for the true target values.
    prediction_col : Union[str, List[str]]
        Column names for the predicted values.
    segment_names : Dict[int, Any]
        Mapping from segment_id to the original group name (category or interval).

    """

    gdf: DataFrameGroupBy
    label_col: str
    prediction_col: Union[str, List[str]]
    segment_names: Dict[int, Any]

    def _reduce_metric_results(
        self,
        results: Union[float, pd.Series],
    ) -> float:
        """Reduce the metric results to a single value if the metric produces a series.

        If it produces a single value, return it. Used for getting a single value for each segment.

        Parameters
        ----------
        results : Union[float, pd.Series]
            The metric results to reduce.

        Returns
        -------
        float
            The reduced metric result.

        """
        if isinstance(results, pd.Series):
            return float(results.mean())
        return float(results)

    def __call__(self, metric: Callable) -> pd.DataFrame:
        """Apply the metric to each group and return scores with segment names.

        Ensures all segments defined in `segment_names` are included in the output,
        assigning NaN to segments with no data.

        Parameters
        ----------
        metric : Callable
            Metric function to apply.

        Returns
        -------
        pd.DataFrame
            DataFrame with segment_id, name, and score.

        """
        # Initialize results with all segment IDs from segment_names, default score NaN
        results = {segment_id: float("nan") for segment_id in self.segment_names}

        # Calculate scores for segments present in the grouped data
        for name, group in self.gdf:
            segment_id = int(name)  # Ensure name is treated as the segment_id (integer)
            if segment_id in results:  # Process only if the segment is expected
                if not group.empty:
                    y_true = group[self.label_col]
                    y_pred = group[self.prediction_col]
                    score = metric(y_true, y_pred)
                    results[segment_id] = self._reduce_metric_results(score)
                # If group is empty but segment_id is in results, it keeps the NaN score

        # Convert results dictionary to DataFrame
        scores_df = pd.DataFrame(list(results.items()), columns=["segment_id", "score"])

        # Add the segment names using the stored mapping
        scores_df["name"] = scores_df["segment_id"].map(self.segment_names)

        # Convert interval names to strings for consistent output
        scores_df["name"] = scores_df["name"].apply(lambda x: str(x) if isinstance(x, pd.Interval) else x)

        # Reorder columns and ensure correct order even if some segments were empty
        scores_df = scores_df.sort_values("segment_id").reset_index(drop=True)
        return scores_df[["segment_id", "name", "score"]]
