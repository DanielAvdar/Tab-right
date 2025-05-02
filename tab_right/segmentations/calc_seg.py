"""Segmentation statistics utilities for tab-right package."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import pandas as pd
from pandas.api.typing import DataFrameGroupBy

from tab_right.base_architecture.seg_protocols import BaseSegmentationCalc


@dataclass
class SegmentationStats(BaseSegmentationCalc):
    """Segmentation statistics for tabular data implementing BaseSegmentationCalc protocol.

    Parameters
    ----------
    gdf : DataFrameGroupBy
        Grouped DataFrame, each group represents a segment.
    label_col : str
        Column name for the true target values.
    prediction_col : Union[str, List[str]]
        Column names for the predicted values. Can be a single column or a list of columns.
        Can be probabilities (multiple columns) or classes or continuous values.
    feature : str, optional
        The feature column used for segmentation.
    is_categorical : bool, default False
        Whether to treat the feature as categorical (True) or continuous (False).

    """

    # Required parameters from BaseSegmentationCalc protocol
    gdf: DataFrameGroupBy
    label_col: Union[str, List[str]]
    prediction_col: Union[str, List[str]]

    # Additional parameters specific to this implementation
    feature: Optional[str] = None
    is_categorical: bool = False
    metric: Optional[Callable[[pd.Series, pd.Series], float]] = None

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        label_col: Optional[Union[str, List[str]]] = None,
        feature: Optional[str] = None,
        metric: Optional[Callable[[pd.Series, pd.Series], float]] = None,
        prediction_col: Optional[Union[str, List[str]]] = None,
        is_categorical: bool = False,
        pred_col: Optional[Union[str, List[str]]] = None,
        gdf: Optional[DataFrameGroupBy] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SegmentationStats class.

        This custom init method allows various initialization approaches:
        1. With a DataFrame and feature (will create groupby internally)
        2. With a pre-created DataFrameGroupBy object
        3. With backward compatibility for pred_col
        """
        # Handle the prediction_col and pred_col parameters for backward compatibility
        if prediction_col is not None:
            self.prediction_col = prediction_col
        elif pred_col is not None:
            self.prediction_col = pred_col
        else:
            self.prediction_col = None

        # Set other parameters
        self.feature = feature
        self.is_categorical = is_categorical
        self.metric = metric

        # Handle different initialization approaches
        if gdf is not None:
            # Direct initialization with a groupby object
            self.gdf = gdf
            if label_col is not None:
                self.label_col = label_col
            else:
                # Try to infer label_col from the dataframe
                raise ValueError("label_col must be provided when initializing with a DataFrameGroupBy object")
        elif df is not None and label_col is not None and feature is not None:
            # Initialize from a dataframe, creating the groupby internally
            self.label_col = label_col

            # Create groupby if needed
            segments = self._prepare_segments(df, feature)
            df_with_segments = df.copy()
            df_with_segments["_segment"] = segments
            self.gdf = df_with_segments.groupby("_segment")
        else:
            raise ValueError("Either (df, label_col, feature) or (gdf, label_col) must be provided")

    def _prepare_segments(self, df: pd.DataFrame, feature: str, bins: int = 10) -> pd.Series:
        """Prepare segments from the feature column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        feature : str
            Feature column to segment by.
        bins : int, default=10
            Number of bins for continuous segmentation.

        Returns
        -------
        pd.Series
            Series of segment values for each row.

        """
        if self.is_categorical:
            return df[feature]
        return pd.qcut(df[feature], q=bins, duplicates="drop")

    def __call__(self, metric: Optional[Callable[[pd.Series, pd.Series], float]] = None) -> pd.DataFrame:
        """Apply the metric to each group in the DataFrameGroupBy object.

        Parameters
        ----------
        metric : Callable[[pd.Series, pd.Series], float], optional
            A function that takes two pandas Series (true and predicted values)
            and returns a float representing the error metric.
            If None, uses the metric provided at initialization.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated error metrics for each segment.
            with 2 main columns:
            - `segment_id`: The ID of the segment.
            - `score`: The calculated error metric for the segment.

        """
        if metric is not None:
            self.metric = metric

        if self.metric is None:
            raise ValueError("Metric function must be provided either at initialization or when calling")

        if isinstance(self.label_col, list):
            return self._run_probability_mode()
        return self._run_metric_mode()

    def _run_probability_mode(self) -> pd.DataFrame:
        """Run in probability mode for multi-class classification.

        Returns
        -------
        pd.DataFrame
            DataFrame with segment and score columns.

        """
        prob_means = self.gdf[self.label_col].mean()
        prob_means = prob_means.reset_index().rename(columns={"_segment": "segment_id"})
        prob_means["score"] = prob_means[self.label_col].apply(lambda row: row.to_dict(), axis=1)
        return prob_means[["segment_id", "score"]]

    def _run_metric_mode(self) -> pd.DataFrame:
        """Run in metric mode for regression or binary classification.

        Returns
        -------
        pd.DataFrame
            DataFrame with segment and score columns.

        """

        def score_func(group: pd.DataFrame) -> float:
            # Extract values from columns to match the metric function signature
            labels_data = group[self.label_col]
            preds_data = group[self.prediction_col]

            # Convert to Series if needed
            labels_series: pd.Series = labels_data if isinstance(labels_data, pd.Series) else labels_data.iloc[:, 0]
            preds_series: pd.Series = preds_data if isinstance(preds_data, pd.Series) else preds_data.iloc[:, 0]

            return float(self.metric(labels_series, preds_series))

        scores = self.gdf.apply(score_func)
        return pd.DataFrame({"segment_id": scores.index, "score": scores.values})

    def check(self) -> None:
        """Check for NaN and probability sum errors in the label columns.

        Raises
        ------
        ValueError
            If NaN or invalid probability sums are found.

        """
        df = self.gdf.obj  # Get the original DataFrame from the GroupBy object

        if isinstance(self.label_col, list):
            if df[self.label_col].isnull().values.any():
                raise ValueError("Probability columns contain NaN values.")
            prob_sums = df[self.label_col].sum(axis=1)
            if not ((prob_sums - 1).abs() < 1e-6).all():
                raise ValueError("Probabilities in label columns do not sum to 1 for all rows.")
        else:
            if df[self.label_col].isnull().any():
                raise ValueError("Label column contains NaN values.")
