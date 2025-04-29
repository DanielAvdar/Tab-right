"""Segmentation statistics utilities for tab-right package."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import pandas as pd
from pandas.api.typing import DataFrameGroupBy


@dataclass
class SegmentationStats:
    """SegmentationStats provides vectorized segmentation and scoring for tabular data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    label_col : str or list of str
        The label column(s) to use for scoring.
    prediction_col : str, optional
        The prediction column to use for scoring. Can also be provided as 'pred_col'
        for backward compatibility.
    feature : str
        The feature column to segment by.
    metric : Callable
        The metric function to use for scoring.
    is_categorical : bool, default False
        Whether to treat the feature as categorical (True) or continuous (False).
    pred_col : str, optional
        Deprecated alias for prediction_col.

    """

    # All required parameters with no defaults first
    df: pd.DataFrame
    label_col: Union[str, List[str]]
    feature: str
    metric: Callable
    # Parameters with defaults
    prediction_col: Optional[str] = None
    is_categorical: bool = False
    # We'll create the gdf in methods as needed instead of storing as instance variable
    _gdf: Optional[DataFrameGroupBy] = field(default=None, repr=False)

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: Union[str, List[str]],
        feature: str,
        metric: Callable,
        prediction_col: Optional[str] = None,
        is_categorical: bool = False,
        pred_col: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the SegmentationStats class.

        This custom init method allows us to handle both prediction_col and pred_col for backward compatibility.
        """
        self.df = df
        self.label_col = label_col
        self.feature = feature
        self.metric = metric
        self.is_categorical = is_categorical
        self._gdf = None

        # Handle the prediction_col and pred_col parameters
        if prediction_col is not None:
            self.prediction_col = prediction_col
        elif pred_col is not None:
            self.prediction_col = pred_col
        else:
            self.prediction_col = None

    # Property to make the class compatible with the SegmentationCalc protocol
    @property
    def gdf(self) -> DataFrameGroupBy:
        """Get the grouped DataFrame."""
        if self._gdf is None:
            # Create a default grouping if none exists yet
            df = self._add_segments_column()
            self._gdf = df.groupby("_segment")
        return self._gdf

    @gdf.setter
    def gdf(self, value: DataFrameGroupBy) -> None:
        """Set the grouped DataFrame."""
        self._gdf = value

    def _prepare_segments(self, bins: int = 10) -> pd.Series:
        """Prepare segments from the feature column.

        Parameters
        ----------
        bins : int, default=10
            Number of bins for continuous segmentation.

        Returns
        -------
        pd.Series
            Series of segment values for each row.

        """
        if self.is_categorical:
            return self.df[self.feature]
        return pd.qcut(self.df[self.feature], q=bins, duplicates="drop")

    def _add_segments_column(self, bins: int = 10) -> pd.DataFrame:
        df = self.df.copy()
        df["_segment"] = self._prepare_segments(bins)
        return df

    def _run_probability_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        self.gdf = df.groupby("_segment")
        prob_means = self.gdf[self.label_col].mean()
        prob_means = prob_means.reset_index().rename(columns={"_segment": "segment"})
        prob_means["score"] = prob_means[self.label_col].apply(lambda row: row.to_dict(), axis=1)
        return prob_means[["segment", "score"]]

    def _run_metric_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        self.gdf = df.groupby("_segment")

        def score_func(group: pd.DataFrame) -> float:
            return float(self.metric(group[self.label_col], group[self.prediction_col]))

        scores = self.gdf.apply(score_func)
        return pd.DataFrame({"segment": scores.index, "score": scores.values})

    def run(self, bins: int = 10) -> pd.DataFrame:
        """Segment the data and compute scores for each segment.

        Parameters
        ----------
        bins : int, default 10
            Number of bins for continuous segmentation.

        Returns
        -------
        pd.DataFrame
            DataFrame with segment and score columns.

        """
        df = self._add_segments_column(bins)
        if isinstance(self.label_col, list):
            return self._run_probability_mode(df)
        return self._run_metric_mode(df)

    def __call__(self, metric: Callable[[pd.Series, pd.Series], float] = None) -> pd.DataFrame:
        """Apply the metric to each group in the DataFrameGroupBy object.

        Parameters
        ----------
        metric : Callable[[pd.Series, pd.Series], float], optional
            A function that takes two pandas Series and returns a float.
            If None, uses the metric provided at initialization.

        Returns
        -------
        pd.DataFrame
            DataFrame with segment and score columns.

        """
        if metric is not None:
            self.metric = metric

        # Default to 10 bins for continuous features
        bins = 10
        return self.run(bins=bins)

    def check(self) -> None:
        """Check for NaN and probability sum errors in the label columns.

        Raises
        ------
        ValueError
            If NaN or invalid probability sums are found.

        """
        if isinstance(self.label_col, list):
            if self.df[self.label_col].isnull().values.any():
                raise ValueError("Probability columns contain NaN values.")
            prob_sums = self.df[self.label_col].sum(axis=1)
            if not ((prob_sums - 1).abs() < 1e-6).all():
                raise ValueError("Probabilities in label columns do not sum to 1 for all rows.")
        else:
            if self.df[self.label_col].isnull().any():
                raise ValueError("Label column contains NaN values.")
