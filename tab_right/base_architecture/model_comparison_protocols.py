"""Protocol definitions for model comparison analysis in tab-right.

This module defines protocol classes for model comparison analysis,
including interfaces for prediction calculation and comparison between multiple models.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
@dataclass
class PredictionCalculationP(Protocol):
    """Protocol for prediction calculation implementations.

    This protocol defines the interface for calculating pointwise errors
    between multiple sets of predictions and true labels.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the data for analysis.
    label_col : str
        Column name for the true target values.

    """

    df: pd.DataFrame
    label_col: str

    def __call__(
        self,
        pred_data: List[pd.Series],
        error_func: Optional[Callable[[pd.Series, pd.Series], pd.Series]] = None,
    ) -> pd.DataFrame:
        """Calculate pointwise errors for multiple prediction series against the label.

        Parameters
        ----------
        pred_data : List[pd.Series]
            List of prediction Series to compare against the label.
            Each Series should have the same index as the DataFrame.
        error_func : Optional[Callable[[pd.Series, pd.Series], pd.Series]], default None
            Function for calculating pairwise error between predictions and labels.
            If None, defaults to a standard metric (e.g., absolute error for regression,
            0/1 loss for classification).
            Function signature: error_func(y_true, y_pred) -> pd.Series

        Returns
        -------
        pd.DataFrame
            DataFrame containing pointwise errors for each prediction series.
            Expected columns:
            - Original DataFrame columns (as context)
            - `{label_col}`: The true label values
            - `pred_0_error`, `pred_1_error`, ...: Pointwise errors for each prediction series
            - `model_id`: Optional identifier for the prediction series

        """
