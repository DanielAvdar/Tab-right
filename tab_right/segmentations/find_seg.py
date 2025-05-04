"""Module for finding segmentations."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.tree import BaseDecisionTree


@dataclass
class FindSegmentationImp:
    """Test class for double segmentation."""

    df: pd.DataFrame
    label_col: str
    prediction_col: str  # Changed from Union[str, List[str]] to str only

    def __post_init__(self) -> None:
        """Post-initialization logic for `FindSegmentationImp`."""
        super().__init__()

    @classmethod
    def _calc_error(
        cls,
        metric: Callable[[pd.Series, pd.Series], pd.Series],
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> pd.Series:
        """Calculate error using the provided metric.

        Args:
            metric: Function to calculate error between true and predicted values
            y_true: True values
            y_pred: Predicted values

        Returns:
            pd.Series: Error values for each data point

        """
        error = metric(y_true, y_pred)

        return error

    @classmethod
    def _fit_model(
        cls,
        model: BaseDecisionTree,
        feature: pd.Series,
        error: pd.Series,
    ) -> BaseDecisionTree:
        """Fit decision tree model on feature and error data.

        Args:
            model: Decision tree model to fit
            feature: Feature values
            error: Error values for each data point

        Returns:
            BaseDecisionTree: Fitted model

        """
        # Convert to numpy array first to handle both numpy arrays and pandas ExtensionArrays
        feature_array = np.asarray(feature.values)
        model.fit(feature_array.reshape(-1, 1), error)
        return model

    @classmethod
    def _extract_leaves(
        cls,
        model: BaseDecisionTree,
    ) -> pd.DataFrame:
        """Extract leaf nodes from fitted model.

        Args:
            model: Fitted decision tree model

        Returns:
            pd.DataFrame: DataFrame with segment information

        """
        # Extract leaf indices and map them to segment IDs
        leaf_indices = model.apply(model.tree_.value.reshape(-1, 1))
        unique_leaves = pd.unique(leaf_indices)
        return pd.DataFrame({
            "segment_id": range(len(unique_leaves)),
            "segment_name": unique_leaves,
            "score": [0.0] * len(unique_leaves),
        })

    def __call__(
        self,
        feature_col: str,
        error_func: Callable[[pd.Series, pd.Series], pd.Series],
        model: BaseDecisionTree,
    ) -> pd.DataFrame:
        """Find segmentations based on feature and error metric.

        Args:
            feature_col: Name of the feature column to use for segmentation
            error_func: Function to calculate error between true and predicted values
            model: Decision tree model to use for segmentation

        Returns:
            pd.DataFrame: Segmentation results containing segment IDs, names and scores

        """
        feature = self.df[feature_col]
        y_true = self.df[self.label_col]
        y_pred = self.df[self.prediction_col]  # Now only handling a single string column

        error = self._calc_error(error_func, y_true, y_pred)
        fitted_model = self._fit_model(model, feature, error)
        return self._extract_leaves(fitted_model)
