import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass

from tab_right.base_architecture.model_comparison_protocols import PredictionCalculationP

from .model_comparison_check import CheckPredictionCalculation


@dataclass
class DummyPredictionCalculation:
    """Dummy implementation of PredictionCalculationP for testing."""

    df: pd.DataFrame
    label_col: str

    def __call__(self, pred_data, error_func=None):
        """Simple implementation that calculates pointwise errors."""
        result_df = self.df.copy()

        # Default error function (absolute error)
        if error_func is None:
            def error_func(y_true, y_pred):
                return abs(y_true - y_pred)

        # Get label values
        y_true = self.df[self.label_col]

        # Calculate errors for each prediction
        for i, pred_series in enumerate(pred_data):
            error_col = f"pred_{i}_error"
            result_df[error_col] = error_func(y_true, pred_series)

        return result_df


@pytest.fixture
def instance_to_check() -> PredictionCalculationP:
    """Provides an instance of PredictionCalculationP for protocol testing."""
    # Create a simple test DataFrame
    df = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.choice(["A", "B", "C"], 20),
        "label": np.random.rand(20)
    })
    return DummyPredictionCalculation(df, "label")


class TestPredictionCalculation(CheckPredictionCalculation):
    """Test class for `PredictionCalculationP` protocol compliance."""

    # Note: instance_to_check pytest fixture is implemented above