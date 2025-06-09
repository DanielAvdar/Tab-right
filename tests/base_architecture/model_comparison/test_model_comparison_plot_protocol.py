import numpy as np
import pandas as pd
import pytest
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure

from tab_right.base_architecture.model_comparison_protocols import PredictionCalculationP
from tab_right.base_architecture.model_comparison_plot_protocols import ModelComparisonPlotP

from .model_comparison_plot_check import CheckModelComparisonPlot


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


@dataclass
class DummyModelComparisonPlot:
    """Dummy implementation of ModelComparisonPlotP for testing."""

    comparison_calc: PredictionCalculationP

    def plot_error_distribution(
        self,
        pred_data: List[pd.Series],
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        bins: int = 30,
        **kwargs: Any,
    ):
        """Dummy implementation - returns None for protocol testing."""
        return None

    def plot_pairwise_comparison(
        self,
        pred_data: List[pd.Series],
        model_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        **kwargs: Any,
    ):
        """Dummy implementation - returns None for protocol testing."""
        return None

    def plot_model_performance_summary(
        self,
        pred_data: List[pd.Series],
        model_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        **kwargs: Any,
    ):
        """Dummy implementation - returns None for protocol testing."""
        return None


@pytest.fixture
def instance_to_check() -> ModelComparisonPlotP:
    """Provides an instance of ModelComparisonPlotP for protocol testing."""
    # Create a simple test DataFrame
    df = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.choice(["A", "B", "C"], 20),
        "label": np.random.rand(20)
    })
    comparison_calc = DummyPredictionCalculation(df, "label")
    return DummyModelComparisonPlot(comparison_calc)


class TestModelComparisonPlot(CheckModelComparisonPlot):
    """Test class for `ModelComparisonPlotP` protocol compliance."""

    # Note: instance_to_check pytest fixture is implemented above