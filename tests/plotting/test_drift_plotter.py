"""Tests for the DriftPlotter implementation."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tab_right.drift.drift_calculator import DriftCalculator
from tab_right.plotting.drift_plotter import DriftPlotter


@pytest.fixture
def sample_drift_calculator():
    """Provides a DriftCalculator instance with sample data."""
    df1 = pd.DataFrame({"numeric": np.random.normal(0, 1, 100), "category": np.random.choice(["A", "B"], 100)})
    df2 = pd.DataFrame({"numeric": np.random.normal(1, 1.2, 120), "category": np.random.choice(["B", "C"], 120)})
    return DriftCalculator(df1, df2)


def test_drift_plotter_init(sample_drift_calculator):
    """Test DriftPlotter initialization."""
    plotter = DriftPlotter(sample_drift_calculator)
    assert isinstance(plotter, DriftPlotter)
    assert plotter.drift_calc is sample_drift_calculator


def test_plot_multiple(sample_drift_calculator):
    """Test plot_multiple runs and returns a Figure."""
    plotter = DriftPlotter(sample_drift_calculator)
    fig = plotter.plot_multiple()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)  # Close the figure to prevent display during tests

    # Test with threshold
    fig_thresh = plotter.plot_multiple(threshold=0.1)
    assert isinstance(fig_thresh, plt.Figure)
    assert len(fig_thresh.axes[0].lines) > 0  # Check if threshold line was added
    plt.close(fig_thresh)

    # Test with top_n
    fig_top = plotter.plot_multiple(top_n=1)
    assert isinstance(fig_top, plt.Figure)
    assert len(fig_top.axes[0].get_yticks()) == 1  # Check if only one feature is plotted
    plt.close(fig_top)


def test_plot_single(sample_drift_calculator):
    """Test plot_single runs and returns a Figure for each type."""
    plotter = DriftPlotter(sample_drift_calculator)

    # Test numeric plot
    fig_num = plotter.plot_single("numeric")
    assert isinstance(fig_num, plt.Figure)
    assert "Continuous Distribution Comparison: numeric" in fig_num.axes[0].get_title()
    plt.close(fig_num)

    # Test categorical plot
    fig_cat = plotter.plot_single("category")
    assert isinstance(fig_cat, plt.Figure)
    assert "Categorical Distribution Comparison: category" in fig_cat.axes[0].get_title()
    plt.close(fig_cat)

    # Test with show_metrics=False
    fig_no_metrics = plotter.plot_single("numeric", show_metrics=False)
    assert isinstance(fig_no_metrics, plt.Figure)
    assert len(fig_no_metrics.axes[0].texts) == 0  # No metric text box expected
    plt.close(fig_no_metrics)


def test_get_distribution_plots(sample_drift_calculator):
    """Test get_distribution_plots runs and returns a dict of Figures."""
    plotter = DriftPlotter(sample_drift_calculator)
    plots = plotter.get_distribution_plots()

    assert isinstance(plots, dict)
    assert set(plots.keys()) == {"numeric", "category"}
    assert isinstance(plots["numeric"], plt.Figure)
    assert isinstance(plots["category"], plt.Figure)

    # Ensure figures were closed and not displayed
    assert len(plt.get_fignums()) == 0

    # Clean up any potentially remaining figures just in case
    plt.close("all")  # Force close all figures


def test_empty_dataframe():
    """Test DriftPlotter handles empty dataframes gracefully."""
    df1 = pd.DataFrame(columns=["numeric", "category"])
    df2 = pd.DataFrame(columns=["numeric", "category"])

    calculator = DriftCalculator(df1, df2)
    with pytest.raises(ValueError, match="Both dataframes must be non-empty."):
        DriftPlotter(calculator)


def test_invalid_column(sample_drift_calculator):
    """Test DriftPlotter raises errors for invalid column names."""
    plotter = DriftPlotter(sample_drift_calculator)

    with pytest.raises(ValueError):
        plotter.plot_single("non_existent_column")


def test_drift_plotter_static_plot_drift():
    """Test DriftPlotter static method plot_drift works correctly."""
    # Create a sample DataFrame with drift values
    drift_df = pd.DataFrame({"feature": ["feature1", "feature2", "feature3"], "value": [0.8, 0.5, 0.2]})

    # Test the static method
    fig = DriftPlotter.plot_drift(None, drift_df)

    # Verify the output is a plotly Figure
    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_drift_plotter_static_plot_drift_mp():
    """Test DriftPlotter static method plot_drift_mp works correctly."""
    # Create a sample DataFrame with drift values
    drift_df = pd.DataFrame({"feature": ["feature1", "feature2", "feature3"], "value": [0.8, 0.5, 0.2]})

    # Test the static method
    fig = DriftPlotter.plot_drift_mp(None, drift_df)

    # Verify the output is a matplotlib Figure
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_drift_plotter_static_plot_feature_drift():
    """Test DriftPlotter static method plot_feature_drift works correctly."""
    ref = pd.Series([1.0, 2.0, 2.5, 3.0, 4.0])
    cur = pd.Series([1.5, 2.2, 2.8, 3.5, 4.2])

    # Test the static method
    fig = DriftPlotter.plot_feature_drift(ref, cur, feature_name="test_feature")

    # Verify the output is a plotly Figure
    import plotly.graph_objects as go

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # Two KDE lines and two mean lines


def test_drift_plotter_static_plot_feature_drift_mp():
    """Test DriftPlotter static method plot_feature_drift_mp works correctly."""
    ref = pd.Series([1.0, 2.0, 2.5, 3.0, 4.0])
    cur = pd.Series([1.5, 2.2, 2.8, 3.5, 4.2])

    # Test the static method
    fig = DriftPlotter.plot_feature_drift_mp(ref, cur, feature_name="test_feature")

    # Verify the output is a matplotlib Figure
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0

    # Close the figure to prevent memory leaks
    plt.close(fig)


@dataclass
class DummyEmptyDensityDriftCalc:
    """Dummy DriftCalcP implementation that returns empty density DataFrame."""

    df1: pd.DataFrame
    df2: pd.DataFrame
    kind: Optional[Dict[str, str]] = None

    def __post_init__(self):
        # Set up feature types for the test
        self._feature_types = {"test_feature": "continuous"}

    def __call__(self, columns: Optional[Iterable[str]] = None, bins: int = 10, **kwargs: Mapping) -> pd.DataFrame:
        """Return basic drift metrics DataFrame."""
        return pd.DataFrame({"feature": ["test_feature"], "type": ["wasserstein"], "score": [0.5], "raw_score": [0.5]})

    def get_prob_density(self, columns: Optional[Iterable[str]] = None, bins: int = 10) -> pd.DataFrame:
        """Return empty DataFrame to simulate empty density case."""
        return pd.DataFrame(columns=["feature", "bin", "ref_density", "cur_density"])

    @classmethod
    def _categorical_drift_calc(cls, s1: pd.Series, s2: pd.Series) -> float:
        return 0.0

    @classmethod
    def _continuous_drift_calc(cls, s1: pd.Series, s2: pd.Series, bins: int = 10) -> float:
        return 0.0


def test_plot_single_with_empty_density():
    """Test that plot_single yields a valid matplotlib Figure when the density DataFrame is empty."""
    # Create non-empty DataFrames for the dummy calculator to pass DriftPlotter validation
    df1 = pd.DataFrame({"test_feature": [1, 2, 3]})
    df2 = pd.DataFrame({"test_feature": [2, 3, 4]})

    # Create dummy drift calculator that returns empty density
    dummy_calc = DummyEmptyDensityDriftCalc(df1, df2)

    # Create DriftPlotter with the dummy calculator
    plotter = DriftPlotter(dummy_calc)

    # Call plot_single - this should return an empty figure since density is empty
    fig = plotter.plot_single("test_feature")

    # Assert that the result is a valid matplotlib Figure
    assert isinstance(fig, plt.Figure)

    # Assert that the figure contains at least one axis
    assert len(fig.axes) >= 1

    # Assert that the axis contains the expected "No data available" message
    ax = fig.axes[0]
    texts = [text.get_text() for text in ax.texts]
    expected_message = "No data available for column 'test_feature'."
    assert any(expected_message in text for text in texts)

    # Close the figure to prevent memory leaks
    plt.close(fig)
