"""Tests for the DriftPlotter implementation."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests
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
