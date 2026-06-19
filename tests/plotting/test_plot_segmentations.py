"""Tests for the plot_segmentations module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure as MatplotlibFigure
from pytest import approx

from tab_right.plotting.plot_segmentations import (
    DoubleSegmPlotting,
    normalize_scores,
    plot_single_segmentation,
    plot_single_segmentation_mp,
)


@pytest.fixture
def single_segmentation_df():
    """Create a sample DataFrame for testing single segmentation plotting."""
    return pd.DataFrame({
        "segment_id": [1, 2, 3, 4],
        "segment_name": ["A <= 10", "10 < A <= 20", "20 < A <= 30", "A > 30"],
        "score": [0.1, 0.25, 0.15, 0.3],
    })


@pytest.fixture
def double_segmentation_df():
    """Create a sample DataFrame for testing double segmentation plotting."""
    return pd.DataFrame({
        "segment_id": [1, 2, 3, 4, 5, 6],
        "feature_1": ["A <= 10", "A <= 10", "10 < A <= 20", "10 < A <= 20", "A > 20", "A > 20"],
        "feature_2": ["B <= 5", "B > 5", "B <= 5", "B > 5", "B <= 5", "B > 5"],
        "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })


def test_plot_single_segmentation(single_segmentation_df):
    """Test that plot_single_segmentation creates a valid Plotly figure."""
    fig = plot_single_segmentation(single_segmentation_df)

    # Check that we got a plotly figure
    assert isinstance(fig, go.Figure)

    # Check that the figure contains a bar trace
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)

    # Check that the data was correctly passed to the plot
    assert len(fig.data[0].x) == len(single_segmentation_df)
    assert list(fig.data[0].y) == approx(list(single_segmentation_df["score"]))


def test_plot_single_segmentation_mp(single_segmentation_df):
    """Test that plot_single_segmentation_mp creates a valid Matplotlib figure."""
    fig = plot_single_segmentation_mp(single_segmentation_df)

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # Check that labels and title are set
    ax = fig.axes[0]
    assert "Segmentation Analysis" in ax.get_title()
    assert ax.get_xlabel() == "Feature Segments"
    assert ax.get_ylabel() == "Error Score"

    # Check for value labels (text annotations on the plot)
    [child for child in ax.get_children() if isinstance(child, plt.Text)]

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_plot_single_segmentation_mp_lower_is_better_false(single_segmentation_df):
    """Test plot_single_segmentation_mp with lower_is_better=False."""
    fig = plot_single_segmentation_mp(single_segmentation_df, lower_is_better=False)

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_double_segm_plotting_mp(double_segmentation_df):
    """Test that DoubleSegmPlotting with matplotlib backend creates a valid Matplotlib figure."""
    double_plotter = DoubleSegmPlotting(df=double_segmentation_df, backend="matplotlib")
    fig = double_plotter.plot_heatmap()

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Check that figure has at least one axis
    assert len(fig.axes) > 0

    # The main axis should have a pcolormesh for the heatmap and a colorbar
    ax = fig.axes[0]
    mesh_collections = [child for child in ax.get_children() if "QuadMesh" in str(type(child))]
    assert len(mesh_collections) == 1  # Should have exactly one heatmap

    # Check that title and labels are set
    assert "Double Segmentation Heatmap" in ax.get_title()
    assert ax.get_xlabel() == "Feature 1"
    assert ax.get_ylabel() == "Feature 2"

    # Check we have value annotations
    texts = [child for child in ax.get_children() if isinstance(child, plt.Text)]

    # Given our 3x2 heatmap (3 values for feature_1, 2 for feature_2), should have 6 cells
    # But some may be NaN, so just check we have some text elements
    assert len(texts) > 0

    # We should have a colorbar (usually the second axis in the figure)
    assert len(fig.axes) >= 2

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_double_segm_plotting_mp_lower_is_better_false(double_segmentation_df):
    """Test DoubleSegmPlotting with matplotlib backend and lower_is_better=False."""
    double_plotter = DoubleSegmPlotting(df=double_segmentation_df, lower_is_better=False, backend="matplotlib")
    fig = double_plotter.plot_heatmap()

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Close the figure to prevent memory leaks
    plt.close(fig)


def test_double_segm_plotting_mp_custom_metric(double_segmentation_df):
    """Test DoubleSegmPlotting with matplotlib backend and a custom metric name."""
    # Create a copy with a renamed score column
    df_custom = double_segmentation_df.rename(columns={"score": "custom_metric"})

    double_plotter = DoubleSegmPlotting(df=df_custom, metric_name="custom_metric", backend="matplotlib")
    fig = double_plotter.plot_heatmap()

    # Check that we got a matplotlib figure
    assert isinstance(fig, MatplotlibFigure)

    # Close the figure to prevent memory leaks
    plt.close(fig)


# Tests for the new scaling functionality
def test_normalize_scores_minmax():
    """Test normalize_scores with minmax method."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    normalized = normalize_scores(scores, method="minmax")

    # Should be normalized to [0, 1] range
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert len(normalized) == len(scores)


def test_normalize_scores_std():
    """Test normalize_scores with std method."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    normalized = normalize_scores(scores, method="std", k=2)

    # Should be normalized to [0, 1] range
    assert 0.0 <= normalized.min() <= 1.0
    assert 0.0 <= normalized.max() <= 1.0
    assert len(normalized) == len(scores)


def test_normalize_scores_std_with_outliers():
    """Test normalize_scores with std method handling outliers."""
    # Create data with outliers
    scores = np.array([0.1, 0.2, 0.3, 0.4, 2.0])  # 2.0 is an outlier
    normalized = normalize_scores(scores, method="std", k=2)

    # Outlier should be clipped close to 1.0 (or exactly 1.0 if it exceeds the upper bound)
    assert normalized[-1] >= 0.95  # Allow some tolerance
    assert len(normalized) == len(scores)


def test_normalize_scores_zero_variance():
    """Test normalize_scores with zero variance data."""
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    normalized = normalize_scores(scores, method="std")

    # Should return zeros for zero variance
    assert np.all(normalized == 0.0)


def test_normalize_scores_invalid_method():
    """Test normalize_scores with invalid method."""
    scores = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="Unknown method"):
        normalize_scores(scores, method="invalid")


def test_plot_single_segmentation_scaling_methods(single_segmentation_df):
    """Test plot_single_segmentation with different scaling methods."""
    # Test with minmax scaling
    fig_minmax = plot_single_segmentation(single_segmentation_df, scaling_method="minmax", backend="plotly")
    assert isinstance(fig_minmax, go.Figure)

    # Test with std scaling
    fig_std = plot_single_segmentation(single_segmentation_df, scaling_method="std", backend="plotly")
    assert isinstance(fig_std, go.Figure)


def test_plot_single_segmentation_mp_scaling_methods(single_segmentation_df):
    """Test plot_single_segmentation_mp with different scaling methods."""
    # Test with matplotlib backend
    fig_mp_minmax = plot_single_segmentation_mp(single_segmentation_df, scaling_method="minmax")
    assert isinstance(fig_mp_minmax, MatplotlibFigure)
    plt.close(fig_mp_minmax)

    fig_mp_std = plot_single_segmentation_mp(single_segmentation_df, scaling_method="std")
    assert isinstance(fig_mp_std, MatplotlibFigure)
    plt.close(fig_mp_std)


def test_double_segmentation_scaling(double_segmentation_df):
    """Test DoubleSegmPlotting with scaling methods."""
    # Test with std scaling
    plotter_std = DoubleSegmPlotting(df=double_segmentation_df, scaling_method="std", backend="plotly")
    fig_std = plotter_std.plot_heatmap()
    assert isinstance(fig_std, go.Figure)

    # Test with minmax scaling
    plotter_minmax = DoubleSegmPlotting(df=double_segmentation_df, scaling_method="minmax", backend="matplotlib")
    fig_minmax = plotter_minmax.plot_heatmap()
    assert isinstance(fig_minmax, MatplotlibFigure)
    plt.close(fig_minmax)
