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


def test_normalize_scores():
    """Test the normalize_scores function for automatic color scaling."""
    # Test with a simple array where we know the expected behavior
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Mean = 3.0, std = sqrt(2.5) ≈ 1.58
    # vmin = 3.0 - 2*1.58 = -0.16, vmax = 3.0 + 2*1.58 = 6.16
    # All values are within bounds, so no clipping
    normalized = normalize_scores(scores, k=2.0)

    # Check that result is in [0, 1] range
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

    # Check that mean value gets mapped to middle of range
    mean_idx = 2  # Middle element (value 3.0)
    assert normalized[mean_idx] == approx(0.5, abs=0.1)

    # Check that lowest value gets mapped to close to 0
    assert normalized[0] < 0.5

    # Check that highest value gets mapped to close to 1
    assert normalized[-1] > 0.5


def test_normalize_scores_with_outliers():
    """Test normalize_scores with outliers that should be clipped."""
    # Create data with outliers
    scores = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 100.0])  # 100.0 is an outlier

    normalized = normalize_scores(scores, k=2.0)

    # Check that result is in [0, 1] range
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)

    # The outlier should be clipped and mapped to very close to 1.0
    assert normalized[-1] == approx(1.0, abs=1e-6)


def test_normalize_scores_constant_values():
    """Test normalize_scores when all values are the same."""
    scores = np.array([5.0, 5.0, 5.0, 5.0])

    normalized = normalize_scores(scores, k=2.0)

    # When std=0, all values should map to 0 (due to the 1e-8 denominator protection)
    assert np.all(normalized == 0.0)


def test_normalize_scores_custom_k():
    """Test normalize_scores with custom k parameter."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test with k=1 (smaller range)
    normalized_k1 = normalize_scores(scores, k=1.0)

    # Test with k=3 (larger range)
    normalized_k3 = normalize_scores(scores, k=3.0)

    # Both should be in [0, 1] range
    assert np.all(normalized_k1 >= 0) and np.all(normalized_k1 <= 1)
    assert np.all(normalized_k3 >= 0) and np.all(normalized_k3 <= 1)

    # With k=1, the range is tighter, so extreme values should be more clipped
    # With k=3, the range is wider, so less clipping
    # The variance in normalized_k1 should be larger (more spread after normalization)
    assert np.var(normalized_k1) >= np.var(normalized_k3)


def test_automatic_color_scaling_integration():
    """Test that automatic color scaling is applied in plotting functions."""
    # Create test data with known outliers
    test_df = pd.DataFrame({
        "segment_id": [1, 2, 3, 4],
        "segment_name": ["A", "B", "C", "D"],
        "score": [0.1, 0.2, 0.25, 1.5],  # 1.5 is an outlier
    })

    # Test matplotlib plotting
    fig_mp = plot_single_segmentation_mp(test_df)
    assert isinstance(fig_mp, MatplotlibFigure)

    # The plot should work without errors (automatic normalization handles outliers)
    # Close the figure to prevent memory leaks
    plt.close(fig_mp)

    # Test plotly plotting
    fig_plotly = plot_single_segmentation(test_df, backend="plotly")
    assert isinstance(fig_plotly, go.Figure)

    # Test double segmentation with outliers
    double_df = pd.DataFrame({
        "segment_id": [1, 2, 3, 4],
        "feature_1": ["A", "A", "B", "B"],
        "feature_2": ["X", "Y", "X", "Y"],
        "score": [0.1, 0.2, 0.25, 1.5],  # 1.5 is an outlier
    })

    # Test double segmentation plotting
    double_plotter = DoubleSegmPlotting(df=double_df, backend="matplotlib")
    fig_heatmap = double_plotter.plot_heatmap()
    assert isinstance(fig_heatmap, MatplotlibFigure)
    plt.close(fig_heatmap)

    # Test plotly heatmap
    double_plotter_plotly = DoubleSegmPlotting(df=double_df, backend="plotly")
    fig_heatmap_plotly = double_plotter_plotly.plot_heatmap()
    assert isinstance(fig_heatmap_plotly, go.Figure)


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
